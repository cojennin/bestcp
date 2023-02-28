from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from composer import Trainer
from composer.models import mnist_model
from composer.loggers import InMemoryLogger
from composer.callbacks import CheckpointSaver
import os 
import tempfile
from composer.utils import is_model_deepspeed
from composer.utils.file_helpers import create_symlink_file, format_name_with_dist, format_name_with_dist_and_time, is_tar
from composer.loggers import ObjectStoreLogger, Logger
from composer.utils.object_store import S3ObjectStore

class BestCheckpointSaver(CheckpointSaver):
    def __init__(
        self,
        metric_name='metrics/eval/Accuracy',
        maximize = True,
        folder = '{run_name}/checkpoints',
        filename = 'ep{epoch}-ba{batch}-rank{rank}.pt',
        artifact_name = '{run_name}/checkpoints/ep{epoch}-ba{batch}-rank{rank}',
        latest_filename = 'latest-rank{rank}.pt',
        latest_artifact_name = '{run_name}/checkpoints/latest-rank{rank}',
        save_interval = '1ep',
        overwrite = False,
        num_checkpoints_to_keep = -1,
        weights_only = False,
        best_filename = 'best-rank{rank}.pt',
        best_artifact_name = '{run_name}/checkpoints/best-rank{rank}',
    ):
        super().__init__(folder,
                        filename,
                        artifact_name,
                        latest_filename,
                        latest_artifact_name,
                        save_interval,
                        overwrite=overwrite,
                        num_checkpoints_to_keep=num_checkpoints_to_keep,
                        weights_only=weights_only)

        self.best_filename = best_filename
        self.best_artifact_name = best_artifact_name
        self.metric_name = metric_name
        self.current_best = None
        self.maximize = maximize

    def _save_checkpoint(self, state, logger, log_level):
        super()._save_checkpoint(state, logger, log_level)
        
        in_mem_loggers = [logger_destination for logger_destination in logger.destinations 
                if isinstance(logger_destination, InMemoryLogger)]
        if in_mem_loggers:
            in_mem_logger = in_mem_loggers[0]
        else:
            raise ValueError('You must specify an InMemoryLogger to use BestCheckpointSaver!')
            
        
        current_metric_value = in_mem_logger.get_timeseries(self.metric_name)[self.metric_name][-1]
        if self.current_best is None:
            self.current_best = current_metric_value
        if self.maximize:
            is_current_metric_best = current_metric_value >= self.current_best
        else:
            is_current_metric_best = current_metric_value <= self.current_best

        if is_current_metric_best:
            self.current_best = current_metric_value
            self._save_best_checkpoint(state, logger, log_level)


    def _save_best_checkpoint(self, state, logger: Logger, log_level):
        formatted_folder_path = format_name_with_dist(self.folder, state.run_name)
        symlink_name = os.path.join(
            formatted_folder_path,
            format_name_with_dist_and_time(
                self.best_filename,
                state.run_name,
                state.timestamp,
            ).lstrip('/'),
        )
        if is_model_deepspeed(state.model) and not is_tar(symlink_name):
            # Deepspeed requires tarballs; appending `.tar`
            symlink_name += '.tar'
        symlink_dirname = os.path.dirname(symlink_name)
        if symlink_dirname:
            os.makedirs(symlink_dirname, exist_ok=True)
        try:
            os.remove(symlink_name)
        except FileNotFoundError:
            pass
        checkpoint_filepath = os.path.join(format_name_with_dist(self.folder, state.run_name), self.filename)
        relative_checkpoint_path = os.path.relpath(checkpoint_filepath, formatted_folder_path)
        os.symlink(relative_checkpoint_path, symlink_name)
        if self.artifact_name is not None and self.best_artifact_name is not None:
            symlink_artifact_name = format_name_with_dist_and_time(self.best_artifact_name, state.run_name,
                                                                    state.timestamp).lstrip('/') + '.symlink'
            artifact_name = format_name_with_dist_and_time(self.artifact_name, state.run_name,
                                                            state.timestamp).lstrip('/')
            with tempfile.TemporaryDirectory() as tmpdir:
                symlink_filename = os.path.join(tmpdir, 'best.symlink')
                create_symlink_file(artifact_name, symlink_filename)
                logger.file_artifact(
                    log_level=log_level,
                    artifact_name=symlink_artifact_name,
                    file_path=symlink_filename,
                    overwrite=True,
                )

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=128)
    eval_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128)


    osl = ObjectStoreLogger(object_store_cls=S3ObjectStore, object_store_kwargs={'bucket':'evan-mosaic-test'})
    bcps = BestCheckpointSaver(folder='./cps', save_interval="1ba", overwrite=True)
    in_mem_logger = InMemoryLogger()
    trainer = Trainer(
        model=mnist_model(num_classes=10),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration="3ba",
        eval_interval='1ba',
        callbacks=[bcps],
        loggers=[in_mem_logger, osl]
        
    )
    trainer.fit()


