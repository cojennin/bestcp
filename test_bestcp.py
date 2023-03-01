from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import logging
import sys


from composer import Trainer
from composer.models import mnist_model
from composer.loggers import InMemoryLogger, Logger
from composer.callbacks import CheckpointSaver
from composer.utils import (parse_uri, maybe_create_remote_uploader_downloader_from_uri)
from pathlib import Path
from typing import Callable, Optional, Union
from composer.core import (Event, State, Time)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

class BestCheckpointSaver(CheckpointSaver):
    def __init__(
        self,
        metric_name='metrics/eval/Accuracy',
        maximize=True,
        save_folder: Optional[str] = None,
        save_filename: str = 'best-rank{rank}.pt',
        save_latest_filename: Optional[str] = 'latest-rank{rank}.pt',
        save_overwrite: bool = False,
        save_interval: Union[str, int, Time, Callable[[State, Event], bool]] = '1ep',
        save_weights_only: bool = False,
        save_num_checkpoints_to_keep: int = -1,
    ):
        latest_remote_file_name = None
        if save_folder is not None:
            _, _, parsed_save_folder = parse_uri(save_folder)

            # If user passes a URI with s3:// and a bucket_name, but no other
            # path then we assume they just want their checkpoints saved directly in their
            # bucket.
            if parsed_save_folder == '':
                folder = '.'
                remote_file_name = save_filename
                latest_remote_file_name = save_latest_filename

            # If they actually specify a path, then we use that for their local save path
            # and we prefix save_filename with that path for remote_file_name.
            else:
                folder = parsed_save_folder
                remote_file_name = str(Path(parsed_save_folder) / Path(save_filename))
                if save_latest_filename is not None:
                    latest_remote_file_name = str(Path(parsed_save_folder) / Path(save_latest_filename))
                else:
                    latest_remote_file_name = None

        print(folder, save_filename, remote_file_name,)
        super().__init__(
            folder=folder,
            filename=save_filename,
            remote_file_name=remote_file_name,
            latest_filename=save_latest_filename,
            latest_remote_file_name=latest_remote_file_name,
            overwrite=save_overwrite,
            weights_only=save_weights_only,
            save_interval=save_interval,
            num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        )

        self.save_folder = save_folder
        self.metric_name = metric_name
        self.current_best = None
        self.maximize = maximize
    
    def init(self, state: State, logger: Logger) -> None:
        if self.save_folder is not None:
            remote_ud = maybe_create_remote_uploader_downloader_from_uri(self.save_folder, logger.destinations)
            remote_ud.init(state, logger)
            if remote_ud is not None:
                logger.destinations += (remote_ud,)


    def _save_checkpoint(self, state, logger):
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
            print("SAVING BEST")
            self.current_best = current_metric_value
            super()._save_checkpoint(state, logger)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=128)
    eval_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=128)


    bcps = BestCheckpointSaver(save_folder='s3://mosaic-checkpoints/my-run-name/checkpoints', save_interval="1ba", save_overwrite=True)
    in_mem_logger = InMemoryLogger()
    trainer = Trainer(
        model=mnist_model(num_classes=10),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        max_duration="10ba",
        eval_interval='1ba',
        callbacks=[bcps],
        loggers=[in_mem_logger]
        
    )
    trainer.fit()


