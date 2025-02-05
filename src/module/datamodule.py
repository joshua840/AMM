import lightning.pytorch as pl
from torch.utils.data import DataLoader

from .dataset import ImageNet


class DataModule(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
        data_dir: str = "~/Data",
        num_workers: int = 4,
        batch_size_train: int = 32,
        batch_size_test: int = 100,
        **kwargs,
    ):
        """
        Args:
            dataset: dataset name
            data_dir: directory of dataset
            num_workers: number of workers
            batch_size_train: batchsize of data loaders
            batch_size_test: batchsize of data loaders
        """
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage=None):
        Dataset = {
            "imagenet": ImageNet,
        }[self.hparams.dataset]

        dataset = Dataset(data_dir=self.hparams.data_dir)

        self.train_dataset = dataset.get_train_dataset()
        self.val_dataset = dataset.get_val_dataset()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
