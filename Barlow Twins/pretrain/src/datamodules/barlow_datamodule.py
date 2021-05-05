from typing import Optional, Sequence

from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from src.transforms.barlow_transforms import Transform
from src.datasets.barlow_dataset import CheXpertDataset, ImageClefDataset


class BARLOWDataModule(LightningDataModule):
    """
    
    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.data_dir = kwargs["data_dir"]
        self.batch_size = kwargs["batch_size"]
        self.num_workers = kwargs["num_workers"]
        self.label_list = kwargs['label_list']
        self.pin_memory = kwargs['pin_memory']

        self.dataset = kwargs["dataset"]

      
        if kwargs["dataset"] == "CheXpert":
            self.data_train = CheXpertDataset(
                directory=self.data_dir+"/CheXpert",
                split='train',
                transform= Transform(),
                label_list=self.label_list,
                )

            self.data_val: Optional[Dataset] = CheXpertDataset(
                directory=self.data_dir,
                split='val',
                transform= Transform(),
                label_list=self.label_list,
                )

        elif kwargs["dataset"] == "ImageClef":
            self.data_train = ImageClefDataset(
                directory=self.data_dir,
                transform= Transform(),
                )

            self.data_val = None




    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory= self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    # def test_dataloader(self):
    #     return DataLoader(
    #         dataset=self.data_test,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #     )
