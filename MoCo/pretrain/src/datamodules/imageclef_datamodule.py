from typing import Optional, Sequence

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from src.datamodules.imageclef import ImageclefDataset, TwoImageDataset
from src.transforms.transforms import (
    AddGaussianNoise,
    Compose,
    HistogramNormalize,
    #RandomGaussianBlur,
    TensorToRGB,
)

def worker_init_fn(worker_id):
    """Handle random seeding."""
    worker_info = torch.utils.data.get_worker_info()
    seed = worker_info.seed % (2 ** 32 - 1)  # pylint: disable=no-member

    np.random.seed(seed)




class ImageclefDataModule(LightningDataModule):
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
        self.im_size = kwargs["im_size"]
        self.label_list = kwargs["label_list"]
        self.use_two_images = kwargs["use_two_images"]

        self.transforms = Compose([
            transforms.RandomResizedCrop(self.im_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #RandomGaussianBlur(),
            AddGaussianNoise(snr_range=(4, 8)),
            HistogramNormalize(),
            TensorToRGB(),]
        )

        

        self.data_train =  ImageclefDataset(
            directory=self.data_dir,
            split='all',
            transform=self.transforms,
            label_list=self.label_list,
        )

        if self.use_two_images:
            self.data_train = TwoImageDataset(self.data_train)
            """
            We will get a sample of same image with different augmentations.
            sample = {
            "image0": item0["image"],
            "image1": item1["image"],
            "label": item0["labels"],}
            """

        self.data_val = ImageclefDataset(
            directory=self.data_dir,
            split='val',
            transform=self.transforms,
            label_list=self.label_list,
        )

        if self.use_two_images:
            self.data_val = TwoImageDataset(self.data_val)

    def train_dataloader(self):
        dl_train = DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )
        print(len(dl_train))
        return dl_train

    def val_dataloader(self):
        dl_val = DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        )
        return dl_val
