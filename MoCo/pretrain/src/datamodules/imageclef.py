import os
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageclefDataset(Dataset):
    """
    Data loader for CheXpert data set.
    Args:
        directory: Base directory for data set with subdirectory
            'CheXpert-v1.0-small'.
        split: String specifying split.
            options include:
                'all': Include all splits.
                'train': Include training split.
                'val': Include validation split.
        label_list: String specifying labels to include. Default is 'all',
            which loads all labels.
        transform: A composible transform list to be applied to the data.
    Irvin, Jeremy, et al. "Chexpert: A large chest radiograph dataset with
    uncertainty labels and expert comparison." Proceedings of the AAAI
    Conference on Artificial Intelligence. Vol. 33. 2019.
    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/
    Code is inpired from:
    https://github.com/facebookresearch/CovidPrognosis/
    """

    def __init__(
        self,
        directory: Union[str, os.PathLike],
        split: str = "all",
        transform: Optional[Callable] = None,
        label_list: Union[str, List[str]] = "all",
    ):
        super().__init__()


        self.directory = Path(directory)
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.transform = transform
        self.metadata_keys: List[str] = []
        
        if self.split == "train":
            self.csv_path = self.directory / "train" / "train_ImageIds_all.txt"
            self.csv = pd.read_csv(self.directory / self.csv_path,header=None,names=['image_name'])
            self.csv.image_name = self.csv.image_name.apply(lambda x: str(x)+'.jpg')
        elif self.split == "val":
            self.csv_path = self.directory / "validation" / "val_ImageIds_all.txt"
            self.csv = pd.read_csv(self.directory / self.csv_path,header=None,names=['image_name'])
            self.csv.image_name = self.csv.image_name.apply(lambda x: str(x)+'.jpg')
        elif self.split == "test":
            self.csv_path = self.directory / "test" / "test_ImageIds_all.txt"
            self.csv = pd.read_csv(self.directory / self.csv_path,header=None,names=['image_name'])
            self.csv.image_name = self.csv.image_name.apply(lambda x: str(x)+'.jpg')
        elif self.split == "all":
            #self.csv_path = self.directory / "train.csv"
            self.csv = pd.concat(
                [
                    pd.read_csv(self.directory / "train" / "train_ImageIds_all.txt",header=None,names=['image_name']),
                    pd.read_csv(self.directory / "validation" / "val_ImageIds_all.txt",header=None,names=['image_name']),
                    pd.read_csv(self.directory / "test" / "test_ImageIds_all.txt",header=None,names=['image_name']),

                ]
            )
            self.csv.image_name = self.csv.image_name.apply(lambda x: str(x)+'.jpg')
        else:
            logging.warning(
                "split {} not recognized for dataset {}, "
                "not returning samples".format(split, self.__class__.__name__)
            )



    def open_image(self, path: Union[str, os.PathLike]) -> Image:
        with Image.open(path) as img:
            return img.convert("F")

   
    def __len__(self) -> int:
        length = 0
        if self.csv is not None:
            length = len(self.csv)

        return length

    def __getitem__(self, idx: int) -> Dict:
        assert self.csv is not None
        exam = self.csv.iloc[idx]
        
        filename = self.directory / 'images_all'/exam['image_name']
        image = self.open_image(filename)


        sample = {"image": image}
        #self.tranform is not None, it's Compose, so transformation is happening here
        if self.transform is not None:
            sample = self.transform(sample)


        return sample




class TwoImageDataset(Dataset):
    """
    Wrapper for returning two augmentations of the same image.
    Args:
        dataset: Pre-initialized data set to return multiple samples from.
    """

    def __init__(self, dataset):
        
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # randomness handled via the transform objects
        # this requires the transforms to sample randomness from the process
        # generator
        item0 = self.dataset[idx] #these two are the same images
        item1 = self.dataset[idx]

        sample = {
            "image0": item0["image"],
            "image1": item1["image"],
            #"label": item0["labels"],
        }

        return sample