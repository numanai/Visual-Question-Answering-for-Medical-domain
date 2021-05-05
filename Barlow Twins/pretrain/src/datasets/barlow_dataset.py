import os

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CheXpertDataset(Dataset):
    """"""

    def __init__(self, directory, split = 'train', transform = None, label_list = 'all'):

        super().__init__()

        self.directory = Path(directory)
        self.csv = None
        self.split = split
        self.label_list = label_list
        self.transform = transform
        self.metadata_keys: List[str] = []


        if label_list == "all":
            self.label_list = self.default_labels()
        else:
            self.label_list = label_list

        self.metadata_keys = [
            "Patient ID",
            "Path",
            "Sex",
            "Age",
            "Frontal/Lateral",
            "AP/PA",
        ]

        if self.split == "train":
            self.csv_path = self.directory / "CheXpert-v1.0-small" / "train.csv"
            self.csv = pd.read_csv(self.directory / self.csv_path)
        elif self.split == "val":
            self.csv_path = self.directory / "CheXpert-v1.0-small" / "valid.csv"
            self.csv = pd.read_csv(self.directory / self.csv_path)
        elif self.split == "all":
            self.csv_path = self.directory / "train.csv"
            self.csv = pd.concat(
                [
                    pd.read_csv(self.directory / "CheXpert-v1.0" / "train.csv"),
                    pd.read_csv(self.directory / "CheXpert-v1.0" / "valid.csv"),
                ]
            )
        else:
            logging.warning(
                "split {} not recognized for dataset {}, "
                "not returning samples".format(split, self.__class__.__name__)
            )

        self.csv = self.preproc_csv(self.csv)

    @staticmethod
    def default_labels():
        return [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]

    def preproc_csv(self, csv: pd.DataFrame) -> pd.DataFrame:
        if csv is not None:
            csv["Patient ID"] = csv["Path"].str.extract(pat="(patient\\d+)")
            csv["view"] = csv["Frontal/Lateral"].str.lower()

        return csv


    def open_image(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("L")
    


    def __len__(self):
        length = 0

        if self.csv is not None:
            length = len(self.csv)

        return length


    def retrieve_metadata(
        self, idx: int, filename, exam):
        metadata = {}
        metadata["dataloader class"] = self.__class__.__name__
        metadata["idx"] = idx  # type: ignore
        for key in self.metadata_keys:
            # cast to string due to typing issues with dataloader
            metadata[key] = str(exam[key])
        metadata["filename"] = str(filename)

        metadata["label_list"] = self.label_list  # type: ignore

        return metadata



    def __getitem__(self, idx):
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.directory / exam["Path"]
        image = self.open_image(filename)

        metadata = self.retrieve_metadata(idx, filename, exam)

        # retrieve labels while handling missing ones for combined data loader
        labels = np.array(exam.reindex(self.label_list)[self.label_list]).astype(
            np.float
        )

        

        if self.transform is not None:
            img0, img1 = self.transform(image)


        sample = {"image": [img0, img1], "labels": labels, "metadata": metadata}


        return sample

    


class ImageClefDataset(Dataset):
    """"""

    def __init__(self, directory, transform = None):

        super().__init__()

        self.directory = Path(directory)
        self.csv = None
        self.transform = transform





        self.csv_path = self.directory / "ImageClef" / "filenames.csv"
        self.csv = pd.read_csv(self.csv_path)
    



    def open_image(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("L")
    


    def __len__(self):

        if self.csv is not None:
            length = len(self.csv)
        return length


    def __getitem__(self, idx):
        assert self.csv is not None
        exam = self.csv.iloc[idx]

        filename = self.directory /"ImageClef"/ exam["filename"]
        image = self.open_image(filename)

        

        if self.transform is not None:
            img0, img1 = self.transform(image)


        sample = {"image": [img0, img1]}


        return sample