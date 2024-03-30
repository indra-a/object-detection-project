import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Union

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

from src.data.data_augmentation import Resize, RandomHorizontalFlip, RandomTranslation, RandomVerticalFlip, ImageAdjustment, ToTensor, ToPILImage
from src.data.dataset import FoveaDataset
import yaml

class DataPreprocessing(ABC):
    """
    Data Preprocessing that takes pd.DataFrame of labels and image path
    and returns three pytorch dataloaders: train, test, val
    """
    @abstractmethod
    def preprocess_data(self, config: dict):
        pass


class ProjectDataPreprocessing(DataPreprocessing):
    def __init__(self):
        super(ProjectDataPreprocessing, self).__init__()

    def preprocess_data(self, config: dict):
        try:
            train_transformations = Compose([Resize(), RandomHorizontalFlip(), RandomTranslation(), RandomVerticalFlip(), ImageAdjustment(), ToTensor()])
            val_transformation = Compose([Resize(), ToTensor()])

            labels_df = pd.read_excel(config["data"]["label_path"])
            train_df, val_df = train_test_split(labels_df, test_size= 0.05)
            train_df, test_df = train_test_split(train_df, test_size= 0.2)

            trainset = FoveaDataset(config["data"]["images_path"], train_df.reset_index(drop=True), train_transformations)
            testset = FoveaDataset(config["data"]["images_path"], test_df.reset_index(drop=True), val_transformation)
            valset = FoveaDataset(config["data"]["images_path"], val_df.reset_index(drop=True), val_transformation)

            train_loader = DataLoader(trainset, batch_size=config['train']['batch_size'])
            test_loader = DataLoader(testset, batch_size=config['train']['batch_size'])
            val_loader = DataLoader(valset, batch_size=config['train']['batch_size'])

            return train_loader, test_loader, val_loader
        except Exception as e:
            logging.error('Error in data preprocessing: {}'.format(e))
            raise e