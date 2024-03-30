import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Union
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split

from data_augmentation import Resize, RandomHorizontalFlip, RandomTranslation, RandomVerticalFlip, ImageAdjustment, ToTensor, ToPILImage
from dataset import FoveaDataset
from configs import data_config
import yaml

class DataPreprocessing(ABC):
    """
    Data Preprocessing that takes pd.DataFrame of labels and image path
    and returns three pytorch dataloaders: train, test, val
    """
    @abstractmethod
    def preprocess_data(self, config: dict) -> Union[torch.DataLoader, torch.DataLoader, torch.DataLoader]:
        pass


def ProjectDataPreprocessing(DataPreprocessing):

    def preprocess_data(self, config: dict) -> Union[torch.DataLoader, torch.DataLoader, torch.DataLoader]:
        try:
            train_transformations = Compose([Resize(), RandomHorizontalFlip(), RandomTranslation(), RandomVerticalFlip(), ImageAdjustment(), ToTensor()])
            val_transformation = Compose([Resize(), ToTensor()])

            labels_df = pd.read_excel(config["data"]["label_path"])
            train_df, val_df = train_test_split(labels_df, test_size= 0.05)
            train_df, test_df = train_test_split(train_df, test_size= 0.2)

            trainset = MyDataset(config["data"]["image_path"], train_df.reset_index(drop=True), train_transformations)
            testset = MyDataset(config["data"]["image_path"], test_df.reset_index(drop=True), val_transformation)
            valset = MyDataset(config["data"]["image_path"], val_df.reset_index(drop=True), val_transformation)

            train_loader = DataLoader(trainset, batch_size=config['model']['batch_size'])
            test_loader = DataLoader(testset, batch_size=config['model']['batch_size'])
            val_loader = DataLoader(valset, batch_size=config['model']['batch_size'])

            return train_loader, test_loader, val_loader
        except Exception as e:
            logging.error('Error in data preprocessing: {}'.format(e))
            raise e