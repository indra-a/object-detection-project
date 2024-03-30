import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
from pathlib import Path

class FoveaDataset(Dataset):
    def __init__(self, data_path, labels_df, transforms):
        self.data_path = Path(data_path)
        self.labels_df = labels_df
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path, label_1, label_2 = self.labels_df.loc[index,'imgName'], self.labels_df.loc[index,'Fovea_X'], self.labels_df.loc[index,'Fovea_Y']
        if img_path.startswith('A'):
            sub_dir = Path('AMD/')
        else:
            sub_dir = Path('Non-AMD/')
        image = Image.open(self.data_path / sub_dir / Path(img_path))
        label = (label_1, label_2)
        return self.transforms((image, label))

    def __len__(self):
        return(len(self.labels_df))