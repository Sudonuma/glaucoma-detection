import numpy as np
import random
from PIL import Image

from torch.utils.data import Dataset
import torch
import os
import glob
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class ResnetDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        file_path = glob.glob(f'{self.root_dir}/**/{img_name}.jpg', recursive=True)
        
        image = pil_loader(file_path[0])
        label = self.df.iloc[idx, 1]  # The label encoded as an integer (remove the class from the trainer)

        if self.transform:
            image = self.transform(image)

        return image, label
    
#  for testing (primarly)
if __name__ == "__main__":
    # Define your data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])  
    resnet_dataset = ResnetDataset(root_dir='/home/sudonuma/Documents/glaucoma/data/dataset', csv_file='dataset/train_labels.csv', transform=transform)

    
    data_loader = DataLoader(resnet_dataset, batch_size=1, shuffle=True)

    for batch in data_loader:
        image, label = batch
