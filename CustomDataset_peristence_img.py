
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import pandas as pd
import os
from skimage import io
import geopandas as gpd
import matplotlib.pyplot as plt

class data_loader_persistence_img(Dataset):

    def __init__(self,geo_file_path,root_dir,transform=None):
        self.annotations = gpd.read_file(geo_file_path)
        self.root_dir = root_dir
        self.transform = transform


    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self,index):

        img_path = os.path.join(self.root_dir, self.annotations.iloc[index,4] + '.png')

        img = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index]['percentile']))

        if self.transform:
            img = self.transform(img)

        return (img, y_label)
        