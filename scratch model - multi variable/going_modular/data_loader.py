
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

import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from torchvision.transforms import ToPILImage


class data_loader_persistence_img(Dataset):

    def __init__(self,annotation_file_path,root_dir,transform=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self,index):
        pass

    def get_class_names(self):
        pass




root_dir = "/Users/h6x/ORNL/git/persistence-image-classification/data/tennessee/2018/percentiles/below 90/h1/npy 3 channels"
annotation_file_path = "/Users/h6x/ORNL/git/persistence-image-classification/data/SVI - census tract level/SVI 2018 with HepVu census tracts/SVI2018_US_census_with_opioid_indicators.shp"

# dataset = data_loader_persistence_img(annotation_file_path=,root_dir=,transform=transforms.ToTensor())

# print(len(dataset))

# print(dataset[0][1])
# print(dataset[0][0].shape)

# train_set, test_set = torch.utils.data.random_split(dataset, [70, 25])


# #---
# train_data = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
# test_data = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# class_names = dataset.get_class_names()
# print(class_names)







