
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
        self.annotations = gpd.read_file(annotation_file_path)
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(self.annotations['percentile'].unique())
        self.to_pil = ToPILImage()  # Initialize ToPILImage transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self,index):
        npy_file_path = os.path.join(self.root_dir, self.annotations.iloc[index,4] + '.npy')

        img = np.load(npy_file_path)
        img = self.to_pil(img)

        y_label = torch.tensor(int(self.annotations.iloc[index]['percentile']))

        if self.transform:
            img = self.transform(img)
        return (img, y_label)
    
    def get_class_names(self):
        return self.class_names





# dataset = data_loader_persistence_img(annotation_file_path='/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/data/data/tennessee/2018/SVI2018 TN counties with death rate HepVu/SVI2018_TN_counties_with_death_rate_HepVu.shp',root_dir='/Users/h6x/ORNL/git/persistence-image-classification/scratch model 1/data/data/tennessee/2018/percentiles/H0H1-3 channels',transform=transforms.ToTensor())

# print(len(dataset))

# print(dataset[0][1])
# print(dataset[0][0].shape)

# train_set, test_set = torch.utils.data.random_split(dataset, [70, 25])


# #---
# train_data = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
# test_data = DataLoader(dataset=test_set, batch_size=16, shuffle=False)

# class_names = dataset.get_class_names()
# print(class_names)







