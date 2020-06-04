# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 18:47:50 2020

@author: josea
"""
import torch
from torch.utils.data import Dataset
from utils import load_SVHN
from PIL import Image

class svhn_dataset(Dataset):
    def __init__(self, folder_path, transform = None):
        self.folder_path = folder_path
        self.data = load_SVHN(folder_path=folder_path)
        self.transform = transform
    def __len__(self):
        return len(self.data[1])
    def __getitem__(self, idx):
        #image = np.moveaxis(self.data[0][idx], -1, 0)
        image = self.data[0][idx]
        labels = self.data[1][idx]
        if(self.transform):
            image = self.transform(image)
        sample = {"image": image, "labels": torch.tensor(labels)}
        return(sample)

class toPIL(object):
    def __call__(self, image):
        image = Image.fromarray(image, mode="RGB")
        return image
    