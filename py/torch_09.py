# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import sys,os,re,glob
import pandas as pd
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

import matplotlib
import matplotlib.pyplot as plt


# wine = datasets.load_wine()
# x,y = wine.data, wine.target
# print(x.shape, y.shape)
# sys.exit()
"""
2021.02.17
https://www.youtube.com/watch?v=X_QOZEko5uE&list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4&index=10
"""

class WineDataset(Dataset):
  def __init__(self,transform=None):
    
    xy = datasets.load_wine()
    self.x = torch.from_numpy(xy.data)
    self.y = torch.from_numpy(xy.target.reshape(xy.data.shape[0], 1))
    self.n_samples = xy.data.shape[0]
    self.transform = transform
    
    
    
  def __getitem__(self,index):
    sample = self.x[index], self.y[index]
    if self.transform:
      
    
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples
  

dataset = WineDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

num_epoch = 2
toral_samples = len(dataset)

for epoch in range(num_epoch):
  for i (X,y) in enumerate(dataloader):
    