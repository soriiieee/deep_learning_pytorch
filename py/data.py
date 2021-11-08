# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
#amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
from tool_110570 import get_110570,open_110570
from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
# import torchvision.datasets 
import torchvision.transforms as transforms
transform = transforms.Compose(
  [ transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

from PIL import Image
ROOT="/home/ysorimachi/work/sori_py2/deepl/dat/data"

def load_dataset(cate="MNIST"):
  # print(dir(torchvision.datasets))
  use_torch_list = ['CIFAR10', 'CIFAR100', 'Caltech101', 'Caltech256', 'CelebA', 'Cityscapes', 'CocoCaptions', 'CocoDetection', 'DatasetFolder', 'EMNIST', 'FakeData', 'FashionMNIST', 'Flickr30k', 'Flickr8k', 'HMDB51', 'ImageFolder', 'ImageNet', 'KMNIST', 'Kinetics400', 'LSUN', 'LSUNClass', 'MNIST', 'Omniglot', 'PhotoTour', 'Places365', 'QMNIST', 'SBDataset', 'SBU', 'SEMEION', 'STL10', 'SVHN', 'UCF101', 'USPS']
  
  if cate not in use_torch_list:
    sys.exit("Not DATA category !")
  
  if cate == "MNIST":
    train_set = torchvision.datasets.MNIST(root=ROOT, train=True, download=True,transform = transform)
    test_set = torchvision.datasets.MNIST(root=ROOT, train=False, download=True,transform = transform)
  if cate == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True,transform = transform)
    test_set = torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True,transform = transform)
  
  if cate == "FashionMNIST":
    train_set = torchvision.datasets.FashionMNIST(root=ROOT, train=True, download=True,transform = transform)
    test_set = torchvision.datasets.FashionMNIST(root=ROOT, train=False, download=True,transform = transform)
  return train_set,test_set


if __name__ == "__main__":
  load_dataset()

