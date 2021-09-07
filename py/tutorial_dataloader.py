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
# sys.path.append('/home/ysorimachi/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
from PIL import Image
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

DATADIR="/home/ysorimachi/work/sori_py2/deepl/dat/mnist"


"""
dataloader を理解する
https://qiita.com/typecprint/items/2d616b502dd063ba4e27
"""


class PairMnistDataset(Dataset):
  """
  https://qiita.com/typecprint/items/233eec0c949cc5df29e3
  """
  def __init__(self,mnist_dataset, train=True):
    self.train = train #bool]
    self.dataset = mnist_dataset
    self.transform = mnist_dataset.transform
  
    if self.train:
      self.train_data   = self.dataset.train_data
      self.train_labels = self.dataset.train_labels
      self.train_label_set = set(self.train_labels.numpy()) #labelの種類の合計
      self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                for label in self.train_label_set}
    else:
      self.test_data = self.dataset.test_data
      self.test_labels = self.dataset.test_labels
      self.test_label_set = set(self.test_labels.numpy())
      self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                for label in self.test_label_set}
      
      positive_pair = [
        [i, np.random.choice(self.label_to_indices[self.test_labels[i].item()]),
        1] for i in range(0,len(self.test_data),2)
      ]
      
      negative_pair = [
        [i, np.random.choice(self.label_to_indices[self.test_labels[i].item()]),
        0] for i in range(0,len(self.test_data),2)
      ]

      self.test_pairs = positive_pair + negative_pair
  
  def __getitem(self,idx):
    if self.train:
      target = np.random.randint(0,2)
      #img1,label1 は先に決めてしまう
      img1,label1 = self.train_data[idx], self.train_labels[idx].items()
      if target==1:
        """
        positive pair
        """
        siamese_index = idx #labelが同じになるindexを選んでくる処理
        while siamese_index == idx:
          siamese_index = np.random.choice(self.label_to_indices[label1])
      else:
        """
        negative pair
        """
        siamese_label = np.random.choice(list(self.train_label_set - set([label1])))
        siamese_index = np.random.choice(self.label_to_indices[siamese_label])
      
      img2 = self.train_data[siamese_index]
    
    else:
      img1 = self.test_data[self.test_pairs[index][0]]
      img2 = self.test_data[self.test_pairs[index][1]]
      target = self.test_pairs[idx][2]

    img1 = Image.fromarray(img1.numpy(), mode='L')
    img2 = Image.fromarray(img2.numpy(), mode='L')
    
    if self.transform:
      img1 = self.transform(img1)
      img2 = self.transform(img2)
    
    return (img1,img2), target

  def __len__(self):
    return len(self.dataset) 


def main():
  """
  get data
  """
  train_dataset = torchvision.datasets.MNIST(
    root=DATADIR,
    train=True,
    download=True,
    # transform=transform
    transform =transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
    ])
  )
  test_dataset = torchvision.datasets.MNIST(
    root=DATADIR,
    train=False,
    download=True,
    # transform=transform
    transform =transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
    ])
  )

  pair_train_dataset = PairMnistDataset(train_dataset, train=True)

  pair_train_loader = torch.utils.data.DataLoader(
    pair_train_dataset, batch_size=16)


  pair_test_dataset = PairMnistDataset(test_dataset, train=True)
  pair_test_loader = torch.utils.data.DataLoader(
    pair_test_dataset, batch_size=16)
  
  # 

  return


if __name__=="__main__":
  main()