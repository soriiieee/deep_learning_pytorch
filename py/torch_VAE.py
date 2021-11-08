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
from data import load_dataset


def main():
  # ------
  train,test = load_dataset(cate="MNIST")
  train_loader = DataLoader(train, batch_size=40)
  test_loader = DataLoader(test, batch_size=40)
  
  # ----
  chack(train_loader, n=1)
  
  
  return

def chack(data_loader, n):
  for img in data_loader:
    # print(img,lbl)
    sys.exit()


if __name__ == "__main__":
  main()

