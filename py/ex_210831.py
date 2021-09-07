# -*- coding: utf-8 -*-
# when   : 2021.08.31
# who : [sori-machi]
# what : [ deep learning の学習用]
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
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
import requests

# tutorials ---------------
# https://pytorch.org/tutorials/beginner/nn_tutorial.html
#--------------------------
import pickle
import gzip
from pathlib import Path

#pytorch package import 
import torch


DATA_PATH = "/home/ysorimachi/work/sori_py2/deepl/dat/data"
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

def png_mnist(_img,png_out,num=25):
  """
  2021.08.31
  サンプルを可視化するようなツール
  """
  f,ax= plt.subplots(5,5,figsize=(25,25))
  ax = ax.flatten()
  for (i,img) in tqdm(list(enumerate(_img[:num]))):
    ax[i].imshow(img,cmap="gray")
    ax[i].set_title(i)
  f.savefig(png_out, bbox_inches="tight")
  return 
  
OUTD="/home/ysorimachi/work/sori_py2/deepl/out/torch_tutorial_210831"
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
  ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
  
  # _img = [ x_train[i,:].reshape((28,28)) for i in range(50)]
  # png_out = f"{OUTD}/sample.png"
  # png_mnist(_img,png_out,num=25) #sample 表示]
  x_train,y_train, x_valid,y_valid = map(torch.tensor, (x_train, y_train,x_valid, y_valid))
  
  
  