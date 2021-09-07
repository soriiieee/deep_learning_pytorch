# 操作方法
# 参考：　https://kuune.org/text/2017/05/14/use-jupyter-notebook-on-server-through-ssh-port-forwarding/

## 一連の流れ
* cd /home/ysorimachi/work/sori_py2/deepl/jupyter
* 環境変更 -> `conda activate sori_torch` 等にしておかないと、jupyterの起動でミスる
* `jupyter notebook`
* ローカルのターミナルから、ssh接続&ポートフォワーディングする
  ssh -L 8888:localhost:8888 ysorimachi@133.105.83.41
  ssh -L 6006:localhost:6006 ysorimachi@133.105.83.41 #tensorboard


## jupyter notebook に最初に記述tenplate
### ------------------------------------------------
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
# sori -module
sys.path.append('/home/ysorimachi/tool')
from getErrorValues import me,rmse,mae,r2 #(x,y)
from convSokuhouData import conv_sfc #(df, ave=minutes,hour)

#---------------------------------------------------
import subprocess
import requests
#--------------------------
import pickle
import gzip
from pathlib import Path

# deep learning modules 
# imports

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
### ------------------------------------------------

# jupyter のショートカットキーたち
参考URL： https://qiita.com/zawawahoge/items/baa2a5318df079c5f7e5


