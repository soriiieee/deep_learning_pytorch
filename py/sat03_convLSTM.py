# -*- coding: utf-8 -*-
# when   : 2021.10.15
# who : [sori-machi]
# what : [ sat03画像から画像生成を行うprogram]
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
from PIL import Image
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
from tool_time import dtinc
# outd ='/home/griduser/work/sori-py2/deep/out/0924_01'
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)
OUT="/home/ysorimachi/work/sori_py2/deepl/out"
DIR03="/work2/ysorimachi/sat/convLSTM/store/03"

def get_file(dd="20190110"):
  _tt = [ t.strftime("%Y%m%d%H%M") for t in pd.date_range(start=f"{dd}0900", freq="5T",periods=12*6)]
  _tt = [ dtinc(t,4,-9) for t in _tt]
  _ff = [ f"{DIR03}/122841-{t[10:12]}0003-R050-{t}00.png" for t in _tt]
  return _ff

def check_file(_path):
  num=0
  _path2 = []
  for p in _path:
    if os.path.exists(p):
      num +=1
      _path2.append(p)
  percent = np.round(num*100/len(_path),1)
  print(f"データ存在{num}個 | {percent}%")
  return _path2

def read_time_from_prd(path):
  prd_u = os.path.basename(path).split("-")[3][:12]
  ini_j = dtinc(prd_u,4,9)
  return ini_j

def triming(img,center_position,delta):
  if type(img) != np.ndarray:
    # print("Image")
    img = np.array(img)
  else:
    pass

  center_lon = center_position[0]
  center_lat = center_position[1]
  lon0,lon1 = center_lon-delta,center_lon+delta
  lat0,lat1 = center_lat-delta,center_lat+delta
  _lat = np.linspace(20,50,img.shape[0])
  _lon = np.linspace(120,150,img.shape[1])
  iy0,iy1 = np.argmin(np.abs(_lat - lat0)),np.argmin(np.abs(_lat - lat1))
  ix0,ix1 = np.argmin(np.abs(_lon - lon0)),np.argmin(np.abs(_lon - lon1))
  img = img[iy0:iy1, ix0:ix1]
  img = Image.fromarray(img)
  return img

def read_png(path):
  H,W = 360,360
  img = Image.open(path)
  # 東京　140/36
  img = triming(img,center_position=[140,36],delta=5)
  img = img = img.resize((W, H))
  return img



def main():
  dd = "20190110"
  _ff = get_file(dd)
  _ff = check_file(_ff) #データ存在しているファイルのみreturnしてくるようなmodule
  # print(_ff)
  
  n=6
  f,ax = plt.subplots(1,6,figsize=(24,4))
  # ax = ax.flatten()
  for i,path in enumerate(_ff[:n]):
    
    ini_j = read_time_from_prd(path)
    img = read_png(path)
    ax[i].imshow(np.array(img))
    ax[i].set_title(ini_j)
    ax[i].tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
  
  plt.subplots_adjust(wspace=0.0, hspace=0)
  f.savefig(f"{OUT}/check_211015/img_{dd}.png", bbox_inches="tight")
  plt.close()
  return 


if __name__ =="__main__":
  main()