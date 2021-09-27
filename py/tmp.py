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
# os.makedirs(outd, exist_ok=True)
# subprocess.run('rm -f *.png', cwd=outd, shell=True)

def calc_time(x):
  yy,mm,dd = map(int, x[0].split("/"))
  hh,mi = map(int, x[1].split(":"))
  return datetime(yy,mm,dd,hh,mi)
  
def main():
  DIR=f"/home/ysorimachi/work/sori_py2/deepl/dat/data/power"
  if 0: #5 years data concat
    _p = sorted(glob.glob(f"{DIR}/juyo_*.csv"))
    _df = [ pd.read_csv(p,skiprows=1) for p in _p]
    df = pd.concat(_df,axis=0)
    df.to_csv("./all_demand.csv", index=False)
  
  df = pd.read_csv("./all_demand.csv")
  df["time"] = df[["DATE","TIME"]].apply(lambda x: calc_time(x),axis=1)
  df = df.drop(["DATE","TIME"],axis=1).set_index("time")
  df.columns = ["DEMAND(10^4kW)","ESTIMATE(10^4kW)"]
  df.to_csv("./all_demand2.csv")
  return


if __name__ =="__main__":
  main()