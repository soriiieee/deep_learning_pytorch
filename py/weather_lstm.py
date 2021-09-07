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
from dateutil.relativedelta import relativedelta
import warnings
warnings.simplefilter('ignore')
from tqdm import tqdm
import seaborn as sns
#---------------------------------------------------
# sori -module
sys.path.append('/home/ysorimachi/tool')
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
from convAmedasData import conv_amd #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
# sklearn module
from sklearn.model_selection import train_test_split

# torch module
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
torch.manual_seed(1)
"""
参考にしたサイト：
kerasで実装しているので、これをpytorch用に変更して用いる
参考1:　"https://qiita.com/nvtomo1029/items/689c0a19880d1dc41d43"
参考2： "https://hilinker.hatenablog.com/entry/2018/06/23/204910"

"""


def get_data(out_d,name="東京",ini_month="201201",n_month=12*5, isDownload=False):
  """
  main: 
  """
  code = name2code(name)
  os.makedirs(out_d, exist_ok=True)
  ini_j = pd.to_datetime(f"{ini_month}010000")

  _ini_j=[ini_j]
  for n in range(n_month):
    _ini_j.append(ini_j+relativedelta(months=n))
  
  _month = [ t.strftime("%Y%m") for t in _ini_j]

  if isDownload:
    subprocess.run("rm -f *.csv",cwd =out_d, shell=True)
    for month in tqdm(_month):
      com ="sh /home/ysorimachi/work/hokuriku/bin/get_AMeDas.sh {} {} {}".format(month,code,out_d)
      subprocess.run(com,cwd =out_d, shell=True)
      
  
  _f_path = glob.glob(f"{out_d}/*.csv")

  _df=[]
  for f_path in _f_path:
    df = pd.read_csv(f_path)
    df  = conv_amd(df,ave=30)
    _df.append(df)
  
  df = pd.concat(_df,axis=0)
  df = df.sort_values("time")
  df.to_csv("../dat/weather.csv", index=False)
  return


def pre_process(df,st,ed):
  """
  st: datetime()
  ed: datetime()
  """
  df["temp"] = df["temp"].apply(lambda x: np.nan if x>50 else x)
  df = df.dropna()
  #rule3 average temprature
  df["temp"] = df["temp"].rolling(48).mean()
  df= df.dropna()
  df = df.iloc[::48,:]
  #normalization
  # x_min,x_max = np.min(df["temp"]),np.max(df["temp"])
  # df["temp"] = (df["temp"] -x_min) /(x_max - x_min)
  return df

def generate_data(n_length, n_dimension):
  df = pd.read_csv("../dat/weather.csv")
  df["time"] = pd.to_datetime(df["time"])
  
  #cuttign
  print("before", df.shape)
  df = pre_process(df,datetime(2015,1,1,0,0),datetime(2019,1,1,0,0))
  print("after", df.shape)

  #init
  data = df["temp"].values
  time = df["time"].values

  x,y,t=[],[],[]
  n_data = len(df)
  for i in range(0,n_data- n_length):
    x.append(data[i:i+n_length])
    y.append(data[i+n_length])
    t.append(time[i+n_length])
  
  x = np.array(x,dtype=np.float32).reshape(-1,n_length,n_dimension)
  y = np.array(y, dtype = np.float32).reshape(-1, 1)

  return x,y,t
  
class LSTM(nn.Module):
  def __init__(self,input_size=1, hidden_size=100, output_size=1, batch_size=100):
    super().__init__()
    self.hidden_size = hidden_size
    self.batch_size = batch_size

    self.lstm = nn.LSTM(1,self.hidden_size,batch_first=True)
    # self.hidden_cell = {
    #   torch.zeros(1,self.batch_size,self.hidden_size),
    #   torch.zeros(1,self.batch_size,self.hidden_size),
    # }
    self.linear = nn.Linear(self.hidden_size, output_size)
  
  def forward(self,x,hidden0=None):
    batch_size,input_size = x.shape[0],x.shape[1]
    
    #lstm
    output,(hidden,cell) = self.lstm(x,hidden0)
    #linear
    output = self.linear(output[:,-1,:])
    return output

class MyDataSet(Dataset):
  def __init__(self,x,y):
    self.data = x
    self.target = y

  def __len__(self):
    return len(self.target)

  def __getitem__(self, idx):
    x_tmp = self.data[idx]
    y_tmp= self.target[idx]

    x_get = torch.tensor(x_tmp, dtype=torch.float32) 
    y_get = torch.tensor(y_tmp, dtype=torch.float32)
    use_idx = torch.tensor(idx, dtype=torch.int)

    return x_get, y_get,use_idx


if __name__ =="__main__":
  out_d="../dat/ame"
  name = "東京"
  if 0:
    get_data(out_d,name,ini_month="201101",n_month=12*10,isDownload=False)

  if 1:
    #initilize----------------
    input_size = 90
    batch_size = 128
    n_dim = 1 #only temperatur
    #initilize----------------
    x,y,t = generate_data(input_size, n_dim)

    x_train,x_test, y_train,y_test, t_train,t_test = train_test_split(x,y,t, test_size=0.2, shuffle=False)

    res_train,res_test = pd.DataFrame(),pd.DataFrame()
    res_train["time"],res_test["time"] = t_train,t_test
    res_train["target"],res_test["target"] = y_train,y_test

    train_data_set = MyDataSet(x_train,y_train)
    test_data_set = MyDataSet(x_test,y_test)

    train_loader = DataLoader(train_data_set,batch_size=batch_size)
    test_loader = DataLoader(test_data_set,batch_size=batch_size)

    #setting--------------
    hidden_size = 100
    n_epoch=50
    input_size=1
    #setting--------------
    #model
    # (self,input_size=1, hidden_size=100, output_size=1, batch_size=100)
    # model = LSTM(n_length, hidden_size, output_size=1, batch_size=batch_size)
    model = LSTM(input_size=1,hidden_size = hidden_size,batch_size=batch_size)
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # main fitting
    sample_len = len(x_train)

    print("train start ...")
    subprocess.run("rm -f {}".format("../log/log_weather.log"),shell=True)
    with open("../log/log_weather.log" , "+a") as f:
      for epoch in range(1,n_epoch+1):
        for b,tup in enumerate(train_loader):
          x,y,idx = tup
          optimizer.zero_grad()
          y_pred = model(x)
          # print(y_pred.shape)
          loss_val = loss(y,y_pred)
          loss_val.backward()
          optimizer.step()
          # print(epoch,b)

        # _loss.append(loss_val.item())
        text = f"{datetime.now()},[info],{epoch},loss,{loss_val.item()}\n"

        y_pred_train = model(torch.tensor(x_train).view(-1,input_size,1)).detach().numpy()
        y_pred_test = model(torch.tensor(x_test).view(-1,input_size,1)).detach().detach().numpy()

        print(y_pred_train.shape)
        sys.exit()
        print(y_pred_test)
        print(x_train.shape)
        sys.exit()
        res_train[f"pred_{epoch}"]= y_pred_train
        res_test[f"pred_{epoch}"]= y_pred_test
        f.write(text)
        print(f"epoch {epoch} end...")
      
      res_train.to_csv("../out/weather/res_train.csv", index=False)
      res_test.to_csv("../out/weather/res_test.csv", index=False)
        # print(f"{epoch}({b}): loss={loss_val}")








