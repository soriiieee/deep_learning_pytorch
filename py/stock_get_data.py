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
# from getErrorValues import me,rmse,mae,r2 #(x,y)
# from convSokuhouData import conv_sfc #(df, ave=minutes,hour)
# #amedas relate 2020.02,04 making...
# from tool_AMeDaS import code2name, name2code
# from tool_110570 import get_110570,open_110570
# from tool_100571 import get_100571,open_100571
#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess
import quandl
api_key = open("/home/ysorimachi/env/api_quandl.env","r").read()
quandl.ApiConfig.api_key = api_key

#torch module import 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
torch.manual_seed(1)

#sci-kit learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


#----------------------------------------------

class GetData:
  """
  2021.02.07 start(class GetData)
  https://qiita.com/creaith/items/2d606a7f0effd5d839d5
  """
  def __init__(self,n_prev, data_code):
    self.n_prev = n_prev
    # self.data = quandl.get(data_code, start_date = "2017/07/01",end_date = "2019/07/25")

  def get_data(self,today):
    tmpX,tmpY=[],[]
    print(today)

    for k in range(self.n_prev):
      tmpX.append(self.data[today - self.n_prev + k][1])
    
    tmpY.append(self.data[today][1])
    return
  
  def get_raw_data(self):
    self.data = quandl.get(data_code)
    return self.data

  def preProcess(self, df):
    df= df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    use_col = ["Date","Close","Volume"]
    df =df[use_col]
    return df

  def splitData(self,df,n_dim=20,sta=datetime(1997,1,1,0,0),end=datetime(2016,12,31,0,0)):

    if df["Date"].dtypes==object:
      df["Date"] = pd.to_datetime(df["Date"])
      df = df[(df["Date"] >= sta)&(df["Date"] <= end)]
    train,test = train_test_split(df, test_size=0.3, shuffle=False)
    t_train,t_test = train["Date"].values,test["Date"].values
    x_min,x_max = np.min(train["Close"]),np.max(train["Close"])

    train = ((train["Close"] -x_min) /(x_max - x_min)).values
    test = ((test["Close"] -x_min) /(x_max - x_min)).values


    x_train,x_test, y_train,y_test= [],[],[],[]
    for i in range(len(train)-n_dim):
      x_train.append(train[i:i+n_dim])
      y_train.append(train[i+n_dim])

    for i in range(len(test)-n_dim):
      x_test.append(test[i:i+n_dim])
      y_test.append(test[i+n_dim])
    return x_train,x_test, y_train,y_test,t_train,t_test

class MyDataSet(Dataset):
  """
  2021.02.08 add
  https://dajiro.com/entry/2020/05/06/183255
  """
  def __init__(self,x,y):
    self.data = x
    self.teacher = y
    # self.time = t
  
  def __len__(self):
    return len(self.teacher)
  
  def __getitem__(self,idx):
    out_data = self.data[idx]
    out_label = self.teacher[idx]
    # out_time = self.time[idx]
    
    #to tensor
    out_data = torch.tensor(out_data,dtype=torch.float32)
    out_label = torch.tensor(out_label,dtype=torch.float32)
    return out_data, out_label

class LSTM(nn.Module):
  def __init__(self,input_size=1, hidden_layer_size=100,output_size=1, batch_size=32):
    super().__init__()
    #setting
    self.hidden_layer_size = hidden_layer_size
    self.batch_size = batch_size
    #model
    self.lstm = nn.LSTM(input_size, hidden_layer_size)
    self.hidden_cell = (
      torch.zeros(1,self.batch_size,self.hidden_layer_size),
      torch.zeros(1,self.batch_size,self.hidden_layer_size)
    )
    self.linear = nn.Linear(hidden_layer_size, output_size)
  
  def forward(self,input_seq):
    batch_size, seq_len = input_seq.shape
    #lstm
    lstm_out, self.hidden_cell = self.lstm(
      input_seq.view(seq_len,batch_size, 1),self.hidden_cell)
    #linear
    prediction = self.linear(self.hidden_cell[0].view(batch_size,-1))
    return prediction[:,0]


#main ffunction
def get_stock(n_prev, data_code = "WIKI/AAPL"):
  gd = GetData(n_prev, "WIKI/AAPL")
  if not os.path.exists("../dat/stock2.csv"):
    if not os.path.exists("../dat/stock.csv"):
      print("not found getting..")
      data = gd.get_raw_data()
      # data.to_csv("../dat/stock.csv")
    else:
      print("already getting..")
      data = pd.read_csv("../dat/stock.csv")
    data = gd.preProcess(data)
    data.to_csv("../dat/stock2.csv", index=False)
  else:
    data = pd.read_csv("../dat/stock2.csv")
  return gd, data

if __name__ == "__main__":

  n_prev = 30
  hidden_size = 300
  cell_size=100
  epochs=10
  batch_size=32

  if 1:
    gd, data = get_stock(n_prev, "WIKI/AAPL")
    x_train,x_test, y_train,y_test,t_train,t_test = gd.splitData(data,n_dim=30)

    # instance
    dataset = MyDataSet(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # model
    model = LSTM(input_size=1, hidden_layer_size=100,output_size=1, batch_size=32)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # main fitting
    sample_len = len(x_train)
    _loss =[]

    print("sTart traing....")
    for epoch in range(1,epochs+1):
      for b,tup in enumerate(dataloader):
        x,y,t = tup
        

        #zerograd
        optimizer.zero_grad()
        #get hidden& cell
        model.hidden_cell = (
          torch.zeros(1,len(x), model.hidden_layer_size),
          torch.zeros(1,len(x), model.hidden_layer_size),
        )

        #forward
        y_pred = model(x)
        #loss
        loss_tmp = loss(y_pred,y)
        #backward calclation
        loss_tmp.backward()
        #update
        optimizer.step()
      
      #print
      _loss.append(loss_tmp.item())
      # df_pred[f"pred_{epoch+1}"] =  model(torch.tensor(x_train)).numpy()
      print(f"{datetime.now()}[info] epoch:{epoch+1} loss:{loss_tmp.item()}")

      # if epoch ==epochs:

    

    #output(csv/png)
    df_pred.to_csv("../out/prediction.csv", index=False)
    df=pd.DataFrame()
    df["epoch"] = list(range(1,epochs+1))
    df["loss"] = _loss
    df.to_csv("../out/loss_curve.csv", index=False)
    plt.figure(figsize=(15,8))
    plt.plot(_loss, label="loss")
    plt.savefig("../out/loss_curve.png",bbox_inches="tight")
  # if 1:

  sys.exit()



