# -*- coding: utf-8 -*-
"""
torch のseq2seqモデルの実装を行うようにするprogram
date: 2021.10.29
date: 2021.11.02　全然成果が得られなかった-->50000回実施して精度acc＝1.3%前後、やはり、正解値を出力しながらの予測の精度は著しく悪化する
"""

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
import requests
try:
  print("[OK] import torch modules ...")
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import DataLoader, TensorDataset
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except:
  print("[NG] Not import pytorch modules ...")
  sys.exit()

import numpy as np
# from sklearn import datasets
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from net import Encoder,AttentionDecoder
import pickle

from utils import save_model,load_model

# https://qiita.com/m__k/items/b18756628575b177b545
CHAR2ID = { str(i): i for i in range(10)} # "0": 0 のような形
CHAR2ID.update({" ":10,"-": 11,"_":12})
ID2CHAR = { i : str(i) for i in range(10)}
ID2CHAR.update({10:"", 11:"-", 12:""})

# module ---------------------
def generate_number():
  n = [ random.choice(list("0123456789")) for _ in range(random.randint(1,3))]
  return int("".join(n))

def add_padding(n, is_input = True):
  if is_input:
    num = "{: <7}".format(n)
  else:
    num = "{: <5}".format(n)
  return num
#-----------------------------

def dataset(N=50000):
  url = "https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-2/master/dataset/date.txt"
  res = requests.get(url)
  data_path = "./date.txt"
  
  if not os.path.exists(data_path):
    with open(data_path, "w") as f:
      f.write(res.text)
  _idd,_odd =[],[]
  with open(data_path, "r") as f:
    _date = f.readlines()
    for date in _date:
      idd = date.split("_")[0] 
      odd = "_" + date.split("_")[1]
      
      _idd.append(idd), _odd.append(odd)
  
  return _idd,_odd
  
  
def getLoader(X,y,batch_size=4):
  # from torch.utils.data import DataLoader, TensorDataset
  
  # lbl = torch.LongTensor(_ll2) #大きな値の分類について(整数型で格納して利用するというもの)
  X = torch.LongTensor(X)
  y = torch.LongTensor(y)
  dataset = TensorDataset(X,y)
  loader = DataLoader(dataset,batch_size=batch_size)

  return loader


def train(model_set,dataloader):
  encoder, decoder = model_set["model"]
  criterion = model_set["criterion"]
  en_opt, de_opt = model_set["optimizer"]
  
  
  _loss_per_loader  = []
  for X,y in dataloader:
    X,y = X.to(device),y.to(device)
    # print(X.size())
    # print(y.size())
    # print(X[0,:], y[0,:])
    # sys.exit()
    en_opt.zero_grad()
    de_opt.zero_grad()
    
    decoder_input = y[:,:-1]
    decoder_target = y[:,1:]
    # print(decoder_input[0,:], decoder_target[0,:])
    hs,h = encoder(X)
    # print(encoder_state[0].size(),encoder_state[1].size()) #torch.Size([1, 40, 128]) torch.Size([1, 40, 128])
    decoder_output , _ ,attention_weight = decoder(decoder_input, hs,h)
    
    loss = 0
    for j in range(decoder_output.size()[1]):
      loss += criterion(decoder_output[:,j,:], decoder_target[:,j])
    epoch_loss = loss.item()
    _loss_per_loader.append(epoch_loss)
    # print(decoder_output.size())
    # print(decoder_output[0,:,:])
    # print(decoder_target[0,:])
    loss.backward()
    en_opt.step()
    de_opt.step()
  
  loss_value = np.round(np.mean(_loss_per_loader),4)
  model_set ={
    "model" : [encoder,decoder],
    "criterion" : criterion,
    "optimizer" : [en_opt,de_opt]
  }
  return loss_value,model_set

def valid(model_set,dataloader):
  encoder, decoder = model_set["model"]
  batch_size = model_set["batch_size"]
  
  def get_max_index(decoder_output):
    _res = []
    for h in decoder_output:
      idx = torch.argmax(h)
      _res.append(idx)
    
    res_tensror = torch.tensor(_res, device=device).view(batch_size,1)
    return res_tensror
  
  _pred=[]
  with torch.no_grad(): #勾配の計算はしない
    
    for X,y in dataloader:
      X,y = X.to(device),y.to(device)
    
      encoder_state = encoder(X)
      start_char_batch = [ [CHAR2ID["_"]] for _ in range(batch_size)]
      decoder_input_tensor = y = torch.LongTensor(start_char_batch, device=device)
      # 変数名の変換 --->
      decoder_hidden = encoder_state #共通で利用するencoderのoutputを利用する2021.11.2
      # batch　毎の結果を格納する容器
      batch_tmp = torch.zeros(batch_size,1, dtype=torch.long,device=device)
      
      for _ in range(5):#encode(state) & "_"　から文字の生成を行う
        """
        初期のdecoder_hiddenには、encoder_outputの値を入れて、2文字目からは1文字目の出力の値を格納
        """
        decoder_output,decoder_hidden = decoder(decoder_input_tensor,decoder_hidden)
        decoder_out_next = get_max_index(decoder_output.squeeze())
        # print(decoder_output.size())# [400,1,13]
        # print(decoder_output.squeeze().size()) #[400,13] 1次元削除
        batch_tmp = torch.cat([batch_tmp,decoder_out_next], dim=1)
      
      _pred.append(batch_tmp[:,1:]) #"_"を削除して次
  
  return _pred

def set_char2id(_input,_output):
  """ 2021.11.2. -> 登場する単語のIDを登録する試み """
  CHAR2ID = {}
  for input_char, outout_char in zip(_input,_output):
    for c in input_char: #1文字ずつ区切って利用するというもの
      if not c in CHAR2ID:
        CHAR2ID[c] = len(CHAR2ID)
    for c in outout_char:
      if not c in CHAR2ID:
        CHAR2ID[c] = len(CHAR2ID)
  ID2CHAR = { v:k for k,v in CHAR2ID.items()}
  return CHAR2ID,ID2CHAR

    


#--- TRAIN ------------------------------------------------------
def main(N=100, batch_size=40, EPOCH=100):
  _idd, _odd = dataset()
  CHAR2ID,ID2CHAR = set_char2id(_idd,_odd)
  
  
  
  _idd2 = [[ CHAR2ID[c] for c in input_char] for input_char in _idd ] 
  _odd2 = [[ CHAR2ID[c] for c in output_char] for output_char in _odd ]
  # print(len(char2id))
  # print(len(_idd), len(_odd))
  # print(len(_idd2[0]),len(_odd2[0]))
  # sys.exit()
  x_train,x_test,y_train,y_test = train_test_split(_idd2, _odd2, train_size=0.7)
  # print(x_train[0],"<-->",[ ID2CHAR[c] for c in x_train[0]])
  # print(y_train[0],"<-->",[ ID2CHAR[c] for c in y_train[0]])
  # sys.exit()
  train_loader = getLoader(x_train,y_train,batch_size=batch_size)
  test_loader = getLoader(x_test,y_test,batch_size=batch_size)
  # print(train_loader)
  # print("train->", x_train.shape[0],"test->", x_test.shape[0])
  # sys.exit()
  #-setting ---------
  embedding_dim = 200 #文字の埋込次元数
  hidden_dim = 128 #LSTMの隠れ層サイズ
  vocab_size = len(CHAR2ID)
  # print(embedding_dim, hidden_dim,vocab_size)

  # model ----------
  encoder = Encoder(vocab_size, embedding_dim,hidden_dim,CHAR2ID).to(device)
  decoder = AttentionDecoder(vocab_size, embedding_dim,hidden_dim, batch_size,CHAR2ID).to(device)
  # loss function ----
  criterion = nn.CrossEntropyLoss()
  # "optimizer" -----
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
  decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

  model_set ={
    "model" : [encoder,decoder],
    "criterion" : criterion,
    "optimizer" : [encoder_optimizer,decoder_optimizer]
  }
  
  _loss =[]
  _epoch = []
  for epoch in range(EPOCH):
    train_loss, model_set = train(model_set,train_loader)
    print("Epoch %d: %.2f" % (epoch, train_loss))
    _loss.append(train_loss)
    _epoch.append(epoch+1)
    
    if train_loss < 0.009:
      break
  # ------------------------------------
  # csv file ----> 
  if 1:
    df = pd.DataFrame()
    df["EPOCH"] = np.array(list(range(EPOCH))) + 1
    df["train_LOSS"] = _loss
    df.to_csv("./torch_07.csv", index=False)
  # ------------------------------------
  # png file ---->
  if 1:
    f,ax = plt.subplots(figsize=(15,7))
    ax.plot(df["EPOCH"], df["train_LOSS"] ,label="train_LOSS", marker = "o")
    ax.legend()
    f.savefig("./torch_07.png", bbox_inches="tight")
  #----------------------
  # model save
  encoder, decoder = model_set["model"]
  save_model("./encoder_attention.pkl",encoder)
  save_model("./decoder_attention.pkl",decoder)
  # ------------------------------------
  return
  # print(x_train.shape,y_train.shape)
  # print(x_test.shape,y_test.shape)
#------検証の結果を表示するようなprogramについて----
def main2(N=2000, batch_size=400, EPOCH=100):
  _input, _output = dataset(N=N)
  # x_train,x_test,y_train,y_test = train_test_split(_input, _output, train_size=0.7)
  data_loader = getLoader(_input,_output,batch_size=batch_size)
  encoder = load_model("./encoder_fitted.pkl")
  decoder = load_model("./decoder_fitted.pkl")
  model_set ={"model" : [encoder,decoder],"batch_size" : batch_size}
  #main routine ---
  _pred = valid(model_set,data_loader)
  
  _row =[]
  for i,(X,y) in enumerate(data_loader):
    pred = _pred[i]
    # print(X.size(),y.size(), pred.size()) [40, 7])[40, 5])[40, 5])
    for in_x,true_y,pred_y in zip(X,y,pred):
      x = [ ID2CHAR[i.item()] for i in in_x]
      y = [ ID2CHAR[i.item()] for i in true_y]
      p = [ ID2CHAR[i.item()] for i in pred_y]

      x_str = "".join(x)
      y_str = "".join(y)
      p_str = "".join(p)
      
      judge = "o" if y_str == p_str else "x"
      
      _row.append([x_str,y_str,p_str, judge])
  df =  pd.DataFrame(_row,columns = ["input","answer","predict","judge"])
  n_acc = len(df[df["judge"]== "o"])
  print(f"acc[%] = {n_acc*100/len(df):.3f}")
  print("MISTAKE(30)","-"*30)
  print(df[df["judge"]=="x"].head(30))
  sys.exit()

if __name__ == "__main__":
  # fit model ---->
  main(N=50000, batch_size=500, EPOCH=30) #train
  # main2(N=1000, batch_size=40) #test
  
  
