# -*- coding: utf-8 -*-
"""
torch のseq2seqモデルの実装を行うようにするprogram
date: 2021.10.29
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

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
# from sklearn import datasets
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
  def __init__(self,vocab_size, embedding_dim, hidden_dim, CHAR2ID):
    super(Encoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = CHAR2ID[" "])
    # "1"　という情報を200次元ベクトルに拡張して行うようなもの 
    # self.lstm = nn.LSTM(embedding_dim,hidden_dim, batch_first=True)
    self.gru = nn.GRU(embedding_dim,hidden_dim, batch_first=True)
  
  def forward(self,seq):
    embedding = self.word_embeddings(seq) #seq(N, 7[seq-len]) -> (N,7[seq-len],200)
    # _, state = self.lstm(embedding) # seq2seq Modelのみの実装を行った
    # return state
    hs, h = self.gru(embedding)
    return hs, h
  
class Decoder(nn.Module):
  def __init__(self,vocab_size, embedding_dim, hidden_dim,CHAR2ID):
    super(Decoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = CHAR2ID[" "])
    # "1"　という情報を200次元ベクトルに拡張して行うようなもの 
    self.lstm = nn.LSTM(embedding_dim,hidden_dim, batch_first=True)
    # self.gru = nn.GRU(embedding_dim,hidden_dim, batch_first=True)
    
    self.hidden2linear = nn.Linear(hidden_dim, vocab_size) #128次元ベクトルへ射影するので、それを13次元に落とし込むようなlayerにsetthingする
  
  def forward(self,seq,encoder_state):
    """
    --- 2021.10.29 ---
      seq : decoder で入力するinput
      encoder_state : encoderの最終出力層
    """
    embedding = self.word_embeddings(seq) #seq(N,7value) -> (N,7values,200)
    output, state = self.lstm(embedding, encoder_state)
    # state(h,c)
    output = self.hidden2linear(output)
    return output,state
  
class AttentionDecoder(nn.Module):
  def __init__(self,vocab_size, embedding_dim, hidden_dim, batch_size,CHAR2ID):
    super(AttentionDecoder, self).__init__()
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = CHAR2ID[" "])
    # "1"　という情報を200次元ベクトルに拡張して行うようなもの 
    self.gru = nn.GRU(embedding_dim,hidden_dim, batch_first=True)
    # self.gru = nn.GRU(embedding_dim,hidden_dim, batch_first=True)
    
    # Attentition + GRUの出力レイヤーの*2次元分のサイズ確保-----
    self.hidden2linear = nn.Linear(hidden_dim*2, vocab_size)
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self,seq,hs,h):
    """
    --- 2021.10.29 ---
      seq : decoder で入力するinput
      hs :　encoder層の各々cell出力についてを実装する
      h : encoderの最終出力層
    """
    embedding = self.word_embeddings(seq) #seq(N,7value) -> (N,7values,200)
    # output, state = self.lstm(embedding,h)
    output, state = self.gru(embedding,h)
    # ---------------------------------
    # Attention 層　---
    # hs.size() = [100,29,128]
    # output.size() = [100,10,128] #softmaxの関数で表示するもの
    # ---------------------------------
    t_output = torch.transpose(output,1,2) # 転置する行列
    s = torch.bmm(hs,t_output) #s.size() -> [100,29,10]
    attention_weight = self.softmax(s)
    
    # Context -> Vectorの為の入れ物を用意する----
    c = torch.zeros(self.batch_size, 1, self.hidden_dim, device=device)
    # c.size [ 100,1,128] #一番参照がされやすい注目単語(attention)のベクトルを表示
    for i in range(attention_weight.size()[2]):
      unsq_weight = attention_weight[:,:,i].unsqueeze(2)
      weight_hs = hs * unsq_weight #[100,29,128]
      
      weight_sum = torch.sum(weight_hs, axis=1).unsqueeze(1)
      
      c = torch.cat([c,weight_sum], dim=1)
    
    c = c[:,1:,:]
    output = torch.cat([output,c],dim=2)
    output = self.hidden2linear(output)
    return output, state, attention_weight
  
  
    
    
if __name__ == "__main__":
  N=100
  main(N)
  
  
