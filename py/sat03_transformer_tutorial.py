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


import math
from typing import Tuple
import torch
from torch import nn,Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class TransformerModel(nn.Module):
  def __init__(self,ntoken: int, d_model: int, d_hid,
               nlayers: int, dropout: float=0.5):
    super().__init__()
    self.model_type = 'Transformer'
    self.pos_encoder = PositionnalEncoder(d_model, dropout) #making
    encoder_layers = TransformerEncoderLayer(d_model,hhead, d_hid,dropout)
    self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
    self.encoder = nn.Embedding(ntoken,d_model)
    self.d_model = d_model
    self.decoder = nn.Linear(d_model, ntoken)
    
    self.init_weights() #function
    
  def init_weights(self) -> None: #戻り値には何もなし、setting関数
    initrange = 1
    self.encoder.weight.data.uniform_(-1 * initrange, initrange)
    self.decoder.bias.data.zero_()
    self.decoder.weight.data.uniform_(-1*initrange, initrange)
  
  def forward(self,src: Tensor, src_mask: Tensor) -> Tensor:
    """
    args:
      SRC [seq_len, batchsize]
      src_mack [ seq_len,seq_len]
      
    return:
      OUT [seq_len,batchsize,ntoken]
    """
    src = self.encoder(src) * math.sqrt(self.d_model)
    src = self.pos_encoder(src) #position encoder
    out = self.transformer_encoder(src,src_mask)
    out = self.decoder(out)
    return out
  
  def generate_square_subsequent_mask(sz: int) -> Tensor:
    return torch.triu(torch.ones(sz,sz) * float('-inf'), diagonal=1)
  
class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ =="__main__":
  main()