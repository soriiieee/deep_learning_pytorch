# -*- coding: utf-8 -*-
# when   : 2021.09.10
# who : [sori-machi]
# what : [学習用のpurogramの実装を行う]
# p305 に記載されている、seq2se2の実装
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys,os,re,glob
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.simplefilter('ignore')

#amedas relate 2020.02,04 making...

#(code,ini_j,out_d)/(code,path,csv_path)
#---------------------------------------------------
import subprocess


sys.path.append("..")
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.base_model import BaseModel
from common.layers import *
from common.util import eval_seq2seq
from seq2seq import Seq2seq # 2021年作成データ



(x_train,y_train),(x_test, y_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(char_to_id, id_to_char)
#ハイパーパラメータの設定を行う
vocab_size = len(char_to_id) #←　登場する文字の種類とそのid番号の対応表
# print(vocab_size)
wordvec_size = 16 # <- 最大の文字数のサイズ(paddingを実施する必要があるので、このようにしておく)
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5


# model & optimizer / trainer の設定
model = Seq2seq(vocab_size, wordvec_size, max_epoch)

# train program 
_acc = [] # epoch に応じて、出ry九するaccの格納用のprogramの実施を行う予定
for epoch in range(max_epoch):
  trainer.fit() 