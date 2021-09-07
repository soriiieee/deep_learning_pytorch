# -*- coding: utf-8 -*-
# when   : 2020.0x.xx
# who : [sori-machi]
# what : [ ]
#---------------------------------------------------------------------------
# basic-module
import sys,os,re,glob
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

import matplotlib
import matplotlib.pyplot as plt



# 0) prepair data
bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target



n_samples, n_features = X.shape
# print(n_samples, n_features)
# preprocessing
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
y_test2 = y_test.copy()
#sklearn prediction
lr = LogisticRegression(random_state=0).fit(x_train,y_train)
y_pred_sk = lr.predict(x_test)



x_train,x_test = torch.from_numpy(x_train.astype(np.float32)),torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_train = y_train.view(y_train.shape[0],1)

y_test = torch.from_numpy(y_test.astype(np.float32))
y_test = y_test.view(y_test.shape[0],1)
# print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
# sys.exit()

#sklearn -learning
# 1) model

class LogisticRegression(nn.Module):
  def __init__(self,n_input_features):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(n_input_features, 1)
  
  def forward(self,x):
    y_pred = torch.sigmoid(self.linear(x))
    return y_pred
model = LogisticRegression(n_features)
# print(model)
# sys.exit()
# 2) loas and optimizer
learning_rate = 0.01
loss_func = nn.BCELoss() #binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) trainig loop
num_epoch =100
for epoch in range(num_epoch):
  y_pred = model(x_train)
  loss = loss_func(y_pred,y_train)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  # if (epoch+1)%50 ==0:
  #   print(f"epoch = {epoch +1} | loss= {loss.item()}")
  

with torch.no_grad():
  y_pred_test =model(x_test)
  y_pred_cls = y_pred_test.round()
  acc = y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])

  # print(y_pred_sk.shape)
  # print(y_test2.shape)
  # sys.exit()
  # print((y_pred_sk== y_test2).sum())
  # sys.exit()
  num_test = float(y_test2.shape[0])
  acc_sk = (y_pred_sk== y_test2).sum() / num_test
  print(f"torch-acc = {acc.item()} | sklearn-acc = {acc_sk}")    
  sys.exit()
