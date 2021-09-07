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

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets

# 0) prepair data
x_numpy,y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state = 1) 
X = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# 1) model
input_size =n_features
output_size = 1
model = nn.Linear(input_size,output_size)

# 2) loas and optimizer
learning_rate= 0.01
loss_func = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 3) 
num_epoch = 100
for epoch in range(num_epoch):
  #forward pass and loss
  y_pred = model(X)
  loss = loss_func(y_pred, y)
  #backward path"
  loss.backward()
  
  #update
  optimizer.step()
  
  optimizer.zero_grad()
  if (epoch + 1)%10==0:
    print(f'epoch: {epoch + 1}, loss = {loss}')


predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, "ro")
plt.plot(x_numpy, predicted, "b")
plt.savefig("../out/torch_07.png", bbox_inches="tight")
