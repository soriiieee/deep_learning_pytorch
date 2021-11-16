"""
参考のtutorials
https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
"""

import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
# os.chdir("../../..")
import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

DATAHOME="/Users/soriiieee/work2/sci/d0727/deep_learning_pytorch/dat/lightning"

# -- dataset downloads ---- #
from utils_data import load_data

def check_data():
  df = get_stallion_data()
  print(df.head())
  print(df.columns)
  ['agency', 'sku', 'volume', 'date', 'industry_volume', 'soda_volume',
       'avg_max_temp', 'price_regular', 'price_actual', 'discount',
       'avg_population_2017', 'avg_yearly_household_income_2017', 'easter_day',
       'good_friday', 'new_year', 'christmas', 'labor_day', 'independence_day',
       'revolution_day_memorial', 'regional_games', 'fifa_u_17_world_cup',
       'football_gold_cup', 'beer_capital', 'music_fest',
       'discount_in_percent', 'timeseries']
  print(df["date"].dt.year )
  return



def set_data():
  df = load_data(reset=False)
  # print(df.sample(5,random_state=1))
  print(df.describe())
  # data.sample(10, random_state=521)
  # print(df.columns)
  
#commands ---
# check_data()
set_data()



