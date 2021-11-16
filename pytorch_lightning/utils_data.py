import os,sys
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
from pytorch_forecasting.data.examples import get_stallion_data

def load_data(preprocess=True,reset=False):
  def preprocess(df):
    df["time_idx"] = df["date"].dt.year * 12 + df["date"].dt.month
    df["time_idx"] -= df["time_idx"].min()
    # print(df["time_idx"].head())
    # add category ---
    df["month"] = df["date"].dt.month.astype(str).astype("category")
    df["log_volume"] = np.log(df.volume + 1e-8)
    df["avg_volume_by_sku"] = df.groupby(["time_idx", "sku"], observed=True).volume.transform("mean")
    df["avg_volume_by_agency"] = df.groupby(["time_idx", "agency"], observed=True).volume.transform("mean")
    # print(df.head())

    special_days = ["easter_day","good_friday","new_year","christmas","labor_day","independence_day","revolution_day_memorial","regional_games","fifa_u_17_world_cup","football_gold_cup","beer_capital","music_fest"]
    df[special_days] = df[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    return df
  #----- commands ----
  if not os.path.exists(f"{DATAHOME}/stallion.csv") or reset:
    df = get_stallion_data()
    # df.to_csv(f"{DATAHOME}/stallion.csv", index=False)
    df.to_csv(f"{DATAHOME}/stallion.csv")
  else:
    df = pd.read_csv(f"{DATAHOME}/stallion.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop(["Unnamed: 0"],axis=1)
  if preprocess:
    df = preprocess(df)
  return df





