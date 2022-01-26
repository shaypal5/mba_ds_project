"""Data functionalities for mba."""

import os

import pandas as pd

from .shared import (
    DATA_DPATH,
)


def get_text_train_df():
    fpath = os.path.join(DATA_DPATH, 'text_training.csv')
    return pd.read_csv(fpath)


def get_text_rollout_df():
    fpath = os.path.join(DATA_DPATH, 'text_rollout_X.csv')
    return pd.read_csv(fpath)
