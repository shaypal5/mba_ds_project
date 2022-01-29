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


def get_reviews_train_df_fpath():
    return os.path.join(DATA_DPATH, 'reviews_training.csv')


def get_reviews_train_df():
    return pd.read_csv(get_reviews_train_df_fpath())


def get_reviews_rollout_fpath():
    return os.path.join(DATA_DPATH, 'reviews_rollout.csv')


def get_reviews_rollout_df():
    return pd.read_csv(get_reviews_rollout_fpath())


def get_ffp_train_df():
    fpath = os.path.join(DATA_DPATH, 'ffp_train.csv')
    return pd.read_csv(fpath)


def get_ffp_rollout_fpath():
    return os.path.join(DATA_DPATH, 'ffp_rollout_X.csv')


def get_ffp_rollout_df():
    return pd.read_csv(get_ffp_rollout_fpath())


def get_recommendations_fpath():
    return os.path.join(DATA_DPATH, 'recommendations.csv')


def get_recommendations_template_fpath():
    return os.path.join(DATA_DPATH, 'recommendations_template.csv')
