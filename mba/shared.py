"""Shared functions for mba."""

import os
from os.path import dirname, abspath


REPO_DPATH = dirname(dirname(abspath(__file__)))
DATA_DPATH = os.path.join(REPO_DPATH, 'data')
MODELS_DPATH = os.path.join(REPO_DPATH, 'models')


class Column:
    ID = 'ID'
    OTHER_SITE_VALUE = 'OTHER_SITE_VALUE'
    STATUS_PANTINUM = 'STATUS_PANTINUM'
    STATUS_GOLD = 'STATUS_GOLD'
    STATUS_SILVER = 'STATUS_SILVER'
    NUM_DEAL = 'NUM_DEAL'
    LAST_DEAL = 'LAST_DEAL'
    ADVANCE_PURCHASE = 'ADVANCE_PURCHASE'
    FARE_L_Y1 = 'FARE_L_Y1'
    FARE_L_Y2 = 'FARE_L_Y2'
    FARE_L_Y3 = 'FARE_L_Y3'
    FARE_L_Y4 = 'FARE_L_Y4'
    FARE_L_Y5 = 'FARE_L_Y5'
    POINTS_L_Y1 = 'POINTS_L_Y1'
    POINTS_L_Y2 = 'POINTS_L_Y2'
    POINTS_L_Y3 = 'POINTS_L_Y3'
    POINTS_L_Y4 = 'POINTS_L_Y4'
    POINTS_L_Y5 = 'POINTS_L_Y5'
    SERVICE_FLAG = 'SERVICE_FLAG'
    CANCEL_FLAG = 'CANCEL_FLAG'
    CREDIT_FLAG = 'CREDIT_FLAG'
    RECSYS_FLAG = 'RECSYS_FLAG'
    BUYER_FLAG = 'BUYER_FLAG'
    SENTIMENT = 'sentiment'
    SENTIMENT_0 = 'sentiment_0'
    SENTIMENT_1 = 'sentiment_1'


class ContextKey:
    REVIEWS_FPATH = 'reviews_fpath'
