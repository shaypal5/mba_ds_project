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
    SENTIMENT = 'SENTIMENT'
    SENTIMENT_0 = 'SENTIMENT_0'
    SENTIMENT_1 = 'SENTIMENT_1'
    STATUS_ORD = 'STATUS_ORD'


CATEGORICAL_COLUMNS = [
    Column.STATUS_PANTINUM,
    Column.STATUS_GOLD,
    Column.STATUS_SILVER,
    Column.SERVICE_FLAG,
    Column.CANCEL_FLAG,
    Column.CREDIT_FLAG,
    Column.RECSYS_FLAG,
    Column.SENTIMENT_0,
    Column.SENTIMENT_1,
]

NUMERIC_COLUMNS = [
    Column.OTHER_SITE_VALUE,
    Column.NUM_DEAL,
    Column.LAST_DEAL,
    Column.ADVANCE_PURCHASE,
    Column.FARE_L_Y1,
    Column.FARE_L_Y2,
    Column.FARE_L_Y3,
    Column.FARE_L_Y4,
    Column.FARE_L_Y5,
    Column.POINTS_L_Y1,
    Column.POINTS_L_Y2,
    Column.POINTS_L_Y3,
    Column.POINTS_L_Y4,
    Column.POINTS_L_Y5,
]

ORDINAL_COLUMNS = [
    Column.STATUS_ORD,
]


class FeatureGroup:
    FARE = 'FARE'
    POINTS = 'POINTS'


FEATURE_GROUPS = {
    FeatureGroup.FARE: [
        Column.FARE_L_Y1,
        Column.FARE_L_Y2,
        Column.FARE_L_Y3,
        Column.FARE_L_Y4,
        Column.FARE_L_Y5,
    ],
    FeatureGroup.POINTS: [
        Column.POINTS_L_Y1,
        Column.POINTS_L_Y2,
        Column.POINTS_L_Y3,
        Column.POINTS_L_Y4,
        Column.POINTS_L_Y5,
    ],
}


class ContextKey:
    REVIEWS_FPATH = 'reviews_fpath'
