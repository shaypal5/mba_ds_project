"""Sentiment classification for MBA."""

import os
import pickle
from typing import List

import pandas as pd
from pycaret.classification import (
    setup,
    compare_models,
    blend_models,
    predict_model,
    finalize_model,
    save_model,
    load_model,
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score,
)

from .data import (
    get_text_train_df,
)
from .shared import (
    MODELS_DPATH,
)


SENT_MODEL_FPATH_NO_EXT = os.path.join(MODELS_DPATH, 'sentiment_clf')
SENT_MODEL_FPATH = SENT_MODEL_FPATH_NO_EXT + '.pkl'
SENT_MODEL_COL_SCHEMA_FPATH = os.path.join(
    MODELS_DPATH, 'sentiment_col_schema.pkl')

METRICS = (
    accuracy_score, roc_auc_score, recall_score, precision_score, f1_score)


class SentimentPredictor():

    def __init__(
        self,
        column_schema: List[object],
        pycaret_model: object,
    ):
        self.column_schema = column_schema
        self.pycaret_model = pycaret_model

    def predict(self, df: pd.DataFrame) -> pd.Series:
        sub_df = df[self.column_schema]
        prediction = predict_model(self.pycaret_model, data=sub_df)
        return prediction['Label']


def train_sentiment_model_and_save_to_disk():
    print("Training a sentiment model:")
    train_df = get_text_train_df()
    train_df = train_df.drop('ID', axis=1)
    X = train_df.drop(['rating'], axis=1)
    sentiment_column_schema = list(X.columns)
    setup(
        data=train_df,
        target='rating',
        train_size=0.8,
        session_id=1337,
        numeric_features=list(X.columns),
        normalize=True,
        remove_perfect_collinearity=True,
        data_split_stratify=True,
        silent=True,
    )
    top3 = compare_models(n_select=3, sort='kappa')
    print("Top 3 models:")
    print(top3[0])
    print(top3[1])
    print(top3[2])
    blender_top3 = blend_models(top3)
    res = predict_model(blender_top3)
    y_true = res['rating']
    y_pred = res['Label']
    for metric in METRICS:
        print(metric.__name__)
        print(metric(y_true, y_pred))
    final_blended = finalize_model(blender_top3)
    save_model(
        model=final_blended,
        model_name=SENT_MODEL_FPATH_NO_EXT,
        verbose=True,
    )
    with open(SENT_MODEL_COL_SCHEMA_FPATH, 'wb+') as f:
        pickle.dump(
            obj=sentiment_column_schema,
            file=f,
        )


SENT_MODEL = None


def _get_sentiment_predictor_from_file() -> SentimentPredictor:
    global SENT_MODEL
    with open(SENT_MODEL_COL_SCHEMA_FPATH, 'rb') as f:
        sentiment_column_schema = pickle.load(f)
    pycaret_model = load_model(SENT_MODEL_FPATH_NO_EXT)
    SENT_MODEL = SentimentPredictor(
        column_schema=sentiment_column_schema,
        pycaret_model=pycaret_model,
    )
    return SENT_MODEL


def get_sentiment_predictor() -> SentimentPredictor:
    """Returns a SentimentPredictor model, training it if it can't be found."""
    global SENT_MODEL
    if SENT_MODEL is not None:
        return SENT_MODEL
    if os.path.isfile(SENT_MODEL_FPATH) and os.path.isfile(
            SENT_MODEL_COL_SCHEMA_FPATH):
        return _get_sentiment_predictor_from_file()
    # else it hasn't been trained yet, or has been moved, so re-train
    train_sentiment_model_and_save_to_disk()
    if os.path.isfile(SENT_MODEL_FPATH) and os.path.isfile(
            SENT_MODEL_COL_SCHEMA_FPATH):
        return _get_sentiment_predictor_from_file()
    raise Exception("SentimentPredictor generation failed!")
