"""Buyer classification for MBA."""

import os
import pickle
from typing import Optional

import pdpipe
import numpy as np
import pandas as pd
from pycaret.classification import (
    setup,
    create_model,
    tune_model,
    predict_model,
    finalize_model,
    save_model,
    load_model,
    add_metric,
)
from imblearn.over_sampling import RandomOverSampler

from .data import (
    get_ffp_train_df,
    get_ffp_rollout_fpath,
    get_reviews_train_df_fpath,
    get_reviews_rollout_fpath,
    get_recommendations_fpath,
    get_recommendations_template_fpath,
)
from .shared import (
    Column,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    # ORDINAL_COLUMNS,
    FEATURE_GROUPS,
    ContextKey,
    MODELS_DPATH,
)
from .pipeline import build_pipeline
from .sentiment import METRICS


# Metrics constants

TP_REVENUE = 32.7
FP_REVENUE = -6.05
FN_COST = -32.7


# Custom metric functions

def p_count(y_true, y_pred):
    return sum(y_true == 1)


def n_count(y_true, y_pred):
    return sum(np.where((y_true == 0), 1, 0))


def tp(y_true, y_pred):
    return sum(np.where((y_pred == 1) & (y_true == 1), 1, 0))


def fp(y_true, y_pred):
    return sum(np.where((y_pred == 1) & (y_true == 0), 1, 0))


def tn(y_true, y_pred):
    return sum(np.where((y_pred == 0) & (y_true == 0), 1, 0))


def fn(y_true, y_pred):
    return sum(np.where((y_pred == 0) & (y_true == 1), 1, 0))


def revenue_score(y_true, y_pred):
    tp_count = tp(y_true, y_pred)
    fp_count = fp(y_true, y_pred)
    return tp_count * TP_REVENUE + fp_count * FP_REVENUE


def opportunity_cost(y_true, y_pred):
    tp_count = tp(y_true, y_pred)
    fp_count = fp(y_true, y_pred)
    fn_count = fn(y_true, y_pred)
    return tp_count * TP_REVENUE + fp_count * FP_REVENUE + fn_count * FN_COST


# model in-memory handle

BUY_MODEL_FPATH_NO_EXT = os.path.join(MODELS_DPATH, 'buyer_clf')
BUY_MODEL_FPATH = BUY_MODEL_FPATH_NO_EXT + '.pkl'
PIPELINE_FPATH = os.path.join(MODELS_DPATH, 'buyer_pipeline.pkl')


PIPELINE: pdpipe.PdPipeline = None
BUY_MODEL = None


def train_buyer_model_and_save_to_file():
    global PIPELINE, BUY_MODEL
    print("Starting buyer model workflow")
    print("Loading data...")
    raw_tdf = get_ffp_train_df()
    print("Label value counts:")
    print(raw_tdf.BUYER_FLAG.value_counts())
    print("Building pipeline...")
    pline = build_pipeline()
    print("Pipeline:")
    print(pline)
    print("Transforming raw data with pipeline...")
    tdf = pline.fit_transform(
        X=raw_tdf,
        verbose=True,
        context={
            ContextKey.REVIEWS_FPATH: get_reviews_train_df_fpath(),
        },
    )
    PIPELINE = pline

    print("Setting up pycaret classification problem...")
    setup(
        data=tdf,
        target=Column.BUYER_FLAG,
        train_size=0.8,
        session_id=42,
        numeric_features=NUMERIC_COLUMNS,
        categorical_features=CATEGORICAL_COLUMNS,
        group_features=[FEATURE_GROUPS[k] for k in FEATURE_GROUPS],
        group_names=[k for k in FEATURE_GROUPS],
        normalize=True,
        remove_perfect_collinearity=True,
        data_split_stratify=True,
        silent=True,
        fix_imbalance=True,
        fix_imbalance_method=RandomOverSampler(),
    )

    print("Adding custom metrics...")
    add_metric(
        id='p_count',
        name='P',
        score_func=p_count,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='n_count',
        name='N',
        score_func=n_count,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='revenue_score',
        name='Total Revenue',
        score_func=revenue_score,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='opportunity_cost',
        name='Opportunity Cost',
        score_func=opportunity_cost,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='tp',
        name='TP',
        score_func=tp,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='fp',
        name='FP',
        score_func=fp,
        target='pred',
        greater_is_better=False,
        multiclass=False,
    )
    add_metric(
        id='tn',
        name='TN',
        score_func=tn,
        target='pred',
        greater_is_better=True,
        multiclass=False,
    )
    add_metric(
        id='fn',
        name='FN',
        score_func=fn,
        target='pred',
        greater_is_better=False,
        multiclass=False,
    )

    print("Creating base GBC classifier..")
    gbc = create_model('gbc')
    print("Tuning GBC classifier...")
    sktuned_gbc = tune_model(
        gbc,
        fold=8,
        n_iter=10,
        optimize='revenue_score',
    )
    print("Resulting GBC classifier and params:")
    print(sktuned_gbc)
    print("Predicting on test set:")
    res = predict_model(sktuned_gbc)
    y_true = res[Column.BUYER_FLAG]
    y_pred = res['Label']
    print("Results on test set:")
    metrics = list(METRICS) + [tp, fp, tn, fn, revenue_score, opportunity_cost]
    for metric in metrics:
        print(metric.__name__)
        print(metric(y_true, y_pred))

    # final model
    print("Finalizing GBC model...")
    final_gbc = finalize_model(sktuned_gbc)
    BUY_MODEL = final_gbc
    save_model(
        model=final_gbc,
        model_name=BUY_MODEL_FPATH_NO_EXT,
        verbose=True,
    )
    with open(PIPELINE_FPATH, 'wb+') as f:
        pickle.dump(
            obj=pline,
            file=f,
        )


def load_buyer_model_from_file():
    global PIPELINE, BUY_MODEL
    if (not os.path.isfile(PIPELINE_FPATH)) or (
            not os.path.isfile(BUY_MODEL_FPATH)):
        print(
            "Either pipeline or model are missing from disk! "
            "Starting buyer model training workflow!"
        )
        train_buyer_model_and_save_to_file()
    else:
        print("Pipeline and model found on disk! Loading...")
        with open(PIPELINE_FPATH, 'rb') as f:
            PIPELINE = pickle.load(f)
        BUY_MODEL = load_model(BUY_MODEL_FPATH_NO_EXT)
    print("Done! Pipeline and model loaded to memory!")


def predict_buyer_flag(
    input_fpath: Optional[str],
    input_reviews_fpath: Optional[str],
    output_fpath: Optional[str],
):
    if input_fpath is None:
        input_fpath = get_ffp_rollout_fpath()
    if input_reviews_fpath is None:
        input_reviews_fpath = get_reviews_rollout_fpath()
    if output_fpath is None:
        output_fpath = get_recommendations_fpath()
    print("Predicting buyer flag!")
    print(f"for input: {input_fpath}")
    print(f"for input reviews: {input_reviews_fpath}")
    print(f"for output: {output_fpath}")
    if PIPELINE is None or BUY_MODEL is None:
        print("Loading pipeline and model from disk...")
        load_buyer_model_from_file()
        print("Done loading pipeline and model!")
    else:
        print("Pipeline and model already loaded!")
    print("Reading input csv...")
    input_df = pd.read_csv(input_fpath)
    print("Preprocessing input csv...")
    input_df = PIPELINE.transform(
        X=input_df,
        verbose=True,
        context={ContextKey.REVIEWS_FPATH: input_reviews_fpath}
    )
    input_df = input_df.drop([Column.BUYER_FLAG], axis=1)
    print("Producing buyer predictions...")
    pred_res = predict_model(BUY_MODEL, input_df)
    pred_y = pred_res['Label']
    print("Prediction value counts:")
    print(pred_y.value_counts())
    print("Reading recommendations template file...")
    output_df = pd.read_csv(get_recommendations_template_fpath())
    print("Making sure formats match...")
    assert len(pred_y) == len(output_df)
    assert list(input_df.index) == list(output_df[Column.ID])
    print("Building predictions output table...")
    output_df[Column.BUYER_FLAG] = list(pred_y)
    print("Making sure table was built correctly...")
    assert list(output_df[Column.BUYER_FLAG]) == list(pred_y)
    assert sum(output_df[Column.BUYER_FLAG]) == sum(pred_y)
    output_fpath = get_recommendations_fpath()
    print(f"Writing predictions table to {output_fpath}...")
    output_df.to_csv(output_fpath, index=False)
    print("Done!")
