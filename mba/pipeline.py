"""Processing pipeline for mba."""

import warnings
from typing import Tuple, Optional

import pandas as pd
import pdpipe as pdp
from pdpipe.util import out_of_place_col_insert

from .sentiment import (
    SentimentPredictor,
    get_sentiment_predictor,
)
from .shared import (
    Column,
    ContextKey
)


class AddSentimentColumns(pdp.PdPipelineStage):
    """Add sentiment columns to input dataframes.

    This stage - on transform - checks the application context for a key
    'reviews_fpath' mapping to a string containig the fully qualified
    path to a csv file containing review data by clients contained in the input
    dataset, of the schema "ID, continue, name, ..., look, onto" - overall
    2001 columns including the ID column; thus, it assumes the intersection
    between the values of the ID columns of the input dataframe and the review
    dataframe will be non-zero.

    The input dataframe is assumed to be indexed by the ID column.

    If intersection is zero, or if no such file is found in the application
    context, the stage will issue a warning, and add the `sentiment_0` and
    `sentiment_1` columns to the input dataframe will all zeroes.

    If the review dataframe is found, these columns are added, with non-zero
    values for users which issued a review, with the appropriate sentiment
    (`sentiment_0` of 1 and `sentiment_1` of 0 represent a negative sentiment
    review, while the opposite represents a positive sentiment review; 0 on
    both columns means the corresponding user never issued a review, while a
    value of 1 on both is erroneous, and should never be encountered).



    Parameters
    ----------
    sentiment_predictor: SentimentPredictor
        The sentiment_predictor to use.
    """

    def __init__(
        self,
        sentiment_predictor: SentimentPredictor,
        **kwargs,
    ) -> None:
        self.sentiment_predictor = sentiment_predictor
        super_kwargs = {
            'exmsg': "The ID column is missing for the input dataframe!",
            'desc': "Add the sentiment columns to input dataframes",
        }
        super_kwargs.update(**kwargs)
        super().__init__(**super_kwargs)

    def _prec(self, df: pd.DataFrame) -> bool:
        return df.index.name == Column.ID

    def _transform(
            self, df: pd.DataFrame, verbose=None) -> pd.DataFrame:
        rev_fpath = self.application_context.get(
            ContextKey.REVIEWS_FPATH, None)
        if rev_fpath is None:
            warnings.warn(
                "No input review file path in pipeline application context! "
                "Filling both sentiment columns with zeros!"
            )
            res_df = out_of_place_col_insert(
                df=df,
                series=[0] * len(df),
                loc=len(df),
                column_name=Column.SENTIMENT_0,
            )
            res_df = out_of_place_col_insert(
                df=res_df,
                series=[0] * len(res_df),
                loc=len(res_df),
                column_name=Column.SENTIMENT_1,
            )
            return res_df
        rev_df = pd.read_csv(rev_fpath)
        if verbose:
            print(f"Reviews df len: {len(rev_df)}")
            rev_ids = set(rev_df[Column.ID])
            input_ids = set(df.index)
            inter = rev_ids.intersection(input_ids)
            print(
                f"  - {len(inter)} id intersection between input & reviewes.")
        rev_df[Column.SENTIMENT] = self.sentiment_predictor.predict(rev_df)
        subdf = rev_df[[Column.ID, Column.SENTIMENT]]
        dumm = pd.get_dummies(subdf[Column.SENTIMENT])
        subdf[Column.SENTIMENT_0] = dumm[0]
        subdf[Column.SENTIMENT_1] = dumm[1]
        subdf = subdf.set_index(Column.ID)
        subdf = subdf.drop(Column.SENTIMENT, axis=1)
        res_df = df.join(subdf)
        if verbose:
            n = len(res_df) - res_df[Column.SENTIMENT_0].isna().sum()
            print(f"  - None-NA sentiment features adde to {n} rows.")
        res_df[Column.SENTIMENT_0] = res_df[Column.SENTIMENT_0].fillna(0)
        res_df[Column.SENTIMENT_1] = res_df[Column.SENTIMENT_1].fillna(0)
        return res_df


class _StatusColBuilder():
    """A callable that returns an ordinal join of the status features, with
    their ordinality determined by the given weights, as a pandas Series.

    Parameters
    ----------
    status_weights : 3-tuple of floats
        The weights corresponding to silver, gold and platinum status, in this
        order
    """

    def __init__(self, status_weights: Tuple[float, float, float]):
        self.status_weights = status_weights

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return df[Column.STATUS_SILVER] * self.status_weights[0] \
            + df[Column.STATUS_GOLD] * self.status_weights[1] \
            + df[Column.STATUS_PANTINUM] * self.status_weights[2]


def build_pipeline(
    status_weights: Optional[Tuple[float]] = (1, 2, 3),
):
    """Build a preprocessing pipeline for sales recommendations model."""
    print("Starting to build the preprocessing pipeline...")
    print("Building the sentiment predictor...")
    sent_pred = get_sentiment_predictor()
    print("Done.")
    print("Building pipeline stages...")
    stages = [
        pdp.df.set_index(keys=Column.ID),
        AddSentimentColumns(sent_pred),
        # pdp.ColByFrameFunc(
        #     column=Column.STATUS_ORD,
        #     func=_StatusColBuilder(status_weights),
        #     follow_column=Column.STATUS_SILVER,
        #     func_desc="weight-summing the status columns",
        # ),
        # pdp.ColDrop([
        #     Column.STATUS_SILVER,
        #     Column.STATUS_GOLD,
        #     Column.STATUS_PANTINUM,
        # ])
    ]
    print("Done. Returning pipeline.")
    return pdp.PdPipeline(stages)
