"""Predict buyer flag for input dataset and write to output file."""

import sys
import warnings

from mba.buyer import predict_buyer_flag


if __name__ == '__main__':
    try:
        arg1 = sys.argv[1]
    except IndexError:
        warnings.warn(
            "No input file path provided. Going forward with default.")
        arg1 = None
    try:
        arg2 = sys.argv[2]
    except IndexError:
        warnings.warn(
            "No review input file path provided. Going forward with default.")
        arg2 = None
    try:
        arg3 = sys.argv[3]
    except IndexError:
        warnings.warn(
            "No output file path provided. Going forward with default.")
        arg3 = None
    predict_buyer_flag(
        input_fpath=arg1,
        input_reviews_fpath=arg2,
        output_fpath=arg3,
    )
