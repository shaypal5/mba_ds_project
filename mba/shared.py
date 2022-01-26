"""Shared functions for mba."""

import os
from os.path import dirname, abspath


REPO_DPATH = dirname(dirname(abspath(__file__)))
DATA_DPATH = os.path.join(REPO_DPATH, 'data')
MODELS_DPATH = os.path.join(REPO_DPATH, 'models')
