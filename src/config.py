"""
config.py — every tunable number and path lives here, in one place.

Why this file exists
--------------------
So there are no "magic numbers" scattered across six different modules.
Want to change the number of trees in the Random Forest? Change it here,
one line, once.

Rubric item 1 (Modular Code): this is the configuration layer. No other
module hard-codes a path, a column name, or a threshold.
"""

import os

# --------------------------------------------------------------- Data paths
# Raw files exactly as downloaded. Never written to.
RAW_TRAIN_FILE = "data/raw/train.csv"          # 1460 rows, labelled
RAW_TEST_FILE = "data/raw/test.csv"            # 1459 rows, NO SalePrice column

# Files produced by `python -m src.data`
DATA_FOLDER = "data"
TRAIN_FILE = "data/train.csv"                  # model training set
VALIDATION_FILE = "data/validation.csv"        # metrics are measured here
LIVE_DRIFT_FILE = "data/live_drift.csv"        # unlabelled "production" traffic
LIVE_LABELED_FILE = "data/live_labeled.csv"    # same traffic, labels arrive later

# ------------------------------------------------------------ Split sizes
# The Kaggle test.csv has no SalePrice, so it cannot be used to measure
# anything. We therefore carve our own evaluation sets out of train.csv:
#
#   train.csv (1460 labelled rows)
#     |
#     +-- 10%  -> delayed-label pool -> data/live_labeled.csv
#     |
#     +-- 90%
#           +-- 80% -> data/train.csv
#           +-- 20% -> data/validation.csv
#
# data/raw/test.csv keeps its real job: unlabelled traffic hitting the API.
DELAYED_LABEL_SIZE = 0.10
VALIDATION_SIZE = 0.20

# How many rows to synthesise for each "production" file.
LIVE_ROWS = 400

# ------------------------------------------------------------ Feature set
# 15 numeric + 9 categorical = 24 columns.
#
# Not all 79 columns: a Pydantic schema with 79 fields makes the FastAPI
# /docs page unusable to demo, and these 24 already reach R2 ~= 0.89.
#
# The ORDER of this list matters and must be identical at training time and
# at serving time. That is the classic training-serving skew bug.
NUMERIC_FEATURES = [
    "OverallQual",      # overall material and finish quality, 1-10
    "GrLivArea",        # above-ground living area, sq ft  <- drift watch column
    "TotalBsmtSF",      # total basement area, sq ft
    "GarageCars",       # garage capacity in cars
    "GarageArea",       # garage size, sq ft
    "1stFlrSF",         # first floor area, sq ft
    "FullBath",         # full bathrooms above ground
    "YearBuilt",        # original construction year
    "YearRemodAdd",     # remodel year (equals YearBuilt if never remodelled)
    "LotArea",          # lot size, sq ft
    "LotFrontage",      # street frontage, ft — 259 missing values on purpose
    "TotRmsAbvGrd",     # total rooms above ground, excluding bathrooms
    "Fireplaces",       # number of fireplaces
    "OverallCond",      # overall condition, 1-10
    "BedroomAbvGr",     # bedrooms above ground
]

CATEGORICAL_FEATURES = [
    "Neighborhood",     # 25 distinct values
    "MSZoning",         # general zoning classification
    "HouseStyle",       # one storey, two storey, split level, ...
    "ExterQual",        # exterior material quality
    "KitchenQual",      # kitchen quality
    "BsmtQual",         # basement height — NA genuinely means "no basement"
    "GarageFinish",     # NA genuinely means "no garage"
    "CentralAir",       # Y / N
    "SaleCondition",    # Normal, Abnorml, Partial, ...
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "SalePrice"

# The single column the live dashboard watches for drift. Continuous,
# intuitive, and strongly correlated with price (r = 0.71).
DRIFT_WATCH_COLUMN = "GrLivArea"

# ---------------------------------------------------- Reproducibility
# Without a fixed seed, every person running this gets different numbers
# and we end up blaming the wrong thing.
RANDOM_STATE = 42

# ---------------------------------------------------------------- Model
N_ESTIMATORS = 200
MAX_DEPTH = 0            # 0 means "no limit" — let the trees grow fully

# SalePrice is right-skewed (median $163k, max $755k). Training on
# log1p(price) and inverting on the way out costs nothing and buys about
# $600 of MAE. The inversion lives INSIDE the pipeline, so .predict()
# always returns plain dollars and serving cannot forget to undo it.
USE_LOG_TARGET = True

# ------------------------------------------------------- Quality gate
# If validation MAE is worse than this, the pipeline STOPS and the model
# never reaches the registry.
#
# Measured baselines on this feature set (seed 42):
#     LinearRegression       MAE ~ $19,300
#     RandomForest 200       MAE ~ $17,800
#     RandomForest 200 + log MAE ~ $17,200
#
# $25,000 sits above every healthy model but below anything broken.
# TRY IT YOURSELF: lower this to 18000 and run `--model linear`.
# The gate will refuse the model — that is the gate working.
#
# The lesson in Part 4 is that a threshold ALONE is not enough. A model
# that clears the threshold but is still worse than the current champion
# must not be promoted. That is a comparison job, not a threshold job.
MAE_THRESHOLD = 25_000.0

# --------------------------------------------------------------- MLflow
# The Model Registry REQUIRES a database backend. A plain file store
# will not work.
#
# This is read from the environment because the SAME tracking server is
# reached by two different names:
#
#   from your laptop      MLFLOW_TRACKING_URI=http://localhost:5000
#   from inside a container  MLFLOW_TRACKING_URI=http://mlflow:5000
#
# Containers resolve each other by Compose service name on the shared
# network; your browser and your shell are outside that network and must
# use localhost. Same server, two names.
#
# The default keeps the old behaviour, so nothing breaks if the tracking
# server is not running: a local SQLite file in the project root.
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "jakarta-housing-price-prediction"
MODEL_NAME = "jakarta-housing-price"
PRODUCTION_ALIAS = "champion"
CANDIDATE_ALIAS = "challenger"
# MLflow does not keep alias history, so we record one step backwards
# ourselves. This is what src/rollback.py reads.
PREVIOUS_ALIAS = "previous-champion"

# --------------------------------------------------------------- Serving
# A plain-file copy of the champion model. Used when the API runs inside a
# container, which cannot see mlflow.db on the host laptop.
MODEL_EXPORT_FOLDER = "models/champion"
LOG_FILE = "logs/predictions.log"

# ------------------------------------------------------------ Monitoring
# A Kolmogorov-Smirnov p-value below this means "these two distributions
# are not the same".
PVALUE_THRESHOLD = 0.05

# Monitoring compares the LAST WINDOW of requests, not the whole log file.
# The log holds both old and new traffic; mixing them dilutes the signal
# and drift can slip past unnoticed. Every production dashboard has a time
# range ("last 1h", "last 24h"), never "since the beginning of time".
MONITOR_WINDOW = 200

# Mean GrLivArea in the training data. Filled in by src/data.py so the API
# never has to guess. See data/train_profile.json.
TRAIN_PROFILE_FILE = "models/train_profile.json"
