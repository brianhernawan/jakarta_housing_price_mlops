"""
data.py — stations 1, 2, 3 on the conveyor belt: load -> validate -> split.

Run:  python -m src.data

What this module has to solve
-----------------------------
The raw Kaggle files do not line up with what an MLOps demo needs:

  * data/raw/test.csv has NO SalePrice column. It is the Kaggle leaderboard
    holdout. You cannot measure anything with it. So we do not try — we give
    it its real job instead: unlabelled traffic arriving at the API.

  * All evaluation sets are therefore carved out of data/raw/train.csv.

  * Real Ames data has no drift. train.csv and test.csv are drawn from the
    same pool. So the "three months later" scenario has to be constructed,
    and this module constructs it honestly: it RESAMPLES real rows with a
    weighting, rather than inventing fake houses.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


# ------------------------------------------------------------------ loading
def load_raw(path, expect_target):
    """Read one raw CSV.

    Kaggle writes the literal string "NA" for missing values. For many
    columns that string is not missing data at all — PoolQC = "NA" means
    "this house has no pool". pandas reads it as NaN either way, and the
    imputer inside the training pipeline deals with it. We do not drop rows.
    """
    table = pd.read_csv(path)
    print("  [load] " + path + " -> " + str(len(table)) + " rows, "
          + str(len(table.columns)) + " columns")

    if expect_target and config.TARGET not in table.columns:
        raise SystemExit("VALIDATION FAILED: " + path + " has no '"
                         + config.TARGET + "' column.")
    return table


# --------------------------------------------------------------- validation
def validate(table, name, expect_target):
    """Station 2: the checkpoint.

    If the data is broken it is better to stop here than to train a model on
    garbage and spend the afternoon wondering why the numbers look strange.

    Note the policy: structural problems are FATAL, missing values are only
    REPORTED. Missing values are expected in this dataset and are handled by
    the imputer inside the pipeline. Failing on them, as a naive checkpoint
    would, makes the pipeline unusable on real data.
    """
    print("\n  [check] " + name)

    if len(table) == 0:
        raise SystemExit("VALIDATION FAILED: " + name + " is empty.")

    required = list(config.FEATURES)
    if expect_target:
        required.append(config.TARGET)

    for column in required:
        if column not in table.columns:
            raise SystemExit("VALIDATION FAILED: column '" + column
                             + "' is missing from " + name + ".")

    print("    rows                : " + str(len(table)))
    print("    required columns    : all " + str(len(required)) + " present")

    # Missing values: reported, not fatal.
    missing_total = 0
    missing_columns = []
    for column in required:
        count = int(table[column].isna().sum())
        if count > 0:
            missing_total = missing_total + count
            missing_columns.append(column + "=" + str(count))
    print("    missing values      : " + str(missing_total)
          + " (handled by the imputer)")
    if len(missing_columns) > 0:
        print("      " + ", ".join(missing_columns))

    # Structural sanity: these WOULD be fatal.
    bad_area = int((table["GrLivArea"] <= 0).sum())
    print("    GrLivArea <= 0      : " + str(bad_area))
    if bad_area > 0:
        raise SystemExit("VALIDATION FAILED: a house with no living area.")

    if expect_target:
        bad_price = int((table[config.TARGET] <= 0).sum())
        print("    SalePrice <= 0      : " + str(bad_price))
        if bad_price > 0:
            raise SystemExit("VALIDATION FAILED: a house sold for nothing.")

    print("    passed")


# ------------------------------------------------------------ drift weights
def upmarket_weights(table, strength=1.6):
    """Weight each row by how far 'upmarket' it sits.

    The scenario we are simulating: three months later, a new high-end
    development opens and the sales mix shifts toward larger, newer,
    better-finished houses.

    Why upmarket and not downmarket? Absolute error scales with price. A
    model asked mostly about expensive houses produces a visibly worse MAE,
    and Random Forests under-predict at the top of the range because they
    cannot extrapolate past the largest value they saw in training. That is
    the failure we want the monitoring section to catch.

    We resample REAL rows. No synthetic houses are invented.
    """
    score = np.zeros(len(table))
    for column in ["GrLivArea", "OverallQual", "YearBuilt"]:
        values = table[column].astype(float)
        values = values.fillna(values.median())
        spread = values.std()
        if spread == 0:
            continue
        score = score + (values - values.mean()) / spread

    score = score / 3.0
    weights = np.exp(strength * score)
    return weights / weights.sum()


def make_live_sample(source, rows, seed, name):
    """Draw `rows` rows from `source`, biased upmarket."""
    generator = np.random.default_rng(seed)
    weights = upmarket_weights(source)
    picked = generator.choice(len(source), size=rows, replace=True, p=weights)
    sample = source.iloc[picked].reset_index(drop=True)

    before = source["GrLivArea"].mean()
    after = sample["GrLivArea"].mean()
    print("    " + name + ": " + str(rows) + " rows | mean GrLivArea "
          + str(round(before)) + " -> " + str(round(after)) + " sq ft")
    return sample


# ------------------------------------------------------------------ profile
def write_train_profile(train):
    """Save the training-set statistics the API needs at runtime.

    The API compares incoming requests against these numbers to compute a
    live drift score. Hard-coding them in api/main.py would guarantee they
    go stale the moment the data changes.
    """
    profile = {
        "n_rows": int(len(train)),
        "drift_watch_column": config.DRIFT_WATCH_COLUMN,
        "drift_watch_mean": float(train[config.DRIFT_WATCH_COLUMN].mean()),
        "drift_watch_std": float(train[config.DRIFT_WATCH_COLUMN].std()),
        "target_mean": float(train[config.TARGET].mean()),
        "target_median": float(train[config.TARGET].median()),
    }
    os.makedirs(os.path.dirname(config.TRAIN_PROFILE_FILE), exist_ok=True)
    handle = open(config.TRAIN_PROFILE_FILE, "w")
    json.dump(profile, handle, indent=2)
    handle.close()
    print("  [save] " + config.TRAIN_PROFILE_FILE)
    return profile


# --------------------------------------------------------------------- main
def main():
    print("=" * 70)
    print("STATIONS 1-3 : load -> validate -> split")
    print("=" * 70)

    print("\n[1] Loading the raw Kaggle files")
    raw_train = load_raw(config.RAW_TRAIN_FILE, expect_target=True)
    raw_test = load_raw(config.RAW_TEST_FILE, expect_target=False)
    print("  note: raw test.csv has no " + config.TARGET
          + " column — it is the Kaggle holdout, so it becomes our")
    print("        unlabelled production traffic, not an evaluation set.")

    print("\n[2] Validating")
    validate(raw_train, "raw train.csv", expect_target=True)
    validate(raw_test, "raw test.csv", expect_target=False)

    print("\n[3] Splitting")
    # First carve off the delayed-label pool. These rows are never trained
    # on and never validated on, so when their "labels arrive" later they
    # are genuinely unseen.
    keep, delayed = train_test_split(
        raw_train,
        test_size=config.DELAYED_LABEL_SIZE,
        random_state=config.RANDOM_STATE,
    )
    train, validation = train_test_split(
        keep,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
    )

    if len(train) + len(validation) + len(delayed) != len(raw_train):
        raise SystemExit("rows went missing during the split.")

    os.makedirs(config.DATA_FOLDER, exist_ok=True)
    train.to_csv(config.TRAIN_FILE, index=False)
    validation.to_csv(config.VALIDATION_FILE, index=False)
    print("  [save] train      : " + str(len(train)) + " rows -> " + config.TRAIN_FILE)
    print("  [save] validation : " + str(len(validation)) + " rows -> " + config.VALIDATION_FILE)
    print("  [save] delayed    : " + str(len(delayed)) + " rows (held back, not written yet)")

    print("\n[4] Training profile")
    profile = write_train_profile(train)

    print("\n[5] Building the 'three months later' production traffic")
    print("    scenario: a new high-end development opens; the sales mix")
    print("    shifts toward larger, newer, better-finished houses.")

    # (a) unlabelled — what the API actually receives today
    live_drift = make_live_sample(
        raw_test, config.LIVE_ROWS, config.RANDOM_STATE + 1, "live_drift  ")
    live_drift.to_csv(config.LIVE_DRIFT_FILE, index=False)

    # (b) labelled — the same kind of traffic, but with the sale prices that
    #     in the real world only turn up months after the transaction closes
    live_labeled = make_live_sample(
        delayed, config.LIVE_ROWS, config.RANDOM_STATE + 2, "live_labeled")
    live_labeled.to_csv(config.LIVE_LABELED_FILE, index=False)

    print("  [save] " + config.LIVE_DRIFT_FILE + "   (no SalePrice)")
    print("  [save] " + config.LIVE_LABELED_FILE + " (with SalePrice)")

    print("\n" + "-" * 70)
    print("Compare the column the dashboard watches (" + config.DRIFT_WATCH_COLUMN + "):")
    print("  training data : " + str(round(profile["drift_watch_mean"])) + " sq ft")
    print("  live data     : " + str(round(live_drift[config.DRIFT_WATCH_COLUMN].mean()))
          + " sq ft   <-- remember this number")
    print("\nDone. Next:  python -m src.train")


if __name__ == "__main__":
    main()
