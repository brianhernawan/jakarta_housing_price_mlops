"""
monitor.py — station 8: is the model still sane?

Rubric item 3 (Logging and Monitoring): this is the monitoring half.
It reads the structured log written by src/logger.py.

Run:  python -m src.monitor

Two independent checks, and the order matters:

  1. DRIFT CHECK   Compare the distribution of incoming features against the
                   training data. Needs NO ground truth, so it can run today.

  2. ACCURACY CHECK Compare MAE once the real sale prices are known. In the
                   real world those arrive months after the transaction
                   closes.

The lesson is that the DATA layer warns you long before the MODEL layer can,
because the model layer is waiting on labels that do not exist yet.
"""

import json
import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error

from src import config
from src.features import as_model_frame


def read_live_input():
    """Prefer the real request log; fall back to the sample file."""
    if os.path.exists(config.LOG_FILE):
        records = pd.read_json(config.LOG_FILE, lines=True)

        if "event" in records.columns:
            records = records[records["event"] == "prediction"]

        has_all_columns = True
        for column in config.FEATURES:
            if column not in records.columns:
                has_all_columns = False

        if has_all_columns and len(records) >= 30:
            # Only the last window. The log holds old and new traffic mixed
            # together; averaging over all of it dilutes the signal and drift
            # looks milder than it is.
            total = len(records)
            records = records.tail(config.MONITOR_WINDOW)
            print("  live input source : " + config.LOG_FILE
                  + " (last " + str(len(records)) + " of " + str(total)
                  + " requests)")
            return records[config.FEATURES]

    print("  live input source : " + config.LIVE_DRIFT_FILE
          + "  (log not usable yet, using the sample file)")
    return pd.read_csv(config.LIVE_DRIFT_FILE)[config.FEATURES]


def check_drift(train, live):
    """Compare two distributions, one numeric column at a time.

    Kolmogorov-Smirnov asks: could these two samples plausibly have come
    from the same distribution? A small p-value means no.

    Only numeric columns are tested here. Categorical drift needs a
    different test (chi-squared or population stability index) and is
    deliberately left out to keep this readable.
    """
    print("\n[1] DRIFT CHECK — does incoming data still look like training data?")
    print("\n  column            train mean      live mean      shift     p-value   status")
    print("  " + "-" * 78)

    drift_found = False
    for column in config.NUMERIC_FEATURES:
        train_values = train[column].dropna()
        live_values = live[column].dropna()

        if len(live_values) < 5:
            continue

        train_mean = train_values.mean()
        live_mean = live_values.mean()

        if train_mean == 0:
            shift = 0.0
        else:
            shift = (live_mean - train_mean) / train_mean * 100

        result = ks_2samp(train_values, live_values)
        if result.pvalue < config.PVALUE_THRESHOLD:
            status = "DRIFT"
            drift_found = True
        else:
            status = "ok"

        print("  {:<16s} {:>12.1f} {:>14.1f}  {:>+8.1f}%  {:>10.2g}   {}".format(
            column, train_mean, live_mean, shift, result.pvalue, status))

    if drift_found:
        print("\n  Some columns have shifted distribution.")
        print("  The model is not broken. It is being asked about houses")
        print("  unlike the ones it studied.")
    else:
        print("\n  All columns still resemble the training data.")

    return drift_found


def get_champion():
    """Return (model, version, training_mae) for the current champion."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(
        config.MODEL_NAME, config.PRODUCTION_ALIAS)
    run = client.get_run(version.run_id)
    model = mlflow.pyfunc.load_model(
        "models:/" + config.MODEL_NAME + "@" + config.PRODUCTION_ALIAS)
    return model, version.version, run.data.metrics.get("mae", 0.0)


def check_accuracy(model, version, training_mae):
    """Recompute MAE once the true sale prices are known."""
    print("\n[2] ACCURACY CHECK — once the real sale prices are known")

    if not os.path.exists(config.LIVE_LABELED_FILE):
        print("  (no labelled file, skipped)")
        return

    live = pd.read_csv(config.LIVE_LABELED_FILE)
    predictions = model.predict(as_model_frame(live))
    live_mae = mean_absolute_error(live[config.TARGET], predictions)

    print("\n  champion model      : v" + str(version))
    print("  MAE at training     : $ {:>12,.0f}".format(training_mae))
    print("  MAE on live data    : $ {:>12,.0f}".format(live_mae))

    if training_mae > 0:
        ratio = live_mae / training_mae
        print("  degradation         : {:.1f}x".format(ratio))
        if ratio > 1.5:
            print("\n  This model can no longer be trusted on current traffic.")
            print("  Note carefully: not one error or exception was raised.")
            print("  The API answered HTTP 200 every single time — with the")
            print("  wrong number. That is what silent failure looks like.")


def main():
    print("=" * 78)
    print("STATION 8 : MONITORING — is the model still sane?")
    print("=" * 78)

    train = pd.read_csv(config.TRAIN_FILE)
    live = read_live_input()
    print("  rows — train: " + str(len(train)) + " | live: " + str(len(live)))

    if os.path.exists(config.TRAIN_PROFILE_FILE):
        handle = open(config.TRAIN_PROFILE_FILE)
        profile = json.load(handle)
        handle.close()
        print("  watch column      : " + profile["drift_watch_column"]
              + " (training mean " + str(round(profile["drift_watch_mean"])) + ")")

    check_drift(train, live)

    model, version, training_mae = get_champion()
    check_accuracy(model, version, training_mae)

    print("\n" + "=" * 78)
    print("Takeaway: the DATA layer (drift) raises the alarm before the MODEL")
    print("layer (MAE) can, because MAE needs ground truth, and ground truth")
    print("only exists after the sale has actually closed.")


if __name__ == "__main__":
    main()
