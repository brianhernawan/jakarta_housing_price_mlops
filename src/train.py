"""
train.py — stations 4, 5, 6: fit -> measure -> quality gate -> register.

Rubric item 2 (MLflow Integration) lives here in full: parameters, metrics,
tags, and the model artifact are all logged to MLflow.

Run all three, then open the MLflow UI and compare them:

    python -m src.train --model linear                    --run-name linear
    python -m src.train --model rf --n-estimators 50      --run-name rf-50
    python -m src.train --model rf --n-estimators 200     --run-name rf-200

Train on the drifted data instead (Part 4):

    python -m src.train --data data/live_labeled.csv --run-name rf-drifted
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import as_model_frame


# ------------------------------------------------------------ provenance
def get_git_sha():
    """Which version of the code produced these numbers?"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "not-a-git-repo"


def get_data_hash(path):
    """Which data? Two files with different contents get different hashes."""
    handle = open(path, "rb")
    content = handle.read()
    handle.close()
    return hashlib.md5(content).hexdigest()[:12]


# ------------------------------------------------------------ the pipeline
def build_preprocessor():
    """Imputation, scaling and encoding — all inside the model object.

    This is the single most important design decision in the whole repo.

    Every transformation that learns something from the data (the median
    used to fill LotFrontage, the mean and standard deviation used to
    scale, the list of categories seen for Neighborhood) is fitted on the
    training set and stored inside the pipeline. At serving time we only
    ever call .predict(). Nothing is recomputed.

    Do it any other way — impute in a notebook, scale in a script, encode
    by hand in the API — and training and serving will silently disagree.
    That is training-serving skew, and it does not raise an exception.
    It just returns wrong numbers with an HTTP 200.
    """
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]

    categorical_steps = [
        # NaN here usually means something real: BsmtQual is missing exactly
        # when the house has no basement. "most_frequent" is a defensible
        # default; filling with the literal string "None" would be defensible
        # too. What is NOT defensible is dropping the rows.
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" is not optional. Without it, the first
        # request containing a Neighborhood the model never saw would crash
        # the API in production.
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), config.NUMERIC_FEATURES),
        ("categorical", Pipeline(categorical_steps), config.CATEGORICAL_FEATURES),
    ])


def build_model(model_name, n_estimators, max_depth):
    """Preprocessor plus estimator, wrapped into one Pipeline object."""
    if model_name == "linear":
        estimator = LinearRegression()
    elif model_name == "rf":
        depth = max_depth
        if depth <= 0:
            depth = None
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=depth,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise SystemExit("--model must be either 'linear' or 'rf'")

    if config.USE_LOG_TARGET:
        # SalePrice is right-skewed. We fit on log1p(price) and invert with
        # expm1 on the way out. Because the inversion is part of the
        # estimator, .predict() still returns plain dollars — serving cannot
        # forget to undo the transform.
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    steps = [
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ]
    return Pipeline(steps)


# ------------------------------------------------------------ registry ops
def move_champion(client, target_version):
    """Point the @champion alias at a specific version.

    Before moving it, the outgoing champion is tagged @previous-champion.
    MLflow does not keep alias history, so we keep one step of it ourselves.
    That single step is everything rollback needs.
    """
    outgoing = None
    try:
        outgoing = client.get_model_version_by_alias(
            config.MODEL_NAME, config.PRODUCTION_ALIAS)
    except Exception:
        outgoing = None

    if outgoing is not None and int(outgoing.version) != int(target_version):
        client.set_registered_model_alias(
            config.MODEL_NAME, config.PREVIOUS_ALIAS, str(outgoing.version))
        print("      @" + config.PREVIOUS_ALIAS + " -> v" + str(outgoing.version))

    client.set_registered_model_alias(
        config.MODEL_NAME, config.PRODUCTION_ALIAS, str(target_version))
    print("      @" + config.PRODUCTION_ALIAS + " -> v" + str(target_version))

    export_champion(client)


def export_champion(client):
    """Write the champion model out as an ordinary file.

    Why: the API runs inside a Docker container, and that container cannot
    see mlflow.db sitting on the host laptop. In a real deployment MLflow
    would run as its own service and the API would connect to it over the
    network. For a laptop demo, a file is honest and simple.
    """
    version = client.get_model_version_by_alias(
        config.MODEL_NAME, config.PRODUCTION_ALIAS)
    model = mlflow.sklearn.load_model(
        "models:/" + config.MODEL_NAME + "@" + config.PRODUCTION_ALIAS)

    if os.path.isdir(config.MODEL_EXPORT_FOLDER):
        shutil.rmtree(config.MODEL_EXPORT_FOLDER)
    os.makedirs(config.MODEL_EXPORT_FOLDER, exist_ok=True)

    joblib.dump(model, os.path.join(config.MODEL_EXPORT_FOLDER, "model.joblib"))

    handle = open(os.path.join(config.MODEL_EXPORT_FOLDER, "VERSION"), "w")
    handle.write(str(version.version))
    handle.close()

    print("  [export] champion v" + str(version.version)
          + " -> " + config.MODEL_EXPORT_FOLDER)


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Train the house price model")
    parser.add_argument("--model", default="rf", help="linear or rf")
    parser.add_argument("--n-estimators", type=int, default=config.N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=config.MAX_DEPTH,
                        help="0 means no limit")
    parser.add_argument("--data", default=config.TRAIN_FILE)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--promote", action="store_true",
                        help="make this model @champion even if it is worse")
    args = parser.parse_args()

    print("=" * 70)
    print("STATIONS 4-6 : fit -> measure -> quality gate -> register")
    print("=" * 70)

    # ---------- 1. Data ----------
    train = pd.read_csv(args.data)
    validation = pd.read_csv(config.VALIDATION_FILE)
    print("\n[1] Data")
    print("  train      : " + str(len(train)) + " rows from " + args.data)
    print("  validation : " + str(len(validation)) + " rows from " + config.VALIDATION_FILE)
    print("  features   : " + str(len(config.NUMERIC_FEATURES)) + " numeric + "
          + str(len(config.CATEGORICAL_FEATURES)) + " categorical")

    # as_model_frame is the single shared entry point. Training, the API
    # and the monitor all build their input through it.
    X_train = as_model_frame(train)
    y_train = train[config.TARGET]
    X_validation = as_model_frame(validation)
    y_validation = validation[config.TARGET]

    # ---------- 2. MLflow ----------
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = MlflowClient()

    run_name = args.run_name
    if run_name is None:
        run_name = args.model + "-" + str(args.n_estimators)

    with mlflow.start_run(run_name=run_name) as run:
        print("\n[2] MLflow run : " + run_name + " (" + run.info.run_id[:8] + ")")

        # ---------- 3. Fit ----------
        pipeline = build_model(args.model, args.n_estimators, args.max_depth)
        pipeline.fit(X_train, y_train)
        print("[3] Model fitted.")

        encoded_width = pipeline.named_steps["preprocess"].transform(
            X_train.head(5)).shape[1]
        print("    columns after encoding : " + str(encoded_width))

        # ---------- 4. Measure ----------
        predictions = pipeline.predict(X_validation)
        mae = mean_absolute_error(y_validation, predictions)
        rmse = mean_squared_error(y_validation, predictions) ** 0.5
        r2 = r2_score(y_validation, predictions)

        print("[4] Validation results:")
        print("      MAE  : $ {:,.0f}".format(mae))
        print("      RMSE : $ {:,.0f}".format(rmse))
        print("      R2   : {:.4f}".format(r2))

        # ---------- 5. Log everything ----------
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", config.RANDOM_STATE)
        mlflow.log_param("log_target", config.USE_LOG_TARGET)
        mlflow.log_param("n_numeric_features", len(config.NUMERIC_FEATURES))
        mlflow.log_param("n_categorical_features", len(config.CATEGORICAL_FEATURES))
        mlflow.log_param("train_file", args.data)
        mlflow.log_param("train_rows", len(train))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("encoded_features", encoded_width)

        mlflow.set_tag("git_sha", get_git_sha())
        mlflow.set_tag("data_md5", get_data_hash(args.data))

        # The feature list is an artifact in its own right. If it ever
        # changes, we want to know which runs used which version of it.
        feature_spec = {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
            "target": config.TARGET,
        }
        os.makedirs("models", exist_ok=True)
        spec_path = "models/feature_spec.json"
        handle = open(spec_path, "w")
        json.dump(feature_spec, handle, indent=2)
        handle.close()
        mlflow.log_artifact(spec_path)

        print("[5] Parameters, metrics, tags and the feature spec are logged.")

        # ---------- 6. QUALITY GATE ----------
        print("\n[6] Quality gate: MAE must be <= $ {:,.0f}".format(config.MAE_THRESHOLD))
        if mae > config.MAE_THRESHOLD:
            mlflow.set_tag("passed_gate", "no")
            print("      REJECTED. The model is not registered.")
            raise SystemExit(
                "GATE FAILED: MAE $ {:,.0f} exceeds the threshold $ {:,.0f}".format(
                    mae, config.MAE_THRESHOLD))
        mlflow.set_tag("passed_gate", "yes")
        print("      PASSED. The model may enter the registry.")

        # ---------- 7. Register ----------
        # The example is taken from rows that actually contain missing
        # values, so the inferred signature reflects reality rather than a
        # tidy sample that happens to have none.
        example = X_train.head(5)

        # serialization_format matters. MLflow's newer default (skops)
        # refuses to round-trip a ColumnTransformer because it contains
        # numpy.dtype objects it does not consider trusted. cloudpickle
        # handles the full sklearn Pipeline without complaint.
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=config.MODEL_NAME,
            input_example=example,
            serialization_format="cloudpickle",
        )
        print("\n[7] Registered in the model registry as: " + config.MODEL_NAME)

    # ---------- 8. Aliases ----------
    all_versions = client.search_model_versions(
        "name='" + config.MODEL_NAME + "'")
    new_version = 0
    for version in all_versions:
        if int(version.version) > new_version:
            new_version = int(version.version)
    print("      new version : v" + str(new_version))

    current_champion = None
    try:
        current_champion = client.get_model_version_by_alias(
            config.MODEL_NAME, config.PRODUCTION_ALIAS)
    except Exception:
        current_champion = None

    if current_champion is None:
        print("\n[8] No champion yet — the first model is promoted automatically.")
        move_champion(client, new_version)
    elif args.promote:
        print("\n[8] --promote given -> champion moves to v" + str(new_version))
        move_champion(client, new_version)
    else:
        print("\n[8] Champion unchanged (still v"
              + str(current_champion.version) + ").")
        print("      This model becomes the @" + config.CANDIDATE_ALIAS + " candidate.")
        client.set_registered_model_alias(
            config.MODEL_NAME, config.CANDIDATE_ALIAS, str(new_version))

    if config.MLFLOW_TRACKING_URI.startswith("http"):
        print("\nDone. Open the UI:  " + config.MLFLOW_TRACKING_URI)
    else:
        print("\nDone. Open the UI:  make mlflow-ui")


if __name__ == "__main__":
    main()
