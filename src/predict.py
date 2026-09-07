"""
predict.py — load the @champion model from the registry and predict.

Run:
    python -m src.predict                          # one built-in example house
    python -m src.predict --file data/live_drift.csv --limit 5

Notice what is NOT here: a version number. We ask for the alias
@champion and let the registry decide which version that currently means.
That indirection is the whole point of aliases — a rollback becomes a
label move, not a code change and a redeploy.
"""

import argparse

import mlflow
import pandas as pd
from mlflow import MlflowClient

from src import config
from src.features import as_model_frame

# One example house, roughly the median of the training set.
EXAMPLE_HOUSE = {
    "OverallQual": 6,
    "GrLivArea": 1500,
    "TotalBsmtSF": 950,
    "GarageCars": 2,
    "GarageArea": 480,
    "1stFlrSF": 1050,
    "FullBath": 2,
    "YearBuilt": 1998,
    "YearRemodAdd": 2003,
    "LotArea": 9600,
    "LotFrontage": 70,
    "TotRmsAbvGrd": 6,
    "Fireplaces": 1,
    "OverallCond": 5,
    "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr",
    "MSZoning": "RL",
    "HouseStyle": "1Story",
    "ExterQual": "TA",
    "KitchenQual": "Gd",
    "BsmtQual": "Gd",
    "GarageFinish": "RFn",
    "CentralAir": "Y",
    "SaleCondition": "Normal",
}


def load_champion():
    """Return (model, version_number) for whatever @champion points at."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = MlflowClient()

    version = client.get_model_version_by_alias(
        config.MODEL_NAME, config.PRODUCTION_ALIAS)
    uri = "models:/" + config.MODEL_NAME + "@" + config.PRODUCTION_ALIAS
    model = mlflow.pyfunc.load_model(uri)
    return model, version.version


def main():
    parser = argparse.ArgumentParser(description="Predict house prices")
    parser.add_argument("--file", default=None,
                        help="CSV to score; omit to use the built-in example")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    model, version = load_champion()
    print("Loaded model : @" + config.PRODUCTION_ALIAS
          + " -> version " + str(version))

    if args.file is None:
        frame = as_model_frame(EXAMPLE_HOUSE)
        price = model.predict(frame)[0]

        print("\nHouse described:")
        for key in EXAMPLE_HOUSE:
            print("   " + key.ljust(16) + " = " + str(EXAMPLE_HOUSE[key]))
        print("\nPredicted price : $ {:,.0f}".format(price))
        return

    table = pd.read_csv(args.file).head(args.limit)
    frame = as_model_frame(table)
    prices = model.predict(frame)

    print("\nScoring " + str(len(table)) + " rows from " + args.file)
    print("\n  GrLivArea  OverallQual  Neighborhood     predicted")
    print("  " + "-" * 55)
    for position in range(len(table)):
        row = table.iloc[position]
        print("  {:>9.0f}  {:>11}  {:<15}  $ {:>11,.0f}".format(
            row["GrLivArea"], row["OverallQual"],
            str(row["Neighborhood"])[:15], prices[position]))


if __name__ == "__main__":
    main()
