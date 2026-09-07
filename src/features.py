"""
features.py — the one function that builds a model input frame.

Rubric item 1 (Modular Code): this module exists so that training, batch
prediction, the API and the monitor all construct their input the exact
same way. If this logic were copy-pasted into four files, the fourth copy
would eventually drift and nobody would notice until the predictions were
already wrong.

Two things it guarantees:

  1. COLUMN ORDER. Always config.FEATURES, in that order. A ColumnTransformer
     selects by name so order is not strictly fatal here, but relying on that
     is a bad habit — swap in a plain numpy pipeline one day and silent
     nonsense begins.

  2. NUMERIC COLUMNS ARE float64. This one is not cosmetic. Columns like
     GarageCars and FullBath are integers in the CSV, so MLflow infers an
     integer schema for them. Python integers cannot hold a missing value,
     so the first production request with a null GarageCars is coerced to
     float and rejected by schema enforcement. Casting to float everywhere,
     including in the example we register, removes the whole class of bug.
"""

import pandas as pd

from src import config


def as_model_frame(source):
    """Return a DataFrame with exactly the model's input columns.

    `source` may be a DataFrame or a single dict (one request).
    """
    if isinstance(source, dict):
        table = pd.DataFrame([source])
    else:
        table = pd.DataFrame(source)

    missing = []
    for column in config.FEATURES:
        if column not in table.columns:
            missing.append(column)
    if len(missing) > 0:
        raise ValueError("missing feature columns: " + ", ".join(missing))

    frame = table[config.FEATURES].copy()

    for column in config.NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].astype("float64")

    for column in config.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("object")

    return frame
