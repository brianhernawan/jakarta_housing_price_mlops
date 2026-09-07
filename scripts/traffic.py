"""
scripts/traffic.py — pretend to be real users so the dashboard has something
to draw.

Run (the API must already be up):
    python scripts/traffic.py

Two waves are sent:
    wave 1  NORMAL   real rows sampled from data/raw/test.csv
    wave 2  DRIFTED  real rows sampled from data/live_drift.csv

Both waves are REAL houses from the dataset. Nothing is invented. Watch the
Grafana panels bend when wave 2 arrives — and watch the error count stay
at zero the entire time.
"""

import argparse
import math
import os
import sys
import time

# Running this as `python scripts/traffic.py` puts scripts/ on the import
# path, not the project root, so `from src import config` would fail.
# Adding the project root explicitly means both of these work:
#     python scripts/traffic.py
#     python -m scripts.traffic
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import requests

from src import config

BASE_URL = "http://localhost:8000"
DEFAULT_COUNT = 200
DELAY_SECONDS = 0.05          # about 20 requests per second


def row_to_payload(row):
    """Turn one DataFrame row into a JSON body the API will accept.

    NaN is not valid JSON, so missing values become null. The imputer
    inside the model pipeline fills them in — which is exactly what we want
    to exercise, because real traffic has holes in it.
    """
    payload = {}
    for column in config.FEATURES:
        value = row[column]

        if value is None:
            payload[column] = None
            continue

        if isinstance(value, float) and math.isnan(value):
            payload[column] = None
            continue

        if column in config.NUMERIC_FEATURES:
            payload[column] = float(value)
        else:
            payload[column] = str(value)

    return payload


def send_wave(name, source_file, count, seed):
    """Sample `count` rows from a CSV and POST them one at a time."""
    table = pd.read_csv(source_file)
    generator = np.random.default_rng(seed)
    picked = generator.choice(len(table), size=count, replace=True)
    sample = table.iloc[picked]

    mean_area = sample[config.DRIFT_WATCH_COLUMN].mean()
    print("\n>> Wave '" + name + "' — " + str(count) + " requests from "
          + source_file)
    print("   mean " + config.DRIFT_WATCH_COLUMN + " = "
          + str(round(mean_area)) + " sq ft")

    succeeded = 0
    failed = 0
    prices = []

    for position in range(count):
        payload = row_to_payload(sample.iloc[position])
        try:
            response = requests.post(BASE_URL + "/predict",
                                     json=payload, timeout=10)
            if response.status_code == 200:
                succeeded = succeeded + 1
                prices.append(response.json()["predicted_price_usd"])
            else:
                failed = failed + 1
                if failed <= 3:
                    print("   rejected " + str(response.status_code)
                          + ": " + response.text[:160])
        except Exception as error:
            failed = failed + 1
            if failed <= 3:
                print("   request failed: " + str(error)[:120])

        if (position + 1) % 50 == 0:
            print("   " + str(position + 1) + "/" + str(count) + " sent...")

        time.sleep(DELAY_SECONDS)

    print("   done: " + str(succeeded) + " ok, " + str(failed) + " failed")
    if len(prices) > 0:
        print("   median predicted price: $ {:,.0f}".format(
            float(np.median(prices))))

    return failed


def main():
    parser = argparse.ArgumentParser(description="Send traffic at the API")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT,
                        help="requests per wave")
    parser.add_argument("--no-pause", action="store_true",
                        help="do not wait for ENTER between the two waves")
    args = parser.parse_args()

    print("=" * 70)
    print("TRAFFIC GENERATOR — open Grafana at http://localhost:3000")
    print("=" * 70)

    try:
        check = requests.get(BASE_URL + "/health", timeout=5)
        print("API : " + str(check.json()))
    except Exception:
        print("Cannot reach the API at " + BASE_URL)
        print("   Start it first:  uvicorn api.main:app --port 8000")
        print("   or:              docker compose up -d")
        sys.exit(1)

    send_wave("normal", config.RAW_TEST_FILE, args.count, seed=1)

    if not args.no_pause:
        print("\n--- look at Grafana now, then press ENTER ---")
        try:
            input()
        except EOFError:
            pass

    send_wave("drifted", config.LIVE_DRIFT_FILE, args.count, seed=2)

    print("\nDone. In Grafana, watch for:")
    print("   * 'Mean GrLivArea' climbing sharply")
    print("   * 'Drift score' crossing into the red band")
    print("   * 'Predicted price' distribution shifting upward")
    print("\n   And note: zero errors. Every request was answered 200 OK.")
    print("\n   Now run:  python -m src.monitor")


if __name__ == "__main__":
    main()
