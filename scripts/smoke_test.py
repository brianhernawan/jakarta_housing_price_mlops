"""
scripts/smoke_test.py — prove the API actually behaves, in about ten seconds.

    python scripts/smoke_test.py

Checks every branch that matters:
    * /health responds and reports a loaded model
    * a valid house returns a sensible price
    * missing-but-legal values (null LotFrontage) are imputed, not rejected
    * a category the model never saw does NOT crash the service
    * an impossible house is rejected with 422, not silently scored
    * batch scoring works
    * /metrics exposes every metric Grafana expects
"""

import sys

import requests

BASE_URL = "http://localhost:8000"

VALID_HOUSE = {
    "OverallQual": 7, "GrLivArea": 1710, "TotalBsmtSF": 856,
    "GarageCars": 2, "GarageArea": 548, "1stFlrSF": 856,
    "FullBath": 2, "YearBuilt": 2003, "YearRemodAdd": 2003,
    "LotArea": 8450, "LotFrontage": 65, "TotRmsAbvGrd": 8,
    "Fireplaces": 0, "OverallCond": 5, "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr", "MSZoning": "RL", "HouseStyle": "2Story",
    "ExterQual": "Gd", "KitchenQual": "Gd", "BsmtQual": "Gd",
    "GarageFinish": "RFn", "CentralAir": "Y", "SaleCondition": "Normal",
}

EXPECTED_METRICS = [
    "prediction_total",
    "prediction_error_total",
    "prediction_latency_seconds",
    "predicted_price_usd",
    "model_version_active",
    "input_gr_liv_area_mean",
    "drift_score_gr_liv_area",
]

results = []


def record(name, passed, note=""):
    mark = "[ok]" if passed else "[FAIL]"
    print("  " + mark.ljust(8) + name.ljust(42) + note)
    results.append(passed)
    return passed


def test_health():
    try:
        response = requests.get(BASE_URL + "/health", timeout=10)
    except Exception as error:
        return record("health endpoint", False, str(error)[:60])

    body = response.json()
    passed = response.status_code == 200 and body["status"] == "healthy"
    record("health endpoint", passed,
           "v" + str(body.get("model_version")) + " via " + str(body.get("model_source")))
    return passed


def test_valid_prediction():
    response = requests.post(BASE_URL + "/predict", json=VALID_HOUSE, timeout=10)
    if response.status_code != 200:
        return record("valid house scored", False, "HTTP " + str(response.status_code))

    price = response.json()["predicted_price_usd"]
    # Anything outside this range means something is badly wrong — for
    # example the log transform not being inverted.
    sensible = 30_000 < price < 800_000
    return record("valid house scored", sensible, "$ {:,.0f}".format(price))


def test_missing_values_imputed():
    house = dict(VALID_HOUSE)
    house["LotFrontage"] = None
    house["BsmtQual"] = None
    house["GarageFinish"] = None

    response = requests.post(BASE_URL + "/predict", json=house, timeout=10)
    passed = response.status_code == 200
    note = ""
    if passed:
        note = "$ {:,.0f}".format(response.json()["predicted_price_usd"])
    return record("null values imputed, not rejected", passed, note)


def test_unseen_category():
    house = dict(VALID_HOUSE)
    house["Neighborhood"] = "Atlantis"

    response = requests.post(BASE_URL + "/predict", json=house, timeout=10)
    passed = response.status_code == 200
    return record("unseen category does not crash", passed,
                  "HTTP " + str(response.status_code))


def test_invalid_rejected():
    house = dict(VALID_HOUSE)
    house["GrLivArea"] = -50
    house["OverallQual"] = 99

    response = requests.post(BASE_URL + "/predict", json=house, timeout=10)
    passed = response.status_code == 422
    note = "HTTP " + str(response.status_code)
    if passed:
        note = note + ", " + str(len(response.json()["problems"])) + " problems reported"
    return record("impossible house rejected 422", passed, note)


def test_batch():
    payload = {"houses": [VALID_HOUSE, VALID_HOUSE, VALID_HOUSE]}
    response = requests.post(BASE_URL + "/predict/batch", json=payload, timeout=15)
    if response.status_code != 200:
        return record("batch scoring", False, "HTTP " + str(response.status_code))

    body = response.json()
    passed = body["count"] == 3 and len(body["predictions"]) == 3
    return record("batch scoring", passed, str(body["count"]) + " scored")


def test_metrics():
    response = requests.get(BASE_URL + "/metrics", timeout=10)
    if response.status_code != 200:
        return record("metrics endpoint", False, "HTTP " + str(response.status_code))

    body = response.text
    missing = []
    for name in EXPECTED_METRICS:
        if name not in body:
            missing.append(name)

    passed = len(missing) == 0
    note = "all " + str(len(EXPECTED_METRICS)) + " present"
    if not passed:
        note = "missing: " + ", ".join(missing)
    return record("metrics endpoint", passed, note)


def main():
    print("=" * 70)
    print("SMOKE TEST — " + BASE_URL)
    print("=" * 70)

    if not test_health():
        print("\nThe API is not up. Start it first:")
        print("   uvicorn api.main:app --port 8000")
        print("   or: docker compose up -d")
        sys.exit(1)

    test_valid_prediction()
    test_missing_values_imputed()
    test_unseen_category()
    test_invalid_rejected()
    test_batch()
    test_metrics()

    passed_count = 0
    for outcome in results:
        if outcome:
            passed_count = passed_count + 1

    print("\n" + "=" * 70)
    print(str(passed_count) + " / " + str(len(results)) + " checks passed")

    if passed_count != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
