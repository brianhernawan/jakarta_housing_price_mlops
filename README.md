# Jakarta Housing Price Prediction — End-to-End MLOps

Production-grade housing price prediction: modular code, MLflow experiment
tracking and a model registry, structured logging, drift monitoring, a
FastAPI service, and a four-container Docker deployment.

## Data provenance — read this first

This project is **branded** as a Jakarta housing scenario, but the model is
trained on the **public Ames Housing dataset** (Ames, Iowa, USA), obtained
from Kaggle and supplied as `data/raw/train.csv` and `data/raw/test.csv`.
No Jakarta property data is used anywhere in this repository.

That has three visible consequences, and none of them are bugs:

* **Prices are in US dollars.** 1,460 labelled sales ranging from $34,900 to
  $755,000, median $163,000. These are Iowa prices, not Jakarta prices.
* **Neighbourhood codes are American.** `NAmes` is North Ames, `IDOTRR` is
  the Iowa DOT and Rail Road district, `CollgCr` is College Creek. They are
  passed through untouched.
* **`MSZoning` uses US zoning classes** (`RL`, `RM`, `FV` and so on), which
  do not map onto Indonesian zoning.

The Ames dataset is used because it is the standard public benchmark for
regression on tabular housing data, and because the point of this project is
the **MLOps pipeline**, not the property market. Every technique here —
the preprocessing pipeline, the registry, the drift detection, the rollback
— transfers unchanged to a real Jakarta dataset. Only the CSV would change.

Do not present the predicted prices as Jakarta valuations.

---

## The problem this repo solves

**Situation.** A model that scores well in a notebook is not a product. To be
a product it needs to be reproducible, servable, observable, and reversible.

**Complication.** Every one of those four properties fails silently. A model
with a broken preprocessing step does not raise an exception — it returns a
number that is wrong. An API serving a stale model returns HTTP 200. A model
whose input distribution has moved returns HTTP 200. None of these show up in
an error log, because none of them are errors.

**Question.** How do you find out that a model has stopped working, before a
customer does?

**Resolution.** Two layers of observation, and one label you can move.

- The **data layer** compares incoming features against the training
  distribution. It needs no ground truth, so it can run today.
- The **model layer** compares MAE against the training baseline. It needs
  ground truth, which in a housing market arrives months after the sale.
- When either fires, a **registry alias** moves the served model back to a
  known-good version. No rebuild, no redeploy, no commit.

This repo demonstrates all three, with real measured numbers.

---

## What is in here

| File | What it does | Rubric item |
|---|---|---|
| `src/config.py` | Every path, threshold and column name. No magic numbers elsewhere | 1 |
| `src/features.py` | The one function that builds a model input frame | 1 |
| `src/data.py` | Load → validate → split → generate production traffic | 1 |
| `src/train.py` | Fit → measure → **quality gate** → register in MLflow | 1, 2 |
| `src/predict.py` | Load the `@champion` model and score | 1 |
| `src/logger.py` | Structured JSON logging, one line per event | 3 |
| `src/monitor.py` | KS drift test + MAE degradation report | 3 |
| `src/rollback.py` | Move `@champion` back to a previous version | 3 |
| `api/main.py` | FastAPI: validation, error handling, Prometheus metrics | 4 |
| `scripts/traffic.py` | Sends two waves of real traffic — normal, then drifted | 3 |
| `scripts/smoke_test.py` | Seven end-to-end checks against a running API | 4 |
| `monitoring/` | Prometheus config + pre-provisioned Grafana dashboard | 3, 5 |
| `Dockerfile` | Non-root, cache-ordered layers, container healthcheck | 5 |
| `mlflow.Dockerfile` | MLflow tracking server image, version-pinned | 2, 5 |
| `docker-compose.yml` | Four services: mlflow + api + prometheus + grafana | 2, 5 |
| `Makefile` | Short name for every command. `make help` | 1 |

---

## Assignment mapping

**1. Modular code.** Six modules in `src/`, each with one job. Configuration is
isolated in `config.py`; nothing else hard-codes a path, a column name or a
threshold. `features.py` exists specifically so that training, serving, batch
scoring and monitoring build their inputs through one shared function instead
of four copies that quietly diverge.

**2. MLflow integration.** `src/train.py` logs 9 parameters, 4 metrics
(MAE, RMSE, R², encoded feature count), 3 tags (git SHA, data MD5, gate
result), the feature spec as an artifact, and the model itself to the
registry. Registry aliases `@champion` / `@challenger` / `@previous-champion`
control which version is served.

**3. Logging and monitoring.** `src/logger.py` writes one JSON object per
event to `logs/predictions.log` — preprocessing, model loads, every prediction
with its features and latency, every validation rejection, every reload.
`src/monitor.py` reads that log back with pandas and runs a
Kolmogorov-Smirnov test per numeric column. The API exposes 7 Prometheus
metrics, and Grafana draws them on a pre-provisioned dashboard.

**4. FastAPI.** Five endpoints. Pydantic validates all 24 fields with bounds
before anything reaches the model. Two exception handlers turn validation
failures and unexpected errors into logged, structured JSON responses — no
stack traces leak to the caller.

**5. Docker.** A `Dockerfile` running as a non-root user with a container
healthcheck, and a `docker-compose.yml` with a FastAPI service plus two
monitoring services on a named network. A fourth service runs the MLflow
tracking server, so the registry is a real network service rather than a
SQLite file on one laptop.

---

## Run it from nothing

### 0. Setup, once

```bash
make setup
source .venv/bin/activate
```

This creates `.venv`, installs pinned dependencies, and runs `check_setup.py`.
Every line should show `[ok]`.

Prefer conda? `conda create -n house-price python=3.12 -y && conda activate
house-price && pip install -r requirements.txt` works identically — the
Makefile just assumes `.venv`.

### 1. Start the tracking server, then build the datasets

The MLflow tracking server now runs in Docker, so it must be up before any
training happens. It is the registry that `src/train.py` writes to.

```bash
make mlflow-up        # waits until it reports healthy
make data
```

`make mlflow-up` starts only that one service; the other three come later.
Every host-side command (`make train`, `make monitor`, `make rollback`)
picks up `MLFLOW_TRACKING_URI=http://localhost:5000` automatically from the
Makefile. To work against a plain local file instead:

```bash
make train MLFLOW_URI=sqlite:///mlflow.db
```

`data/raw/test.csv` has no `SalePrice` column — it is the Kaggle leaderboard
holdout, so it cannot measure anything. Instead of pretending otherwise, this
project gives it its real job: unlabelled traffic arriving at the API. All
evaluation sets are carved out of `train.csv`:

```
train.csv (1460 labelled)
  ├── 10%  → 146 rows  → delayed-label pool
  └── 90%
        ├── 80% → 1051 rows → data/train.csv
        └── 20% →  263 rows → data/validation.csv
```

It also builds the "three months later" scenario: a new high-end development
opens and the sales mix shifts toward larger, newer houses. This is produced
by **resampling real rows** with an upmarket weighting, not by inventing
synthetic houses. Mean `GrLivArea` moves from 1,533 to 2,348 sq ft.

### 2. Train and compare

```bash
make train-all
```

Measured results on the validation set (seed 42):

| Version | Run | MAE | RMSE | R² |
|---|---|---|---|---|
| v1 | rf-200 | $16,618 | $25,575 | 0.8749 |
| v2 | linear | $16,672 | $23,703 | **0.8925** |
| v3 | rf-50 | $16,937 | $25,725 | 0.8734 |

Worth arguing about rather than glossing over: **linear regression wins on
RMSE and R², random forest wins on MAE.** RMSE squares the errors, so it
punishes the few very expensive houses the forest cannot extrapolate to.
MAE treats every dollar equally. Which metric matters is a business question,
not a modelling one — and this repo gates on MAE.

Open the tracking UI to compare runs side by side:

```bash
make mlflow-ui        # http://localhost:5000
```

### 3. See the quality gate refuse a bad model

```bash
python -m src.train --model rf --max-depth 1 --run-name rf-broken
```

```
      MAE  : $ 39,480
[6] Quality gate: MAE must be <= $ 25,000
      REJECTED. The model is not registered.
```

The pipeline exits non-zero and the model never reaches the registry. In CI,
that is a failed build.

A threshold alone is not enough, though. A model that clears $25,000 but is
still worse than the current champion must not be promoted either — which is
why `--promote` is an explicit flag rather than the default.

### 4. Serve it

```bash
make api              # http://localhost:8000/docs
```

In a second terminal:

```bash
make test             # 7 end-to-end checks
```

```
[ok]    health endpoint                    v4 via mlflow-registry
[ok]    valid house scored                 $ 193,115
[ok]    null values imputed, not rejected  $ 194,698
[ok]    unseen category does not crash     HTTP 200
[ok]    impossible house rejected 422      HTTP 422, 2 problems reported
[ok]    batch scoring                      3 scored
[ok]    metrics endpoint                   all 7 present
```

### 5. Deploy the full stack

```bash
make docker-up
```

| Service | URL | Notes |
|---|---|---|
| MLflow | http://localhost:5000 | tracking UI + model registry |
| API | http://localhost:8000/docs | interactive Swagger UI |
| Grafana | http://localhost:3000 | dashboard already provisioned |
| Prometheus | http://localhost:9090 | raw metric browser |

Port already taken? `GRAFANA_PORT=3001 docker compose up -d`. Copy
`.env.example` to `.env` to make it permanent.

Startup is ordered by healthcheck, not by luck: prometheus waits for api,
and api waits for mlflow. `docker compose ps` should show all four up with
`jakarta-housing-mlflow` and `jakarta-housing-api` marked **healthy**.

Confirm the API is really talking to the registry:

```bash
curl -s http://localhost:8000/health
```

`"model_source":"mlflow-registry"` means it loaded over the network.
`"exported-file"` means the registry was unreachable and it fell back to the
model baked into the image — see the troubleshooting section.

### 6. Watch it fail silently

```bash
make traffic
```

Two waves of **real houses** from the dataset. Wave 1 is normal traffic from
`data/raw/test.csv`. Wave 2 is the drifted mix.

```
>> Wave 'normal'   mean GrLivArea = 1492 sq ft   median predicted $ 159,201
>> Wave 'drifted'  mean GrLivArea = 2347 sq ft   median predicted $ 281,006
   done: 150 ok, 0 failed
```

In Grafana the drift gauge crosses into red, the mean-GrLivArea line
separates from its training baseline, and the predicted-price distribution
jumps. **The error counter stays at zero the entire time.**

### 7. Prove it numerically

```bash
make monitor
```

```
[1] DRIFT CHECK
  column            train mean      live mean      shift     p-value   status
  OverallQual               6.1            7.5     +22.5%     9.8e-19   DRIFT
  GrLivArea              1533.4         2140.7     +39.6%     2.7e-16   DRIFT
  TotalBsmtSF            1065.4         1626.6     +52.7%     6.3e-19   DRIFT
  ...
  BedroomAbvGr              2.9            2.9      +2.0%        0.21   ok

[2] ACCURACY CHECK
  MAE at training     : $       16,618
  MAE on live data    : $       67,491
  degradation         : 4.1x
```

14 of 15 numeric columns drifted. MAE degraded **4.1×**. Not one exception
was raised. Every request was answered HTTP 200 — with the wrong number.

That is the entire point of the exercise.

### 8. Roll back

```bash
make rollback
curl -X POST http://localhost:8000/admin/reload
```

```
  ROLLBACK: v4  ->  v1
      @previous-champion -> v4
      @champion -> v1
  [export] champion v1 -> models/champion
```

No rebuild. No redeploy. No commit. A label moved — and because the
registry is a network service, the API sees the new alias without anything
being copied between machines. The file export in `src/train.py` is now
only a fallback for when the registry is unreachable.

The `/admin/reload` call is necessary because the model is loaded once at
startup, not per request — loading it per request would destroy latency. That
tradeoff is deliberate, and the reload endpoint is the price of it. In
production it would sit behind a token or an IP allow-list.

---

## Design decisions worth defending

**Preprocessing lives inside the model object.** Imputation, scaling and
one-hot encoding are all steps of a single sklearn `Pipeline`, and the log
transform is wrapped in a `TransformedTargetRegressor`. At serving time we
call `.predict()` and nothing else. If any of those steps lived outside the
pipeline — imputing in a notebook, scaling in a script, encoding by hand in
the API — training and serving would eventually disagree, and the symptom
would be wrong numbers with no error. That is training-serving skew, and it
is the most expensive bug in this whole domain.

**24 features, not 79.** 15 numeric plus 9 categorical, expanding to 76
columns after encoding. A Pydantic schema with 79 fields makes the `/docs`
page unusable, and these 24 already reach R² ≈ 0.89.

**Nullable exactly where the data is null.** A field is `Optional` if and only
if it is genuinely missing somewhere in the source files — `LotFrontage`
(259/227 missing), `GarageFinish` (81/78), `BsmtQual` (37/44), `MSZoning`,
`TotalBsmtSF`, `GarageCars`, `GarageArea`, `KitchenQual`. `BsmtQual` is null
precisely when the house has no basement. Null is information, not corruption.
Everything else stays required — making all 24 optional would let a request
through that is missing `OverallQual`, the single most predictive column.

**`handle_unknown="ignore"` is not optional.** Without it, the first request
containing a `Neighborhood` the model never saw would crash the service. With
it, the request is scored and the drift monitor is what tells you it happened.
Detect, do not crash.

**Missing values are reported, not fatal.** A naive data checkpoint that
raises on any NaN is unusable on real data — this dataset has 377 missing
values in the training columns alone. `src/data.py` fails on *structural*
problems (absent columns, a house with zero living area, a sale price of zero)
and merely reports nulls, which the imputer is built to handle.

**`1stFlrSF` needs an alias.** A Python field cannot start with a digit. The
attribute is `first_floor_sf`, aliased to `1stFlrSF`, with
`populate_by_name=True` so both spellings are accepted.

**Monitoring reads a window, not all of history.** `config.MONITOR_WINDOW`
is 200. The log holds old and new traffic mixed together; averaging over all
of it dilutes the signal until drift slips past. Every production dashboard
has a time range — never "since the beginning of time".

---

## Reproducibility

Four things pin a result, and all four are recorded:

| Leg | How |
|---|---|
| Code | `git_sha` tag on every MLflow run |
| Data | `data_md5` tag on every MLflow run |
| Seed | `RANDOM_STATE = 42`, logged as a parameter |
| Environment | `requirements.txt` with `==` pins |

Versions are pinned to a set verified to work together on Python 3.11–3.13,
including Apple Silicon. Note `pandas==2.3.3` is deliberate: pandas 3.x is
available but ships breaking changes.

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + which model version is loaded |
| POST | `/predict` | one house in, one price out |
| POST | `/predict/batch` | up to 500 houses at once |
| POST | `/admin/reload` | re-read the `@champion` alias after a rollback |
| GET | `/metrics` | Prometheus scrape target |

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "OverallQual": 7, "GrLivArea": 1710, "TotalBsmtSF": 856,
    "GarageCars": 2, "GarageArea": 548, "1stFlrSF": 856,
    "FullBath": 2, "YearBuilt": 2003, "YearRemodAdd": 2003,
    "LotArea": 8450, "LotFrontage": 65, "TotRmsAbvGrd": 8,
    "Fireplaces": 0, "OverallCond": 5, "BedroomAbvGr": 3,
    "Neighborhood": "CollgCr", "MSZoning": "RL", "HouseStyle": "2Story",
    "ExterQual": "Gd", "KitchenQual": "Gd", "BsmtQual": "Gd",
    "GarageFinish": "RFn", "CentralAir": "Y", "SaleCondition": "Normal"
  }'
```

```json
{
  "predicted_price_usd": 193115.15,
  "model_version": "4",
  "model_source": "mlflow-registry",
  "latency_ms": 27.49,
  "request_id": "224862b8"
}
```

### Prometheus metrics

| Metric | Type | Meaning |
|---|---|---|
| `prediction_total` | counter | requests received |
| `prediction_error_total` | counter | requests that failed |
| `prediction_latency_seconds` | histogram | time to serve one prediction |
| `predicted_price_usd` | histogram | distribution of predicted prices |
| `model_version_active` | gauge | version currently in memory |
| `input_gr_liv_area_mean` | gauge | mean GrLivArea over last 100 requests |
| `drift_score_gr_liv_area` | gauge | distance from the training mean |

---

## Troubleshooting

**`No model found at models/champion/model.joblib`** — run `make train`
before starting the API.

**`RESOURCE_DOES_NOT_EXIST: alias champion`** — the registry is empty. Run
`make mlflow-up && make data && make train`.

**`/health` reports `"model_source":"exported-file"` when you expected
`"mlflow-registry"`** — the API could not reach the tracking server and fell
back to the model baked into the image. It is serving correctly, just not
from the registry. Check `docker compose logs api | grep registry_unavail`
and confirm `jakarta-housing-mlflow` is healthy. This fallback is deliberate: it
is what stops the API from crash-looping when mlflow is down.

**Model artifacts fail to load with a path error** — the tracking server was
started without `--serve-artifacts`. Without that flag the client writes
model files to its own local disk and records a path the api container
cannot see. The flag is already in `docker-compose.yml`; if you changed the
command, put it back.

**API container sits in `starting` for minutes** — a tracking server that
accepts TCP connections but never answers will hang the MLflow client.
MLflow's defaults are a 120s timeout with 7 retries, so this can block for
several minutes, and the try/except fallback cannot help because a hang
never raises an exception. The three `MLFLOW_HTTP_REQUEST_*` variables in
`docker-compose.yml` bound the worst case to about 18 seconds.

**Grafana panels all say "No data"** — Prometheus cannot reach the API.
Check `docker compose ps` shows `jakarta-housing-api` as healthy, then open
http://localhost:9090/targets and confirm `jakarta-housing-api` is UP.

**Port 3000 already in use** — `GRAFANA_PORT=3001 docker compose up -d`.
macOS AirPlay Receiver and most Node dev servers both want 3000.

**Rollback ran but the API still serves the old version** — call
`POST /admin/reload`. This is expected behaviour, not a bug; see step 8.

**`UntrustedTypesFoundException` from skops** — the model is logged with
`serialization_format="cloudpickle"` for exactly this reason. MLflow's newer
skops default refuses to round-trip a `ColumnTransformer`.

---

## Commands

```
make help          list every target
make setup         create .venv and install dependencies
make data          build train/validation/live splits
make train         train the default model
make train-all     train three models for comparison
make predict       score the built-in example house
make api           run the API locally with auto-reload
make mlflow-up     start only the MLflow tracking server
make mlflow-ui     open the MLflow UI in a browser
make test          seven end-to-end API checks
make traffic       send normal then drifted traffic
make monitor       drift and accuracy report
make rollback      move @champion back one version
make docker-build  build the API image
make docker-up     start mlflow + api + prometheus + grafana
make docker-down   stop everything
make docker-logs   follow the API container logs
make clean         remove generated data, models and logs
make reset         clean, and wipe the MLflow database
```
