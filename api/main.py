"""
api/main.py — station 7: the model becomes an HTTP service.

Rubric item 4 (FastAPI Implementation) lives here: endpoints, request and
response validation, and error handling.

Run:
    uvicorn api.main:app --reload --port 8000
then open http://localhost:8000/docs

Endpoints
    GET  /health          is the service up and which model version is loaded
    POST /predict         one house in, one price out
    POST /predict/batch   many houses in, many prices out
    POST /admin/reload    re-read the champion alias after a rollback
    GET  /metrics         Prometheus scrape target
"""

import json
import os
import time
import uuid
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi import Request
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram
from prometheus_client import generate_latest
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from src import config
from src.features import as_model_frame
from src.logger import get_logger

log = get_logger()

# "auto" tries the MLflow registry first and falls back to the exported file.
# Inside a container we force "file" — see docker-compose.yml.
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "auto")

# ------------------------------------------------------------- metrics
# counter: only ever goes up. gauge: a current value. histogram: a spread.
PREDICTION_TOTAL = Counter(
    "prediction_total", "Prediction requests received")
PREDICTION_ERROR_TOTAL = Counter(
    "prediction_error_total", "Prediction requests that failed")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time to serve one prediction",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
PREDICTED_PRICE = Histogram(
    "predicted_price_usd", "Distribution of predicted sale prices",
    buckets=(80e3, 120e3, 150e3, 180e3, 220e3, 280e3, 350e3, 450e3, 600e3))
MODEL_VERSION_ACTIVE = Gauge(
    "model_version_active", "Model version currently being served")
INPUT_GRLIVAREA_MEAN = Gauge(
    "input_gr_liv_area_mean",
    "Mean GrLivArea across the last 100 requests")
DRIFT_SCORE = Gauge(
    "drift_score_gr_liv_area",
    "How far incoming GrLivArea has moved from the training mean")

# --------------------------------------------------------------- state
# The model is loaded ONCE at startup, not per request. Loading it per
# request would destroy latency. The cost of that choice is that a
# rollback does not reach a running API until /admin/reload is called.
state = {
    "model": None,
    "version": "not-loaded",
    "source": "-",
    "recent_gr_liv_area": [],
    "train_gr_liv_area_mean": 1500.0,
}


def load_train_profile():
    """Read the training statistics written by src/data.py.

    Hard-coding the training mean in this file would guarantee it goes
    stale the first time the data changes.
    """
    if os.path.exists(config.TRAIN_PROFILE_FILE):
        handle = open(config.TRAIN_PROFILE_FILE)
        profile = json.load(handle)
        handle.close()
        state["train_gr_liv_area_mean"] = float(profile["drift_watch_mean"])
        log.info("train_profile_loaded", extra={
            "drift_watch_mean": state["train_gr_liv_area_mean"]})


def load_model():
    """Fill state['model'] and state['version']."""
    if MODEL_SOURCE in ("auto", "registry"):
        try:
            import mlflow
            from mlflow import MlflowClient

            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            client = MlflowClient()
            version = client.get_model_version_by_alias(
                config.MODEL_NAME, config.PRODUCTION_ALIAS)
            uri = "models:/" + config.MODEL_NAME + "@" + config.PRODUCTION_ALIAS

            state["model"] = mlflow.pyfunc.load_model(uri)
            state["version"] = str(version.version)
            state["source"] = "mlflow-registry"
            MODEL_VERSION_ACTIVE.set(float(version.version))
            log.info("model_loaded", extra={
                "source": "mlflow-registry", "version": state["version"]})
            return
        except Exception as error:
            if MODEL_SOURCE == "registry":
                raise
            log.warning("registry_unavailable",
                        extra={"detail": str(error)[:160]})

    # Fallback: the plain file exported by src/train.py.
    model_path = os.path.join(config.MODEL_EXPORT_FOLDER, "model.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError(
            "No model found at " + model_path
            + ". Run `python -m src.train` before starting the API.")

    state["model"] = joblib.load(model_path)
    state["source"] = "exported-file"

    version_path = os.path.join(config.MODEL_EXPORT_FOLDER, "VERSION")
    if os.path.exists(version_path):
        handle = open(version_path)
        state["version"] = handle.read().strip()
        handle.close()

    try:
        MODEL_VERSION_ACTIVE.set(float(state["version"]))
    except ValueError:
        pass

    log.info("model_loaded", extra={
        "source": "exported-file", "version": state["version"]})


@asynccontextmanager
async def lifespan(app):
    """Everything before `yield` runs once, when the server starts."""
    load_train_profile()
    load_model()
    yield
    log.info("server_stopped")


app = FastAPI(
    title="Jakarta Housing Price Prediction API",
    description="Production model serving with MLflow registry, "
                "structured logging and Prometheus metrics. "
                "Model trained on the public Ames Housing dataset "
                "(Kaggle) used as a proxy; prices are in USD.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------- schemas
class HouseRequest(BaseModel):
    """Pydantic validates the request BEFORE it reaches the model.

    If someone sends GrLivArea = -50, FastAPI rejects it with HTTP 422 and
    we did not write one line of validation logic to make that happen.

    Note the alias on first_floor_sf. The raw column is called "1stFlrSF",
    which is not a legal Python identifier — a field cannot start with a
    digit. The alias lets the JSON payload use the real column name while
    the Python attribute stays valid. populate_by_name means both spellings
    are accepted.
    """

    model_config = ConfigDict(populate_by_name=True)

    # WHERE NULL IS ALLOWED, AND WHY
    #
    # A field is Optional here if and only if it is genuinely missing
    # somewhere in the source data. That is not laziness — it is the API
    # contract matching reality. Real records have holes in them, the
    # imputer inside the pipeline is built to fill exactly those holes, and
    # an API that rejects a record the model can happily score is an API
    # that loses business for no reason.
    #
    # Everything else stays required. Making all 24 fields optional would
    # be the lazy version, and it would let a request through that is
    # missing OverallQual — the single most predictive column in the set.
    #
    # Missing counts in the raw files:
    #   LotFrontage 259/227 · GarageFinish 81/78 · BsmtQual 37/44
    #   MSZoning 0/4 · TotalBsmtSF 0/1 · GarageCars 0/1
    #   GarageArea 0/1 · KitchenQual 0/1        (train / test)

    # --- numeric, required ---
    OverallQual: int = Field(..., ge=1, le=10, examples=[7])
    GrLivArea: float = Field(..., gt=0, le=10000, examples=[1710])
    first_floor_sf: float = Field(..., alias="1stFlrSF",
                                  gt=0, le=6000, examples=[856])
    FullBath: int = Field(..., ge=0, le=5, examples=[2])
    YearBuilt: int = Field(..., ge=1800, le=2030, examples=[2003])
    YearRemodAdd: int = Field(..., ge=1800, le=2030, examples=[2003])
    LotArea: float = Field(..., gt=0, le=250000, examples=[8450])
    TotRmsAbvGrd: int = Field(..., ge=1, le=20, examples=[8])
    Fireplaces: int = Field(..., ge=0, le=5, examples=[0])
    OverallCond: int = Field(..., ge=1, le=10, examples=[5])
    BedroomAbvGr: int = Field(..., ge=0, le=10, examples=[3])

    # --- numeric, nullable ---
    TotalBsmtSF: float | None = Field(None, ge=0, le=10000, examples=[856])
    GarageCars: int | None = Field(None, ge=0, le=5, examples=[2])
    GarageArea: float | None = Field(None, ge=0, le=2000, examples=[548])
    LotFrontage: float | None = Field(None, ge=0, le=500, examples=[65])

    # --- categorical, required ---
    # Plain strings, deliberately not Enums. A category the model has never
    # seen must NOT crash the service. OneHotEncoder(handle_unknown="ignore")
    # absorbs it and the drift monitor is what tells us it happened.
    Neighborhood: str = Field(..., min_length=1, examples=["CollgCr"])
    HouseStyle: str = Field(..., min_length=1, examples=["2Story"])
    ExterQual: str = Field(..., min_length=1, examples=["Gd"])
    CentralAir: str = Field(..., min_length=1, examples=["Y"])
    SaleCondition: str = Field(..., min_length=1, examples=["Normal"])

    # --- categorical, nullable ---
    # BsmtQual is null exactly when the house has no basement, and
    # GarageFinish when it has no garage. Null is information, not an error.
    MSZoning: str | None = Field(None, examples=["RL"])
    KitchenQual: str | None = Field(None, examples=["Gd"])
    BsmtQual: str | None = Field(None, examples=["Gd"])
    GarageFinish: str | None = Field(None, examples=["RFn"])

    def to_row(self):
        """Convert to a dict keyed by the real dataset column names."""
        return self.model_dump(by_alias=True)


class HouseResponse(BaseModel):
    predicted_price_usd: float
    model_version: str
    model_source: str
    latency_ms: float
    request_id: str


class BatchRequest(BaseModel):
    houses: list[HouseRequest] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[float]
    count: int
    model_version: str
    latency_ms: float
    request_id: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_source: str
    n_features: int


# ------------------------------------------------------- error handling
@app.exception_handler(RequestValidationError)
async def handle_validation_error(request, error):
    """Turn Pydantic's rejection into a clean, logged 422.

    Without this the client still gets a 422, but nothing is written to our
    log — so a client sending malformed requests all day would be invisible.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"][1:])
        problems.append({"field": location, "problem": item["msg"]})

    PREDICTION_ERROR_TOTAL.inc()
    log.warning("validation_rejected", extra={
        "request_id": request_id,
        "path": request.url.path,
        "problems": problems,
    })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "invalid request", "request_id": request_id,
                 "problems": problems},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request, error):
    """Catch anything unplanned, log it, and return a 500 without a stack trace.

    Leaking a traceback to the caller is an information disclosure problem.
    The detail goes to our log, not to the client.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    PREDICTION_ERROR_TOTAL.inc()
    log.error("unhandled_error", extra={
        "request_id": request_id,
        "path": request.url.path,
        "detail": str(error)[:300],
    })
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal error", "request_id": request_id},
    )


# ------------------------------------------------------- request logging
@app.middleware("http")
async def log_every_request(request: Request, call_next):
    started = time.perf_counter()
    request.state.request_id = uuid.uuid4().hex[:8]

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request.state.request_id

    if request.url.path not in ("/metrics", "/health"):
        log.info("http", extra={
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round(elapsed_ms, 2),
        })

    return response


# ------------------------------------------------------------- helpers
def update_drift_metrics(gr_liv_area):
    """Track the mean GrLivArea over a rolling window of 100 requests."""
    recent = state["recent_gr_liv_area"]
    recent.append(gr_liv_area)
    if len(recent) > 100:
        recent.pop(0)

    total = 0.0
    for value in recent:
        total = total + value
    mean = total / len(recent)

    baseline = state["train_gr_liv_area_mean"]
    INPUT_GRLIVAREA_MEAN.set(mean)
    if baseline > 0:
        DRIFT_SCORE.set(abs(mean - baseline) / baseline)


# ----------------------------------------------------------- endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    ready = state["model"] is not None
    return HealthResponse(
        status="healthy" if ready else "not ready",
        model_version=state["version"],
        model_source=state["source"],
        n_features=len(config.FEATURES),
    )


@app.post("/predict", response_model=HouseResponse)
def predict(house: HouseRequest, request: Request):
    started = time.perf_counter()
    PREDICTION_TOTAL.inc()

    row = house.to_row()
    frame = as_model_frame(row)
    price = float(state["model"].predict(frame)[0])

    elapsed = time.perf_counter() - started
    PREDICTION_LATENCY.observe(elapsed)
    PREDICTED_PRICE.observe(price)
    update_drift_metrics(house.GrLivArea)

    # This log line is the fuel for the drift report. Record the features,
    # the prediction, the latency and the model version.
    # Never log personal data — no names, no addresses, no ID numbers.
    entry = {
        "request_id": request.state.request_id,
        "prediction": round(price, 2),
        "model_version": state["version"],
        "latency_ms": round(elapsed * 1000, 2),
    }
    for column in config.FEATURES:
        entry[column] = row[column]
    log.info("prediction", extra=entry)

    return HouseResponse(
        predicted_price_usd=round(price, 2),
        model_version=state["version"],
        model_source=state["source"],
        latency_ms=round(elapsed * 1000, 2),
        request_id=request.state.request_id,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request):
    started = time.perf_counter()

    rows = []
    for house in payload.houses:
        rows.append(house.to_row())
    PREDICTION_TOTAL.inc(len(rows))

    frame = as_model_frame(pd.DataFrame(rows))
    prices = state["model"].predict(frame)

    elapsed = time.perf_counter() - started
    PREDICTION_LATENCY.observe(elapsed)

    results = []
    for price in prices:
        value = float(price)
        results.append(round(value, 2))
        PREDICTED_PRICE.observe(value)

    for row in rows:
        update_drift_metrics(float(row["GrLivArea"]))

    log.info("prediction_batch", extra={
        "request_id": request.state.request_id,
        "count": len(rows),
        "model_version": state["version"],
        "latency_ms": round(elapsed * 1000, 2),
    })

    return BatchResponse(
        predictions=results,
        count=len(results),
        model_version=state["version"],
        latency_ms=round(elapsed * 1000, 2),
        request_id=request.state.request_id,
    )


@app.post("/admin/reload")
def reload_model(request: Request):
    """Re-read the @champion alias. Used after a rollback.

    In production an endpoint like this must be protected — a token, or an
    IP allow-list. It is wide open here because this is a teaching repo.
    """
    old_version = state["version"]
    load_train_profile()
    load_model()

    log.warning("model_reloaded", extra={
        "request_id": request.state.request_id,
        "old_version": old_version,
        "new_version": state["version"],
    })

    return {
        "status": "model reloaded",
        "old_version": old_version,
        "current_version": state["version"],
    }


@app.get("/metrics")
def metrics():
    """Prometheus PULLS this endpoint every few seconds."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
