# Makefile — every command in this project has a short name.
#   make help    lists everything
#
# Tested on macOS (Apple Silicon) with Python 3.12 and Docker Desktop.

PYTHON := python3
VENV   := .venv
BIN    := $(VENV)/bin

# The MLflow tracking server now runs as a Docker service. Host-side
# commands reach it on localhost; the api container reaches the same server
# at http://mlflow:5000. Override to work against a local file instead:
#     make train MLFLOW_URI=sqlite:///mlflow.db
MLFLOW_PORT ?= 5000
MLFLOW_URI  ?= http://localhost:$(MLFLOW_PORT)
export MLFLOW_TRACKING_URI = $(MLFLOW_URI)

.DEFAULT_GOAL := help
.PHONY: help setup data train train-all api mlflow-up mlflow-ui traffic \
        monitor rollback predict docker-build docker-up docker-down \
        docker-logs test clean reset

help:  ## show this list
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## create .venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/python check_setup.py

data:  ## build train/validation/live splits from data/raw
	$(BIN)/python -m src.data

train:  ## train the default model (random forest, 200 trees)
	$(BIN)/python -m src.train --model rf --run-name rf-200

train-all:  ## train three models so MLflow has something to compare
	$(BIN)/python -m src.train --model linear --run-name linear
	-$(BIN)/python -m src.train --model rf --n-estimators 50 --run-name rf-50
	-$(BIN)/python -m src.train --model rf --n-estimators 200 --run-name rf-200

predict:  ## score the built-in example house
	$(BIN)/python -m src.predict

api:  ## run the API locally with auto-reload
	$(BIN)/uvicorn api.main:app --reload --port 8000

mlflow-up:  ## start ONLY the MLflow tracking server (needed before training)
	docker compose up -d mlflow
	@echo "  waiting for MLflow to report healthy..."
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' jakarta-housing-mlflow 2>/dev/null)" = "healthy" ]; do sleep 3; printf "."; done
	@echo ""
	@echo "  MLflow ready at $(MLFLOW_URI)"

mlflow-ui:  ## open the MLflow UI in a browser (server runs in Docker)
	@open $(MLFLOW_URI) 2>/dev/null || echo "  open $(MLFLOW_URI)"

traffic:  ## send two waves of traffic at the running API
	$(BIN)/python scripts/traffic.py

monitor:  ## run the drift and accuracy report
	$(BIN)/python -m src.monitor

rollback:  ## move @champion back to the previous version
	$(BIN)/python -m src.rollback

test:  ## smoke-test the running API end to end
	$(BIN)/python scripts/smoke_test.py

docker-build:  ## build the API image
	docker compose build

docker-up:  ## start mlflow + api + prometheus + grafana
	docker compose up -d
	@echo ""
	@echo "  MLflow   $(MLFLOW_URI)"
	@echo "  API      http://localhost:$${API_PORT:-8000}/docs"
	@echo "  Grafana  http://localhost:$${GRAFANA_PORT:-3000}"
	@echo "  Promth.  http://localhost:$${PROMETHEUS_PORT:-9090}"

docker-down:  ## stop everything
	docker compose down

docker-logs:  ## follow the API container logs
	docker compose logs -f api

clean:  ## remove generated data, models and logs
	rm -rf data/*.csv models/champion models/*.json logs/*.log
	find . -name __pycache__ -type d -exec rm -rf {} +

reset: clean  ## clean, and wipe the MLflow database AND its Docker volumes
	rm -rf mlruns mlflow.db
	-docker compose down -v
