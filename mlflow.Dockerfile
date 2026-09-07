# mlflow.Dockerfile — the MLflow tracking server as its own container.
#
# Why build this instead of pulling ghcr.io/mlflow/mlflow?
#
#   1. VERSION PARITY. Your laptop runs mlflow==3.15.1. Pinning the same
#      version here means the client and the server are the same code.
#      Client/server skew in MLflow produces confusing errors that look
#      like bugs in your project.
#   2. NO REGISTRY LOGIN. The official image lives on GitHub Container
#      Registry, whose docs walk you through `docker login ghcr.io` with a
#      personal access token. That is friction for no benefit here.
#
# Cost: one extra ~60 second build, cached afterwards.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pinned to match requirements.txt exactly. If you bump one, bump both.
RUN pip install --no-cache-dir mlflow==3.15.1

# backend-store  = the SQLite database holding runs, params, metrics, aliases
# artifacts      = the actual model files
WORKDIR /mlflow
RUN mkdir -p /mlflow/db /mlflow/artifacts

EXPOSE 5000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status==200 else 1)"
