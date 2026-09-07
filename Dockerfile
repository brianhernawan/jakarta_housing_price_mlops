# Layer order matters: things that rarely change go at the TOP so the
# Docker build cache actually gets used.

FROM python:3.12-slim

WORKDIR /app

# Keeps the image small and the logs unbuffered so `docker logs` is live.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Dependencies first (rarely change) -> this layer gets cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Then the application code (changes often)
COPY src/ ./src
COPY api/ ./api

# 3. The model exported by `python -m src.train`, plus the training profile
COPY models/ ./models

RUN mkdir -p logs data

# Run as a non-root user. A container that does not need root should not have it.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The container reports its own health, so `docker compose ps` tells the
# truth instead of just saying "running".
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
