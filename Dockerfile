# NexusQuant API service image. The dev venv runs Python 3.14 and every
# dependency (lightgbm, xgboost, psycopg[binary], ...) installs cleanly on
# it, so the container matches the dev runtime.
FROM python:3.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps: nothing exotic (psycopg[binary] ships its own libpq).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application (data is bind-mounted so fresh MT5 parquet files are
# visible without rebuilding).
COPY api/ ./api/
COPY src/ ./src/
COPY config/ ./config/

# Bind-mounted at runtime: ./data:/app/data, ./models:/app/models
RUN mkdir -p /app/data /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
