FROM python:3.9-slim

LABEL maintainer="Shailesh Pawar" \
      description="MLOps Task 0 - rolling-mean trading signal batch job" \
      version="v1"

WORKDIR /app

# Install dependencies first for better layer caching — this layer only
# rebuilds when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required input artifacts.
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Run as a non-root user rather than the container default (root) —
# standard hardening for anything that reaches production.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Lightweight healthcheck: confirms the interpreter + key deps import
# cleanly inside the image. Doesn't run the full job (that's the
# ENTRYPOINT's responsibility) — this just catches a broken image early.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import pandas, numpy, yaml" || exit 1

# Default entrypoint runs the batch job with relative paths (no hardcoded
# absolute host paths). Produces metrics.json and run.log in /app, and
# prints the final metrics JSON to stdout.
ENTRYPOINT ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
