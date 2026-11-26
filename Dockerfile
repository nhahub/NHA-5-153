FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    ATTRITION_HOME=/app

WORKDIR /app

# Install dependencies first (cached layer) from the project metadata, then the code.
COPY pyproject.toml README.md params.yaml ./
COPY src ./src
RUN pip install .

# The trained champion (model + card + drift reference). Re-created by `python -m attrition.train`.
COPY models ./models

# Non-root user; logs/ monitoring output is written to a mounted volume
RUN useradd --create-home appuser && mkdir -p /app/monitoring && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

# Default: REST API. docker-compose overrides the command for the dashboard / monitor / MLflow services.
CMD ["uvicorn", "attrition.api:app", "--host", "0.0.0.0", "--port", "8000"]
