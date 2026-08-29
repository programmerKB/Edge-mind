FROM python:3.11-slim
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Install dependencies before source code so Docker can reuse this layer.
ARG REQUIREMENTS_FILE=requirements-research.txt
COPY requirements*.txt ./
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    -r "${REQUIREMENTS_FILE}"

# Run immutable application code as an unprivileged user; generated research
# artifacts are written to the dedicated /app/outputs volume.
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/outputs \
    && chown -R app:app /app
COPY --chown=app:app . .
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health/ready', timeout=3)"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
