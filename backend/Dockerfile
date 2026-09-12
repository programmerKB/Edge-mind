FROM python:3.11.16-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534
COPY --from=ghcr.io/astral-sh/uv:0.12.13@sha256:b485bd65cc2cf1c9a93b3554012c9c3778cf7b1b5fd3d3096ce9e1226c97e1e6 /uv /bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# UID 1000 matches the default local developer account and keeps bind-mounted
# source and generated reports writable without running the application as root.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /usr/sbin/nologin app \
    && python -m venv "$VIRTUAL_ENV" \
    && chown -R app:app /app "$VIRTUAL_ENV"

COPY --chown=app:app requirements.txt .

USER app

# Install dependencies before source code so Docker can reuse this layer.
RUN uv pip install -r requirements.txt

# Keep the image runnable without a bind mount; Compose overrides source files
# and enables reload only for the local development workflow.
COPY --chown=app:app . .
RUN mkdir -p /app/outputs

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
