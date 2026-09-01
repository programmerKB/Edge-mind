FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app

# Install dependencies before source code so Docker can reuse this layer.
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Keep the image runnable without a bind mount; Compose overrides source files
# and enables reload only for the local development workflow.
COPY . .
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
