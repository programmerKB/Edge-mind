FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt
# 下面這行改了：透過 python -m 來呼叫 uvicorn，這樣就絕對不會有路徑找不到的問題
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]