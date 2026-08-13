"""Environment-backed application settings."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://agent_user:agent_pass@db:5432/motor_monitor_db",
)
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-3.5-flash-lite")
SEED_DEMO_DATA = os.getenv("SEED_DEMO_DATA", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
)

AGENT_SYSTEM_INSTRUCTION = (
    "你是一個專業的工業馬達與邊緣設備診斷助手。請根據數據回答問題，"
    "查詢狀態時呼叫狀態工具，詢問未來溫度時務必呼叫 30 分鐘預測工具。"
    "若使用者要求用 A 設備訓練的模型推論 B 設備，請傳入 motor_id=B、"
    "training_motor_id=A。回答請使用繁體中文，並給出具體的維護建議。"
)
