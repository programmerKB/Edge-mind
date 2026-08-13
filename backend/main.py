"""FastAPI application composition root."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lifecycle import application_lifespan
from routers import chat, performance, predictions, sensors
from settings import CORS_ORIGINS


app = FastAPI(
    title="邊緣設備診斷 Agent API",
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ORIGINS),
    allow_credentials=CORS_ORIGINS != ("*",),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(sensors.router)
app.include_router(predictions.router)
app.include_router(performance.router)
app.include_router(chat.router)
