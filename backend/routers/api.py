"""Aggregate feature routers into the application's versionless API tree."""

from fastapi import APIRouter

from routers import artifacts, chat, performance, predictions, sensors


api_router = APIRouter()
api_router.include_router(sensors.router)
api_router.include_router(predictions.router)
api_router.include_router(performance.router)
api_router.include_router(artifacts.router)
api_router.include_router(chat.router)
