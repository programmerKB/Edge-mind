"""Assemble all feature routers from explicitly injected dependencies."""

from fastapi import APIRouter

from edgemind.presentation.api.routes import (
    artifacts,
    chat,
    health,
    performance,
    predictions,
    research,
    sensors,
)


def create_api_router(container) -> APIRouter:
    """Build the versionless API tree without accessing global services."""
    router = APIRouter()
    router.include_router(health.create_router(container.uow_factory))
    router.include_router(
        sensors.create_router(container.sensors, container.uow_factory)
    )
    router.include_router(
        predictions.create_router(container.forecasts, container.uow_factory)
    )
    router.include_router(
        research.create_router(container.research, container.uow_factory)
    )
    router.include_router(performance.create_router(container.reports))
    router.include_router(artifacts.create_router(container.reports))
    router.include_router(chat.create_router(container.agent))
    return router
