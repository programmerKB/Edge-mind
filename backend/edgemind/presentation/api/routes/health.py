"""Deployment readiness independent of optional application features."""

from typing import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.ports import UnitOfWork


def create_router(uow_factory: Callable[[], UnitOfWork]) -> APIRouter:
    """Check API and sensor-storage readiness without modifying data."""
    router = APIRouter(prefix="/api", tags=["health"])

    @router.get("/health")
    def health_check():
        """Keep deployment checks sensitive to database/schema failures."""
        try:
            with uow_factory() as uow:
                uow.sensors.list_device_summaries()
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="資料服務尚未就緒",
            ) from error
        return {"status": "ok"}

    return router
