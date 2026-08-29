"""Liveness and database-readiness probes for container orchestration."""

from collections.abc import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.ports import UnitOfWork


def create_router(uow_factory: Callable[[], UnitOfWork]) -> APIRouter:
    """Create probes without coupling the presentation layer to SQLAlchemy."""
    router = APIRouter(prefix="/api/health", tags=["health"])

    @router.get("/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/ready")
    def ready() -> dict[str, str]:
        try:
            with uow_factory() as uow:
                uow.sensors.list_device_summaries()
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database is not ready",
            ) from error
        return {"status": "ready", "database": "reachable"}

    return router
