"""Read-only endpoint for aggregate inference performance."""

from fastapi import APIRouter

from edgemind.application.ports import ReportQueryGateway


def create_router(reports: ReportQueryGateway) -> APIRouter:
    """Create performance routes against the abstract report query port."""
    router = APIRouter(prefix="/api/performance", tags=["performance"])

    @router.get("/summary")
    def get_performance_summary():
        """Return aggregate statistics for finalized inference runs."""
        return reports.performance_summary()

    return router
