"""REST endpoints for the compact Direct Ridge versus History experiment."""

from typing import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.ports import UnitOfWork
from edgemind.application.ridge_lab import RidgeLabService
from edgemind.domain.forecasting import ForecastError
from edgemind.presentation.schemas import (
    RidgeExperimentRequest,
    RidgeForecastRequest,
)


def create_router(
    ridge_lab: RidgeLabService,
    uow_factory: Callable[[], UnitOfWork],
) -> APIRouter:
    """Create Ridge Lab routes with explicit service dependencies."""
    router = APIRouter(prefix="/api/ridge-lab", tags=["ridge-lab"])

    @router.get("/config")
    def get_configuration():
        """Return fixed methodology plus current database availability."""
        with uow_factory() as uow:
            return ridge_lab.configuration(uow)

    @router.post("/experiments", status_code=status.HTTP_201_CREATED)
    def create_experiment(request: RidgeExperimentRequest):
        """Run and persist one chronological dual-model comparison."""
        try:
            with uow_factory() as uow:
                return ridge_lab.run_experiment(
                    uow,
                    training_motor_id=request.training_motor_id,
                    evaluation_motor_id=request.evaluation_motor_id,
                )
        except ForecastError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ridge 實驗失敗：{error}",
            ) from error

    @router.get("/experiments/{experiment_id}")
    def get_experiment(experiment_id: str):
        """Retrieve one saved experiment by its stable identifier."""
        try:
            return ridge_lab.get_experiment(experiment_id)
        except ForecastError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(error),
            ) from error

    @router.post("/forecasts", status_code=status.HTTP_201_CREATED)
    def create_forecast(request: RidgeForecastRequest):
        """Forecast 30 minutes ahead with the selected Ridge variant."""
        try:
            with uow_factory() as uow:
                return ridge_lab.forecast(
                    uow,
                    motor_id=request.motor_id,
                    training_motor_id=request.training_motor_id,
                    model_name=request.model_name,
                )
        except ForecastError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ridge 預測失敗：{error}",
            ) from error

    return router
