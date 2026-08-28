"""REST endpoints for reproducible multi-horizon forecasting experiments."""

from typing import Callable

from fastapi import APIRouter, HTTPException, status

from edgemind.application.ports import UnitOfWork
from edgemind.application.research import ResearchService
from edgemind.domain.research import ResearchError
from edgemind.presentation.schemas import (
    ResearchExperimentRequest,
    ResearchForecastRequest,
)


def create_router(
    research: ResearchService,
    uow_factory: Callable[[], UnitOfWork],
) -> APIRouter:
    """Bind the research workbench to injected application dependencies."""
    router = APIRouter(prefix="/api/research", tags=["research"])

    @router.get("/config")
    def get_research_configuration():
        """Return protocol defaults, model availability, and dataset inventory."""
        try:
            with uow_factory() as uow:
                return research.configuration(uow)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"研究設定讀取失敗：{error}",
            ) from error

    @router.post("/experiments", status_code=status.HTTP_201_CREATED)
    def create_research_experiment(request: ResearchExperimentRequest):
        """Execute the selected comparisons and durably save measured results."""
        try:
            with uow_factory() as uow:
                return research.run(
                    uow,
                    training_motor_id=request.training_motor_id,
                    evaluation_motor_id=request.evaluation_motor_id,
                    history_minutes=request.history_minutes,
                    horizons_minutes=request.horizons_minutes,
                    threshold_c=request.threshold_c,
                    model_names=request.model_names,
                    include_ablations=request.include_ablations,
                    sampling_minutes=request.sampling_minutes,
                    train_fraction=request.train_fraction,
                    validation_fraction=request.validation_fraction,
                    gap_steps=request.gap_steps,
                    walk_forward_folds=request.walk_forward_folds,
                    random_seed=request.random_seed,
                )
        except ResearchError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"研究實驗執行失敗：{error}",
            ) from error

    @router.get("/experiments/{experiment_id}")
    def get_research_experiment(experiment_id: str):
        """Retrieve a completed result after refresh or process restart."""
        result = research.get(experiment_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="找不到指定的研究實驗",
            )
        return result

    @router.post("/forecasts", status_code=status.HTTP_201_CREATED)
    def create_research_forecast(request: ResearchForecastRequest):
        """Forecast the newest exact 12-step window with future truth pending."""
        try:
            with uow_factory() as uow:
                return research.forecast_trajectory(
                    uow,
                    motor_id=request.motor_id,
                    training_motor_id=request.training_motor_id,
                    model_name=request.model_name,
                    history_minutes=request.history_minutes,
                    horizons_minutes=request.horizons_minutes,
                    threshold_c=request.threshold_c,
                    sampling_minutes=request.sampling_minutes,
                    random_seed=request.random_seed,
                )
        except ResearchError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(error),
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"多時域溫度預測失敗：{error}",
            ) from error

    @router.get("/forecasts/{forecast_id}")
    def get_research_forecast(forecast_id: str):
        """Retrieve one saved trajectory while its future truth is pending."""
        result = research.get_forecast(forecast_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="找不到指定的多時域預測",
            )
        return result

    return router
