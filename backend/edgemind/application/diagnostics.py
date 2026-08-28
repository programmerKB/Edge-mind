"""Database-backed diagnostic tools exposed through an application boundary."""

from __future__ import annotations

import json
import time
from typing import Callable, Sequence

from edgemind.application.forecasts import ForecastService
from edgemind.application.ports import UnitOfWork
from edgemind.application.research import ResearchService
from edgemind.application.sensors import SensorService
from edgemind.domain.forecasting import ForecastError
from edgemind.domain.research import ResearchError


class DiagnosticToolService:
    """Execute allow-listed status and forecast use cases for the Agent."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        forecasts: ForecastService,
        research: ResearchService,
        sensors: SensorService,
    ):
        """Bind use cases to the unit-of-work factory used by Agent tools."""
        self._uow_factory = uow_factory
        self._forecasts = forecasts
        self._research = research
        self._sensors = sensors

    def _motor_status_payload(self, motor_id: str) -> dict:
        """Run the latest-status use case in its own unit of work."""
        try:
            with self._uow_factory() as uow:
                return self._sensors.latest_status(uow, motor_id)
        except Exception as error:
            return {"error": str(error)}

    def _temperature_forecast_payload(
        self,
        motor_id: str,
        training_motor_id: str | None = None,
    ) -> dict:
        """Run forecast inference and record its complete tool duration."""
        started = time.perf_counter()
        try:
            with self._uow_factory() as uow:
                result = self._forecasts.forecast(
                    uow,
                    motor_id,
                    auto_train=True,
                    training_motor_id=training_motor_id,
                )
            result["tool_duration_ms"] = round(
                (time.perf_counter() - started) * 1000,
                6,
            )
            return result
        except ForecastError as error:
            return {"error": str(error)}
        except Exception as error:
            return {"error": f"預測失敗：{error}"}

    def execute(self, name: str, arguments: dict) -> dict:
        """Dispatch only explicitly supported tool names."""
        if name == "get_motor_status":
            return self._motor_status_payload(arguments.get("motor_id", ""))
        if name == "get_temperature_forecast":
            return self._temperature_forecast_payload(
                arguments.get("motor_id", ""),
                arguments.get("training_motor_id"),
            )
        if name == "get_temperature_trajectory_forecast":
            return self._temperature_trajectory_payload(
                arguments.get("motor_id", ""),
                arguments.get("training_motor_id"),
                arguments.get("model_name", "ridge_history_trend"),
                arguments.get("threshold_c", 35.0),
                arguments.get("horizons_minutes", (5, 10, 15, 20, 25, 30)),
            )
        return {"error": "未知的工具"}

    def _temperature_trajectory_payload(
        self,
        motor_id: str,
        training_motor_id: str | None = None,
        model_name: str = "ridge_history_trend",
        threshold_c: float = 35.0,
        horizons_minutes: Sequence[int] = (5, 10, 15, 20, 25, 30),
    ) -> dict:
        """Run a configurable multi-horizon forecast with pending future truth."""
        started = time.perf_counter()
        try:
            with self._uow_factory() as uow:
                result = self._research.forecast_trajectory(
                    uow,
                    motor_id=motor_id,
                    training_motor_id=training_motor_id,
                    model_name=model_name,
                    threshold_c=threshold_c,
                    horizons_minutes=horizons_minutes,
                )
            result["tool_duration_ms"] = round(
                (time.perf_counter() - started) * 1000,
                6,
            )
            return result
        except ResearchError as error:
            return {"error": str(error)}
        except Exception as error:
            return {"error": f"多時域預測失敗：{error}"}

    def get_motor_status(self, motor_id: str) -> str:
        """Google function schema and execution adapter for status queries.

        Args:
            motor_id: Device identifier such as ``M1`` or ``STM32-Node-1``.
        """
        return json.dumps(
            self._motor_status_payload(motor_id),
            ensure_ascii=False,
        )

    def get_temperature_forecast(
        self,
        motor_id: str,
        training_motor_id: str | None = None,
    ) -> str:
        """Google function schema and execution adapter for forecasting.

        Args:
            motor_id: Device whose latest vector is used for inference.
            training_motor_id: Optional device supplying the persisted model.
        """
        return json.dumps(
            self._temperature_forecast_payload(motor_id, training_motor_id),
            ensure_ascii=False,
        )

    def get_temperature_trajectory_forecast(
        self,
        motor_id: str,
        training_motor_id: str | None = None,
        model_name: str = "ridge_history_trend",
        threshold_c: float = 35.0,
        horizons_minutes: Sequence[int] = (5, 10, 15, 20, 25, 30),
    ) -> str:
        """Forecast configurable +5 through +60 minute temperatures and risk.

        Args:
            motor_id: Device whose newest exact 12-reading history is forecast.
            training_motor_id: Optional separate device supplying training data.
            model_name: Available research model ID; defaults to ridge_history_trend.
            threshold_c: Engineering-approved warning threshold in degrees Celsius.
            horizons_minutes: Increasing five-minute forecast horizons through 60.
        """
        return json.dumps(
            self._temperature_trajectory_payload(
                motor_id,
                training_motor_id,
                model_name,
                threshold_c,
                horizons_minutes,
            ),
            ensure_ascii=False,
        )
