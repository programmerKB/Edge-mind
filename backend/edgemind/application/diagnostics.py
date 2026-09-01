"""Database-backed diagnostic tools exposed through an application boundary."""

from __future__ import annotations

import time
from typing import Callable

from edgemind.application.forecasts import ForecastService
from edgemind.application.ports import UnitOfWork
from edgemind.application.ridge_lab import RidgeLabService
from edgemind.application.sensors import SensorService
from edgemind.domain.forecasting import ForecastError
from edgemind.domain.ridge_experiments import DIRECT_MODEL


class DiagnosticToolService:
    """Execute allow-listed status and forecast use cases for the Agent."""

    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        forecasts: ForecastService,
        ridge_lab: RidgeLabService,
        sensors: SensorService,
    ):
        """Bind use cases to the unit-of-work factory used by Agent tools."""
        self._uow_factory = uow_factory
        self._forecasts = forecasts
        self._ridge_lab = ridge_lab
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
        model_name: str = DIRECT_MODEL,
    ) -> dict:
        """Run forecast inference and record its complete tool duration."""
        started = time.perf_counter()
        try:
            with self._uow_factory() as uow:
                if model_name == DIRECT_MODEL:
                    result = self._forecasts.forecast(
                        uow,
                        motor_id,
                        auto_train=True,
                        training_motor_id=training_motor_id,
                    )
                    result.update(
                        {
                            "model_name": DIRECT_MODEL,
                            "model_label": "Direct Ridge",
                        }
                    )
                else:
                    result = self._ridge_lab.forecast(
                        uow,
                        motor_id=motor_id,
                        training_motor_id=training_motor_id or motor_id,
                        model_name=model_name,
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
                arguments.get("model_name", DIRECT_MODEL),
            )
        return {"error": "未知的工具"}
