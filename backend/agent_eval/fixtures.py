"""Deterministic tool fixtures used by EdgeMind-AgentEval.

The benchmark evaluates the Agent against controlled observations.  Keeping
the observations stable separates Agent regressions from database and Ridge
model drift, while production-tool integration remains covered by the normal
application test suite.
"""

from __future__ import annotations


STATUS_BY_MOTOR = {
    "M1": {
        "temperature": 31.2,
        "humidity": 55.4,
        "accel_x": 0.12,
        "accel_y": 0.08,
        "accel_z": 0.98,
        "status": "normal",
    },
    "M2": {
        "temperature": 37.8,
        "humidity": 61.1,
        "accel_x": 0.31,
        "accel_y": 0.27,
        "accel_z": 1.14,
        "status": "warning",
    },
    "DEMO-1": {
        "temperature": 32.4,
        "humidity": 58.2,
        "accel_x": 0.14,
        "accel_y": 0.11,
        "accel_z": 1.01,
        "status": "normal",
    },
    "DEMO-2": {
        "temperature": 36.7,
        "humidity": 63.5,
        "accel_x": 0.28,
        "accel_y": 0.24,
        "accel_z": 1.12,
        "status": "warning",
    },
    "PUMP-A": {
        "temperature": 29.8,
        "humidity": 52.6,
        "accel_x": 0.09,
        "accel_y": 0.07,
        "accel_z": 0.96,
        "status": "normal",
    },
    "FAN-7": {
        "temperature": 41.3,
        "humidity": 65.0,
        "accel_x": 0.42,
        "accel_y": 0.38,
        "accel_z": 1.26,
        "status": "critical",
    },
}

FORECAST_TEMPERATURE = {
    "M1": 33.6,
    "M2": 39.1,
    "DEMO-1": 34.2,
    "DEMO-2": 38.4,
    "PUMP-A": 31.5,
    "FAN-7": 43.0,
}

ERROR_FIXTURES = {
    "error_device_missing": "找不到設備 UNKNOWN-404 的感測資料",
    "error_insufficient_data": "設備 NEW-1 的有效資料不足，至少需要 12 組配對",
    "error_model_missing": "找不到 TRAIN-X 的已訓練模型",
    "error_api_timeout": "資料服務逾時，請稍後再試",
    "error_database": "感測資料庫暫時無法連線",
}

MODEL_LABELS = {
    "ridge_direct": "Direct Ridge",
    "ridge_history": "Ridge + History",
}


def tool_result(fixture_id: str, name: str, arguments: dict) -> dict:
    """Return the controlled result for one allow-listed tool call."""
    if fixture_id in ERROR_FIXTURES:
        return {"error": ERROR_FIXTURES[fixture_id]}

    motor_id = str(arguments.get("motor_id", ""))
    if name == "get_motor_status":
        values = STATUS_BY_MOTOR.get(motor_id)
        if values is None:
            return {"error": f"找不到設備 {motor_id} 的感測資料"}
        return {"motor_id": motor_id, **values}

    if name == "get_temperature_forecast":
        model_name = str(arguments.get("model_name", "ridge_direct"))
        base = FORECAST_TEMPERATURE.get(motor_id)
        if base is None:
            return {"error": f"找不到設備 {motor_id} 的感測資料"}
        training_motor_id = str(arguments.get("training_motor_id") or motor_id)
        # Stable differences make model and cross-device claims observable.
        history_adjustment = -0.3 if model_name == "ridge_history" else 0.0
        cross_adjustment = 0.2 if training_motor_id != motor_id else 0.0
        predicted = round(base + history_adjustment + cross_adjustment, 1)
        return {
            "motor_id": motor_id,
            "training_motor_id": training_motor_id,
            "predicted_temperature": predicted,
            "forecast_horizon_minutes": 30,
            "model_name": model_name,
            "model_label": MODEL_LABELS.get(model_name, model_name),
            "evaluation": {"mae": 0.72, "rmse": 0.94},
        }

    return {"error": "未知的工具"}


class FixtureTools:
    """DiagnosticTools implementation that records calls and returns fixtures."""

    def __init__(self, fixture_id: str):
        self.fixture_id = fixture_id
        self.calls: list[tuple[str, dict]] = []
        self.results: list[dict] = []

    def execute(self, name: str, arguments: dict) -> dict:
        copied_arguments = dict(arguments)
        self.calls.append((name, copied_arguments))
        result = tool_result(self.fixture_id, name, copied_arguments)
        self.results.append(result)
        return result
