"""Public API for temperature forecasting.

``engine`` contains deterministic numerical code; ``service`` coordinates
database persistence and report generation. Importing from this package keeps
callers independent from that internal split.
"""

from forecast.engine import (
    FEATURE_NAMES,
    FORECAST_HORIZON_MINUTES,
    MIN_TRAINING_SAMPLES,
    RIDGE_ALPHA,
    TARGET_TOLERANCE_MINUTES,
    ForecastError,
    TrainingExample,
    build_training_examples,
    extract_sensor_features,
    predict_with_payload,
    train_temperature_model,
)


def train_and_save_model(*args, **kwargs):
    """Load persistence code only when database-backed training is requested."""
    from forecast.service import train_and_save_model as implementation

    return implementation(*args, **kwargs)


def forecast_temperature(*args, **kwargs):
    """Load service dependencies lazily so numerical tests stay lightweight."""
    from forecast.service import forecast_temperature as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "FEATURE_NAMES",
    "FORECAST_HORIZON_MINUTES",
    "MIN_TRAINING_SAMPLES",
    "RIDGE_ALPHA",
    "TARGET_TOLERANCE_MINUTES",
    "ForecastError",
    "TrainingExample",
    "build_training_examples",
    "extract_sensor_features",
    "forecast_temperature",
    "predict_with_payload",
    "train_and_save_model",
    "train_temperature_model",
]
