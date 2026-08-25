"""Public reporting API assembled from focused package modules."""

from reports.artifacts import build_chart_attachments, resolve_chart_artifact
from reports.config import (
    ANOMALY_TEMPERATURE_THRESHOLD,
    REPORT_ROOT,
    REPORT_TIMEZONE,
)
from reports.inference import create_inference_report
from reports.io import export_demo_datasets, export_sensor_dataset
from reports.performance import finalize_performance_report

__all__ = [
    "ANOMALY_TEMPERATURE_THRESHOLD",
    "REPORT_ROOT",
    "REPORT_TIMEZONE",
    "build_chart_attachments",
    "create_inference_report",
    "export_demo_datasets",
    "export_sensor_dataset",
    "finalize_performance_report",
    "resolve_chart_artifact",
]
