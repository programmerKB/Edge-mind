"""Facade implementing application report ports over filesystem modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from edgemind.domain.entities import SensorReading
from edgemind.infrastructure.reporting.artifacts import (
    build_chart_attachments,
    resolve_chart_artifact,
)
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.inference import create_inference_report
from edgemind.infrastructure.reporting.io import export_demo_datasets
from edgemind.infrastructure.reporting.performance import (
    finalize_performance_report,
)
from edgemind.infrastructure.reporting.ridge_experiments import (
    get_ridge_experiment,
    save_ridge_experiment,
)
from edgemind.infrastructure.reporting.ridge_inference import create_ridge_inference_report


class FilesystemReportGateway:
    """Expose cohesive report behavior through explicit application ports."""

    def __init__(self, context: ReportContext):
        """Bind one immutable report root, timezone, and threshold context."""
        self._context = context

    def create_inference_report(self, **kwargs) -> dict:
        """Generate one inference report with the injected runtime context."""
        return create_inference_report(context=self._context, **kwargs)

    def create_ridge_inference_report(self, **kwargs) -> dict:
        """Generate live Ridge reports with the same artifact contract."""
        return create_ridge_inference_report(context=self._context, **kwargs)

    def finalize_forecast_result(self, result: dict) -> dict:
        """Attach aggregate performance metadata and browser-safe chart URLs."""
        aggregate = finalize_performance_report(
            result["artifacts"]["system_csv"]
        )
        result["performance_summary"] = {
            "run_count": aggregate["run_count"],
            **aggregate["files"],
        }
        result["attachments"] = build_chart_attachments(
            result["artifacts"].get("charts", []),
            self._context.root,
        )
        return result

    def performance_summary(self) -> dict:
        """Read aggregate inference performance without exposing file layout."""
        path = (
            self._context.root
            / "performance"
            / "performance_summary.json"
        )
        if not path.is_file():
            return {"run_count": 0, "metrics": {}, "files": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve_chart_artifact(self, relative_path: str) -> Path | None:
        """Resolve one allow-listed chart below the configured report root."""
        return resolve_chart_artifact(relative_path, self._context.root)

    def export_demo_datasets(
        self,
        training_records: Sequence[SensorReading],
        inference_records: Sequence[SensorReading],
    ) -> dict[str, str]:
        """Write source demo datasets into their categorized directories."""
        return export_demo_datasets(
            training_records,
            inference_records,
            self._context.root,
        )

    def save_ridge_experiment(self, result: dict) -> dict:
        """Persist one dual-Ridge comparison in a dated result directory."""
        return save_ridge_experiment(self._context, result)

    def get_ridge_experiment(self, experiment_id: str) -> dict | None:
        """Read one prior dual-Ridge comparison by its stable identifier."""
        return get_ridge_experiment(self._context, experiment_id)
