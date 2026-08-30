"""Public chart metadata and path-safe artifact resolution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

# The allow-list controls both display order and which generated files may be
# served. CSV and metadata paths can never be exposed through the image route.
INFERENCE_CHART_TITLES = {
    "04_baseline_mae.svg": "MAE 基準模型比較",
    "05_anomaly_f1.svg": "F1 異常偵測比較",
    "06_error_metrics.svg": "五特徵模型誤差指標",
    "01_actual_vs_predicted.svg": "實際與預測溫度",
    "02_error_curve.svg": "各觀測點預測誤差",
    "03_error_distribution.svg": "預測誤差分佈",
    "07_system_performance.svg": "Training / Inference 系統效能",
}
RESEARCH_FORECAST_CHART_TITLES = {
    "01_temperature_trajectory.svg": "預測溫度軌跡",
    "02_historical_actual_vs_predicted.svg": "歷史回測實際值與預測值",
    "03_historical_mae_by_horizon.svg": "歷史回測各時距 MAE",
}
CHART_TITLES = {
    **INFERENCE_CHART_TITLES,
    **RESEARCH_FORECAST_CHART_TITLES,
}
ARTIFACT_ROUTE_PREFIX = "/api/report-artifacts/"
OPAQUE_ID_PATTERN = re.compile(r"[a-f0-9]{32}")


def resolve_chart_artifact(
    relative_path: str,
    report_root: Path,
) -> Path | None:
    """Resolve an allow-listed SVG while rejecting traversal and other files."""
    requested = Path(relative_path)
    if requested.is_absolute() or ".." in requested.parts:
        return None
    is_inference_chart = (
        len(requested.parts) == 5
        and requested.parts[0] == "inference_runs"
        and requested.parts[-2] == "charts"
        and requested.name in INFERENCE_CHART_TITLES
    )
    is_research_forecast_chart = (
        len(requested.parts) == 4
        and requested.parts[0] == "research_forecasts"
        and OPAQUE_ID_PATTERN.fullmatch(requested.parts[1]) is not None
        and requested.parts[2] == "charts"
        and requested.name in RESEARCH_FORECAST_CHART_TITLES
    )
    if not (is_inference_chart or is_research_forecast_chart):
        return None

    root = report_root.resolve()
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_chart_attachments(
    chart_paths: Iterable[str],
    report_root: Path,
) -> list[dict[str, str]]:
    """Convert internal filesystem paths into ordered browser-safe metadata."""
    root = report_root.resolve()
    available = {Path(path).name: Path(path) for path in chart_paths}
    attachments = []
    for filename, title in CHART_TITLES.items():
        source = available.get(filename)
        if source is None:
            continue
        try:
            relative = source.resolve().relative_to(root)
        except ValueError:
            continue
        if resolve_chart_artifact(relative.as_posix(), report_root) is None:
            continue
        attachments.append(
            {
                "title": title,
                "alt": f"{title}圖表",
                "url": ARTIFACT_ROUTE_PREFIX
                + quote(relative.as_posix(), safe="/"),
                "media_type": "image/svg+xml",
            }
        )
    return attachments
