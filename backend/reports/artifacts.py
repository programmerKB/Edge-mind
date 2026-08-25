"""Public chart metadata and path-safe artifact resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote

from reports import config


# The allow-list controls both display order and which generated files may be
# served. CSV and metadata paths can never be exposed through the image route.
CHART_TITLES = {
    "04_baseline_mae.svg": "MAE 基準模型比較",
    "05_anomaly_f1.svg": "F1 異常偵測比較",
    "06_error_metrics.svg": "五特徵模型誤差指標",
    "01_actual_vs_predicted.svg": "實際與預測溫度",
    "02_error_curve.svg": "各觀測點預測誤差",
    "03_error_distribution.svg": "預測誤差分佈",
    "07_system_performance.svg": "Training / Inference 系統效能",
}
ARTIFACT_ROUTE_PREFIX = "/api/report-artifacts/"


def resolve_chart_artifact(relative_path: str) -> Path | None:
    """Resolve an allow-listed SVG while rejecting traversal and other files."""
    requested = Path(relative_path)
    if requested.is_absolute() or ".." in requested.parts:
        return None
    if (
        len(requested.parts) != 5
        or requested.parts[0] != "inference_runs"
        or requested.parts[-2] != "charts"
        or requested.name not in CHART_TITLES
    ):
        return None

    root = config.REPORT_ROOT.resolve()
    candidate = (root / requested).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_chart_attachments(chart_paths: Iterable[str]) -> list[dict[str, str]]:
    """Convert internal filesystem paths into ordered browser-safe metadata."""
    root = config.REPORT_ROOT.resolve()
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
        if resolve_chart_artifact(relative.as_posix()) is None:
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
