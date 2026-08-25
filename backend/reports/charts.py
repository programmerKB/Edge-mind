"""Dependency-free SVG chart renderers used by inference reports."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Sequence

from reports.io import is_finite


def svg_shell(title: str, body: str, width: int = 1000, height: int = 600) -> str:
    """Wrap chart elements in a consistent, standalone SVG document."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<style>text{font-family:Arial,"Noto Sans TC",sans-serif;fill:#253047}'
        '.grid{stroke:#dce3ec;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.5}'
        '.label{font-size:13px}.title{font-size:24px;font-weight:700}</style>'
        f'<text x="50" y="40" class="title">{escape(title)}</text>{body}</svg>'
    )


def write_placeholder_chart(path: Path, title: str, message: str) -> None:
    """Write an explanatory empty-state chart when no observations exist."""
    body = (
        '<rect x="80" y="100" width="840" height="400" rx="18" fill="#f1f5f9"/>'
        f'<text x="500" y="305" text-anchor="middle" font-size="20">{escape(message)}</text>'
    )
    path.write_text(svg_shell(title, body), encoding="utf-8")


def write_line_chart(
    path: Path,
    title: str,
    series: Sequence[tuple[str, Sequence[float], str]],
    x_labels: Sequence[str],
    y_label: str,
) -> None:
    """Render one or more time-aligned series as an SVG line chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [value for _, points, _ in series for value in points if is_finite(value)]
    if not values or not x_labels:
        write_placeholder_chart(path, title, "No completed observations yet")
        return
    left, top, width, height = 85, 80, 870, 420
    low, high = min(values), max(values)
    padding = max((high - low) * 0.1, 0.1)
    low, high = low - padding, high + padding

    def x_at(index: int) -> float:
        return left + (width * index / max(len(x_labels) - 1, 1))

    def y_at(value: float) -> float:
        return top + height - (value - low) / (high - low) * height

    parts = []
    for tick in range(6):
        y = top + height * tick / 5
        value = high - (high - low) * tick / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" '
            f'class="label">{value:.2f}</text>'
        )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{top+height}" class="axis"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top+height}" x2="{left+width}" '
        f'y2="{top+height}" class="axis"/>'
    )
    label_indexes = sorted({0, len(x_labels) // 2, len(x_labels) - 1})
    for index in label_indexes:
        parts.append(
            f'<text x="{x_at(index):.1f}" y="{top+height+28}" '
            'text-anchor="middle" class="label">'
            f'{escape(x_labels[index])}</text>'
        )
    parts.append(
        f'<text x="22" y="{top+height/2}" '
        f'transform="rotate(-90 22 {top+height/2})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )
    for series_index, (name, points, color) in enumerate(series):
        coordinates = " ".join(
            f"{x_at(index):.1f},{y_at(float(value)):.1f}"
            for index, value in enumerate(points)
        )
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="3"/>'
        )
        for index, value in enumerate(points):
            parts.append(
                f'<circle cx="{x_at(index):.1f}" '
                f'cy="{y_at(float(value)):.1f}" r="3" fill="{color}"/>'
            )
        legend_x = 650 + series_index * 150
        parts.append(
            f'<line x1="{legend_x}" y1="40" x2="{legend_x+28}" y2="40" '
            f'stroke="{color}" stroke-width="4"/>'
        )
        parts.append(
            f'<text x="{legend_x+35}" y="45" class="label">'
            f'{escape(name)}</text>'
        )
    path.write_text(svg_shell(title, "".join(parts)), encoding="utf-8")


def write_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    y_label: str,
) -> None:
    """Render labeled finite values as an SVG bar chart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        write_placeholder_chart(path, title, "No completed observations yet")
        return
    left, top, width, height = 85, 80, 870, 420
    high = max(max(values) * 1.2, 0.01)
    bar_slot = width / len(values)
    colors = ("#94a3b8", "#60a5fa", "#2563eb", "#f59e0b")
    parts = []
    for tick in range(6):
        y = top + height * tick / 5
        value = high * (5 - tick) / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" '
            f'y2="{y:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-10}" y="{y+5:.1f}" text-anchor="end" '
            f'class="label">{value:.3f}</text>'
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_width = bar_slot * 0.58
        x = left + index * bar_slot + bar_slot * 0.21
        bar_height = value / high * height
        y = top + height - bar_height
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" rx="5" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{y-9:.1f}" '
            f'text-anchor="middle" class="label">{value:.4f}</text>'
        )
        parts.append(
            f'<text x="{x+bar_width/2:.1f}" y="{top+height+28}" '
            f'text-anchor="middle" class="label">{escape(label)}</text>'
        )
    parts.append(
        f'<text x="22" y="{top+height/2}" '
        f'transform="rotate(-90 22 {top+height/2})" '
        f'text-anchor="middle" class="label">{escape(y_label)}</text>'
    )
    path.write_text(svg_shell(title, "".join(parts)), encoding="utf-8")


def write_histogram(path: Path, title: str, errors: Sequence[float]) -> None:
    """Bucket signed errors and delegate rendering to the bar-chart writer."""
    if not errors:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_placeholder_chart(path, title, "No completed observations yet")
        return
    low, high = min(errors), max(errors)
    if math.isclose(low, high):
        low, high = low - 0.05, high + 0.05
    bin_count = min(10, max(4, int(math.sqrt(len(errors)))))
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for error in errors:
        index = min(int((error - low) / width), bin_count - 1)
        counts[index] += 1
    labels = [f"{low + (index + 0.5) * width:.2f}" for index in range(bin_count)]
    write_bar_chart(path, title, labels, counts, "Count")
