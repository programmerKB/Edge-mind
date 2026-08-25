"""Prepare completed forecasts for REST and Agent consumers."""

from reports.artifacts import build_chart_attachments
from reports.performance import finalize_performance_report


def finalize_forecast_result(result: dict) -> dict:
    """Attach aggregate performance metadata and browser-safe chart URLs.

    Both the REST router and Gemini tool use the same response enrichment so
    their attachment contracts cannot drift apart.
    """
    aggregate = finalize_performance_report(result["artifacts"]["system_csv"])
    result["performance_summary"] = {
        "run_count": aggregate["run_count"],
        **aggregate["files"],
    }
    result["attachments"] = build_chart_attachments(
        result["artifacts"].get("charts", [])
    )
    return result
