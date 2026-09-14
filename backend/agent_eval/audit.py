"""Post-run service-capacity audit kept separate from Agent scoring."""

from __future__ import annotations

from collections import Counter

from agent_eval.scoring import aggregate_scores


MODEL_SERVICE_ERROR_MARKER = "模型服務目前忙碌"


def build_service_audit(cases: list[dict], traces: list[dict]) -> dict:
    """Separate model-service failures from conditionally scorable responses."""
    service_failures = [
        trace
        for trace in traces
        if MODEL_SERVICE_ERROR_MARKER in str(trace.get("runtime_error") or "")
    ]
    eligible = [trace for trace in traces if trace not in service_failures]
    raw = aggregate_scores(cases, traces)
    conditional = aggregate_scores(cases, eligible) if eligible else None
    if conditional is not None:
        # Repetition reliability cannot be inferred after failed attempts have
        # been removed, because cases no longer have the same run indices.
        conditional.pop("reliability", None)
        conditional.pop("scored_runs", None)
    by_run = {}
    for run_index in sorted({trace.get("run_index", 1) for trace in traces}):
        run_traces = [trace for trace in traces if trace.get("run_index", 1) == run_index]
        failed = [trace for trace in run_traces if trace in service_failures]
        by_run[str(run_index)] = {
            "attempts": len(run_traces),
            "model_service_failures": len(failed),
            "model_service_failure_rate": len(failed) / len(run_traces),
        }
    return {
        "interpretation": (
            "Raw metrics include model-service failures as failed tasks. "
            "Conditional metrics exclude only the explicit busy/retry-exhausted "
            "model-service error. pass^k is intentionally omitted from the "
            "conditional section and is not reportable for this run."
        ),
        "attempted_runs": len(traces),
        "model_service_failures": len(service_failures),
        "model_service_failure_rate": (
            len(service_failures) / len(traces) if traces else 0.0
        ),
        "eligible_agent_runs": len(eligible),
        "eligible_case_coverage": len({trace["case_id"] for trace in eligible}),
        "failure_messages": dict(
            Counter(str(trace.get("runtime_error")) for trace in service_failures)
        ),
        "by_run": by_run,
        "raw_metrics": {key: value for key, value in raw.items() if key != "scored_runs"},
        "conditional_agent_metrics": conditional,
    }
