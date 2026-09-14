"""Transparent, deterministic metrics for EdgeMind-AgentEval traces."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from statistics import fmean, pstdev
from typing import Iterable


_NUMBER = re.compile(r"(?<![A-Za-z0-9_.-])-?\d+(?:\.\d+)?(?![A-Za-z0-9_.-])")
_ADVICE_TERMS = ("建議", "檢查", "監測", "觀察", "維護", "保養")
_SYSTEM_GROUNDED_NUMBERS = (30.0,)


def _normal_calls(calls: Iterable[dict]) -> list[dict]:
    return [
        {"name": call.get("name"), "arguments": dict(call.get("arguments") or {})}
        for call in calls
    ]


def _contains_fact(answer: str, alternatives: list[str]) -> bool:
    folded = answer.casefold().replace(",", "")
    return any(str(value).casefold().replace(",", "") in folded for value in alternatives)


def _flatten_numbers(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _flatten_numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _flatten_numbers(item)]
    return []


def _is_grounded(number: float, allowed: list[float]) -> bool:
    return any(
        abs(number - candidate) <= max(0.05, abs(candidate) * 0.005)
        for candidate in allowed
    )


def score_run(case: dict, trace: dict) -> dict:
    """Score one run without requiring byte-identical natural-language output."""
    expected_calls = _normal_calls(case["expected_tool_calls"])
    actual_calls = _normal_calls(trace.get("actual_tool_calls", []))
    expected_names = [call["name"] for call in expected_calls]
    actual_names = [call["name"] for call in actual_calls]
    expected_need_tool = bool(expected_calls)
    actual_need_tool = bool(actual_calls)
    answer = str(trace.get("final_answer") or "")

    fact_checks = [
        _contains_fact(answer, alternatives)
        for alternatives in case["required_answer_facts"]
    ]
    key_fact_recall = sum(fact_checks) / len(fact_checks) if fact_checks else 1.0
    forbidden_hits = [
        claim for claim in case["forbidden_claims"] if claim.casefold() in answer.casefold()
    ]

    number_grounding_applicable = expected_need_tool
    result_numbers = _flatten_numbers(trace.get("tool_results", []))
    allowed_numbers = result_numbers + [
        float(number) for number in case.get("allowed_answer_numbers", [])
    ] + list(_SYSTEM_GROUNDED_NUMBERS)
    answer_numbers = (
        [float(match.group()) for match in _NUMBER.finditer(answer)]
        if number_grounding_applicable
        else []
    )
    grounded_numbers = [
        number for number in answer_numbers if _is_grounded(number, allowed_numbers)
    ]
    ungrounded_numbers = [
        number for number in answer_numbers if not _is_grounded(number, allowed_numbers)
    ]
    grounded_claim_precision = (
        (len(grounded_numbers) / len(answer_numbers) if answer_numbers else 1.0)
        if number_grounding_applicable
        else None
    )
    advice_ok = not case["requires_maintenance_advice"] or any(
        term in answer for term in _ADVICE_TERMS
    )
    route_ok = trace.get("actual_route") == case["expected_route"]
    tool_sequence_ok = actual_names == expected_names
    arguments_ok = actual_calls == expected_calls
    no_extra_tools = len(actual_calls) <= len(expected_calls)
    answer_complete = key_fact_recall == 1.0 and advice_ok
    faithful = not forbidden_hits and not ungrounded_numbers
    expected_error = case["expected_result"] == "tool_error"
    runtime_ok = bool(answer) and (
        expected_error or not trace.get("runtime_error")
    )
    task_success = all(
        (
            expected_need_tool == actual_need_tool,
            tool_sequence_ok,
            arguments_ok,
            no_extra_tools,
            answer_complete,
            faithful,
            runtime_ok,
        )
    )
    return {
        "case_id": case["case_id"],
        "run_index": trace.get("run_index", 1),
        "expected_need_tool": expected_need_tool,
        "actual_need_tool": actual_need_tool,
        "route_correct": route_ok,
        "tool_name_correct": tool_sequence_ok if expected_need_tool else None,
        "argument_exact_match": arguments_ok if expected_need_tool else None,
        "tool_sequence_correct": tool_sequence_ok,
        "extra_tool_call": len(actual_calls) > len(expected_calls),
        "key_fact_recall": key_fact_recall,
        "grounded_claim_precision": grounded_claim_precision,
        "hallucination": not faithful,
        "answer_complete": answer_complete,
        "error_handling_correct": task_success if expected_error else None,
        "task_success": task_success,
        "forbidden_claim_hits": forbidden_hits,
        "ungrounded_numbers": ungrounded_numbers,
        "latency_ms": trace.get("latency_ms"),
        "token_usage": trace.get("token_usage") or {},
    }


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean_available(values: Iterable[float | int | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return fmean(available) if available else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def aggregate_scores(cases: list[dict], traces: list[dict]) -> dict:
    """Aggregate per-run and repeated-run reliability metrics."""
    case_by_id = {case["case_id"]: case for case in cases}
    unknown = sorted({trace.get("case_id") for trace in traces} - set(case_by_id))
    if unknown:
        raise ValueError(f"Traces reference unknown cases: {unknown}")
    scored = [score_run(case_by_id[trace["case_id"]], trace) for trace in traces]

    tp = sum(item["expected_need_tool"] and item["actual_need_tool"] for item in scored)
    fp = sum(not item["expected_need_tool"] and item["actual_need_tool"] for item in scored)
    fn = sum(item["expected_need_tool"] and not item["actual_need_tool"] for item in scored)
    tn = sum(not item["expected_need_tool"] and not item["actual_need_tool"] for item in scored)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    expected_tool_scores = [
        item for item in scored if item["expected_need_tool"]
    ]
    error_scores = [
        item for item in scored if item["error_handling_correct"] is not None
    ]

    by_case: dict[str, list[bool]] = defaultdict(list)
    for item in scored:
        by_case[item["case_id"]].append(item["task_success"])
    repetitions = max((len(values) for values in by_case.values()), default=0)
    pass_at_k = {
        f"pass^{k}": _safe_divide(
            sum(len(values) >= k and all(values[:k]) for values in by_case.values()),
            len(by_case),
        )
        for k in (1, 3, 5)
        if repetitions >= k
    }
    per_run_success = []
    for run_index in sorted({item["run_index"] for item in scored}):
        subset = [item for item in scored if item["run_index"] == run_index]
        per_run_success.append(
            _safe_divide(sum(item["task_success"] for item in subset), len(subset))
        )

    latencies = [
        float(item["latency_ms"])
        for item in scored
        if item["latency_ms"] is not None
    ]
    total_tokens = [
        float(item["token_usage"].get("total_tokens", 0)) for item in scored
    ]
    argument_field_totals: Counter[str] = Counter()
    argument_field_matches: Counter[str] = Counter()
    for trace in traces:
        case = case_by_id[trace["case_id"]]
        expected_calls = _normal_calls(case["expected_tool_calls"])
        actual_calls = _normal_calls(trace.get("actual_tool_calls", []))
        for call_index, expected_call in enumerate(expected_calls):
            actual_call = actual_calls[call_index] if call_index < len(actual_calls) else {}
            actual_arguments = actual_call.get("arguments", {})
            for field, expected_value in expected_call["arguments"].items():
                argument_field_totals[field] += 1
                if actual_arguments.get(field) == expected_value:
                    argument_field_matches[field] += 1
    category_metrics = {}
    for category in sorted({case["category"] for case in cases}):
        category_ids = {
            case["case_id"] for case in cases if case["category"] == category
        }
        subset = [item for item in scored if item["case_id"] in category_ids]
        if subset:
            category_metrics[category] = {
                "runs": len(subset),
                "task_success_rate": _safe_divide(
                    sum(item["task_success"] for item in subset), len(subset)
                ),
                "key_fact_recall": _mean_available(
                    item["key_fact_recall"] for item in subset
                ),
                "hallucination_rate": _safe_divide(
                    sum(item["hallucination"] for item in subset), len(subset)
                ),
            }

    return {
        "cases_evaluated": len(by_case),
        "runs_evaluated": len(scored),
        "repetitions": repetitions,
        "need_tool": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "tool_name_accuracy": _mean_available(
            int(item["tool_name_correct"]) for item in expected_tool_scores
        ),
        "argument_exact_match": _mean_available(
            int(item["argument_exact_match"]) for item in expected_tool_scores
        ),
        "argument_field_accuracy": {
            field: _safe_divide(argument_field_matches[field], count)
            for field, count in sorted(argument_field_totals.items())
        },
        "tool_sequence_accuracy": _mean_available(
            int(item["tool_sequence_correct"]) for item in scored
        ),
        "extra_tool_call_rate": _mean_available(
            int(item["extra_tool_call"]) for item in scored
        ),
        "route_accuracy": _mean_available(int(item["route_correct"]) for item in scored),
        "key_fact_recall": _mean_available(item["key_fact_recall"] for item in scored),
        "grounded_claim_precision": _mean_available(
            item["grounded_claim_precision"] for item in scored
        ),
        "hallucination_rate": _mean_available(
            int(item["hallucination"]) for item in scored
        ),
        "task_success_rate": _mean_available(
            int(item["task_success"]) for item in scored
        ),
        "error_handling_accuracy": _mean_available(
            int(item["error_handling_correct"]) for item in error_scores
        ),
        "reliability": {
            **pass_at_k,
            "mean_run_success": _mean_available(per_run_success),
            "run_success_stddev": (
                pstdev(per_run_success) if len(per_run_success) > 1 else 0.0
            ),
        },
        "efficiency": {
            "mean_total_tokens": _mean_available(total_tokens),
            "total_tokens": sum(total_tokens),
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "mean_tool_calls": _mean_available(
                len(trace.get("actual_tool_calls", [])) for trace in traces
            ),
        },
        "by_category": category_metrics,
        "failed_case_ids": sorted(
            {item["case_id"] for item in scored if not item["task_success"]}
        ),
        "scored_runs": scored,
    }
