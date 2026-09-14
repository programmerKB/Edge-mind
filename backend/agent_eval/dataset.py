"""Dataset loading and structural validation for EdgeMind-AgentEval."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CASES_PATH = DATA_DIR / "cases.jsonl"

EXPECTED_CATEGORY_COUNTS = {
    "current_status": 30,
    "same_device_forecast": 30,
    "cross_device_forecast": 30,
    "implicit_forecast": 20,
    "knowledge_no_tool": 20,
    "insufficient_negation_conflict": 20,
    "data_or_system_error": 15,
    "boundary_or_attack": 15,
}
EXPECTED_SPLIT_COUNTS = {
    "development": 60,
    "validation": 30,
    "locked_test": 90,
}
ALLOWED_TOOLS = {"get_motor_status", "get_temperature_forecast"}
ALLOWED_MODELS = {"ridge_direct", "ridge_history"}
ALLOWED_ROUTES = {"deterministic", "model"}


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict]:
    """Load a JSONL benchmark without silently accepting blank records."""
    cases = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return cases


def select_cases(
    cases: Iterable[dict],
    *,
    split: str | None = None,
    category: str | None = None,
) -> list[dict]:
    """Filter cases without changing their stable file order."""
    return [
        case
        for case in cases
        if (split is None or case["split"] == split)
        and (category is None or case["category"] == category)
    ]


def validate_cases(cases: list[dict]) -> dict:
    """Validate schema invariants and return reproducible dataset statistics."""
    errors: list[str] = []
    required_keys = {
        "case_id",
        "split",
        "template_family",
        "category",
        "prompt",
        "inference_model",
        "fixture_id",
        "expected_route",
        "expected_tool_calls",
        "required_answer_fields",
        "required_answer_facts",
        "forbidden_claims",
        "allowed_answer_numbers",
        "expected_result",
        "requires_maintenance_advice",
        "difficulty",
        "tags",
    }
    ids: list[str] = []
    prompts: list[str] = []
    family_splits: dict[str, set[str]] = defaultdict(set)

    for index, case in enumerate(cases, start=1):
        missing = sorted(required_keys - set(case))
        if missing:
            errors.append(f"record {index}: missing keys {missing}")
            continue
        case_id = str(case["case_id"])
        ids.append(case_id)
        prompts.append(str(case["prompt"]).strip().casefold())
        family_splits[str(case["template_family"])].add(str(case["split"]))
        if case["inference_model"] not in ALLOWED_MODELS:
            errors.append(f"{case_id}: unsupported inference_model")
        if case["expected_route"] not in ALLOWED_ROUTES:
            errors.append(f"{case_id}: unsupported expected_route")
        calls = case["expected_tool_calls"]
        if not isinstance(calls, list):
            errors.append(f"{case_id}: expected_tool_calls must be a list")
            continue
        for call in calls:
            if call.get("name") not in ALLOWED_TOOLS:
                errors.append(f"{case_id}: unsupported tool {call.get('name')!r}")
                continue
            arguments = call.get("arguments", {})
            if not arguments.get("motor_id"):
                errors.append(f"{case_id}: tool call has no motor_id")
            if call["name"] == "get_temperature_forecast":
                if arguments.get("model_name") != case["inference_model"]:
                    errors.append(f"{case_id}: forecast model does not match UI model")
        if case["expected_route"] == "deterministic" and (
            len(calls) != 1 or calls[0].get("name") != "get_temperature_forecast"
        ):
            errors.append(f"{case_id}: deterministic route must be one forecast call")

    duplicate_ids = sorted(key for key, value in Counter(ids).items() if value > 1)
    duplicate_prompts = sorted(
        key for key, value in Counter(prompts).items() if value > 1
    )
    leaking_families = sorted(
        family for family, splits in family_splits.items() if len(splits) > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate case_id values: {duplicate_ids}")
    if duplicate_prompts:
        errors.append(f"duplicate prompts: {duplicate_prompts[:5]}")
    if leaking_families:
        errors.append(f"template-family leakage: {leaking_families}")

    category_counts = Counter(case.get("category") for case in cases)
    split_counts = Counter(case.get("split") for case in cases)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        errors.append(
            f"category counts {dict(category_counts)} != {EXPECTED_CATEGORY_COUNTS}"
        )
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        errors.append(f"split counts {dict(split_counts)} != {EXPECTED_SPLIT_COUNTS}")

    forecast_cases = [
        case
        for case in cases
        if case.get("category")
        in {
            "same_device_forecast",
            "cross_device_forecast",
            "implicit_forecast",
        }
    ]
    model_counts = Counter(case["inference_model"] for case in forecast_cases)
    if dict(model_counts) != {"ridge_direct": 40, "ridge_history": 40}:
        errors.append(f"forecast model counts are not balanced: {dict(model_counts)}")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return {
        "total_cases": len(cases),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "forecast_model_counts": dict(model_counts),
        "need_tool_cases": sum(bool(case["expected_tool_calls"]) for case in cases),
        "no_tool_cases": sum(not case["expected_tool_calls"] for case in cases),
        "template_families": len(family_splits),
        "template_family_leaks": 0,
    }
