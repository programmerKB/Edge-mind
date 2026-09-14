"""Command-line interface for dataset audit, component, and live evaluation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from agent_eval.dataset import DEFAULT_CASES_PATH, load_cases, select_cases, validate_cases
from agent_eval.audit import build_service_audit
from agent_eval.scoring import aggregate_scores
from edgemind.application.intent import temperature_forecast_arguments


def _write_json(path: Path | None, payload: dict) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if path is None:
        print(serialized)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


def route_baseline(cases: list[dict]) -> dict:
    """Evaluate only the deterministic forecast-intent parser."""
    rows = []
    for case in cases:
        parsed = temperature_forecast_arguments(case["prompt"])
        actual_route = "deterministic" if parsed is not None else "model"
        expected_call = case["expected_tool_calls"][0] if case["expected_tool_calls"] else None
        expected_arguments = None
        if expected_call and expected_call["name"] == "get_temperature_forecast":
            expected_arguments = {
                key: value
                for key, value in expected_call["arguments"].items()
                if key != "model_name"
            }
        argument_correct = (
            parsed == expected_arguments
            if case["expected_route"] == "deterministic"
            else None
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "route_correct": actual_route == case["expected_route"],
                "parsed_arguments": parsed,
                "argument_correct": argument_correct,
            }
        )

    tp = sum(
        row["expected_route"] == "deterministic"
        and row["actual_route"] == "deterministic"
        for row in rows
    )
    fp = sum(
        row["expected_route"] == "model"
        and row["actual_route"] == "deterministic"
        for row in rows
    )
    fn = sum(
        row["expected_route"] == "deterministic"
        and row["actual_route"] == "model"
        for row in rows
    )
    tn = sum(
        row["expected_route"] == "model"
        and row["actual_route"] == "model"
        for row in rows
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    argument_rows = [row for row in rows if row["argument_correct"] is not None]
    by_category = {}
    for category in sorted({row["category"] for row in rows}):
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "cases": len(subset),
            "route_accuracy": sum(row["route_correct"] for row in subset) / len(subset),
        }
    return {
        "evaluation": "deterministic_forecast_intent_parser",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "route_accuracy": sum(row["route_correct"] for row in rows) / len(rows),
        "deterministic_route": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        },
        "argument_exact_match": (
            sum(bool(row["argument_correct"]) for row in argument_rows)
            / len(argument_rows)
            if argument_rows
            else None
        ),
        "by_category": by_category,
        "failed_case_ids": [
            row["case_id"]
            for row in rows
            if not row["route_correct"] or row["argument_correct"] is False
        ],
        "rows": rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_eval.cli")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate dataset invariants")
    validate.add_argument("--output", type=Path)

    route = subparsers.add_parser("route", help="test deterministic intent routing")
    route.add_argument("--split", choices=["development", "validation", "locked_test"])
    route.add_argument("--output", type=Path)

    score = subparsers.add_parser("score", help="score previously captured traces")
    score.add_argument("traces", type=Path)
    score.add_argument("--split", choices=["development", "validation", "locked_test"])
    score.add_argument("--output", type=Path)

    audit = subparsers.add_parser(
        "audit", help="separate model-service failures from Agent metrics"
    )
    audit.add_argument("traces", type=Path)
    audit.add_argument("--split", required=True, choices=["development", "validation", "locked_test"])
    audit.add_argument("--output", type=Path)

    live = subparsers.add_parser("live", help="run Gemini against controlled fixtures")
    live.add_argument(
        "--split",
        required=True,
        choices=["development", "validation", "locked_test"],
    )
    live.add_argument("--repetitions", type=int, default=1)
    live.add_argument(
        "--max-case-runs",
        type=int,
        default=100,
        help="safety budget; planned case runs above this value are rejected",
    )
    live.add_argument(
        "--allow-high-volume",
        action="store_true",
        help="explicitly allow a run above --max-case-runs",
    )
    live.add_argument("--traces", type=Path, required=True)
    live.add_argument("--report", type=Path, required=True)
    live.add_argument(
        "--unlock-locked-test",
        action="store_true",
        help="explicitly acknowledge consuming the locked test set",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cases = load_cases(args.cases)
    stats = validate_cases(cases)
    if args.command == "validate":
        _write_json(args.output, {"validation": "passed", **stats})
        return 0
    if args.command == "route":
        selected = select_cases(cases, split=args.split)
        _write_json(args.output, route_baseline(selected))
        return 0
    if args.command == "score":
        selected = select_cases(cases, split=args.split)
        selected_ids = {case["case_id"] for case in selected}
        with args.traces.open(encoding="utf-8") as source:
            traces = [json.loads(line) for line in source if line.strip()]
        traces = [trace for trace in traces if trace.get("case_id") in selected_ids]
        _write_json(args.output, aggregate_scores(selected, traces))
        return 0
    if args.command == "audit":
        selected = select_cases(cases, split=args.split)
        selected_ids = {case["case_id"] for case in selected}
        with args.traces.open(encoding="utf-8") as source:
            traces = [json.loads(line) for line in source if line.strip()]
        traces = [trace for trace in traces if trace.get("case_id") in selected_ids]
        _write_json(args.output, build_service_audit(selected, traces))
        return 0
    if args.command == "live":
        if args.repetitions < 1:
            raise ValueError("--repetitions must be at least 1")
        if args.split == "locked_test" and not args.unlock_locked_test:
            print("locked_test requires --unlock-locked-test", file=sys.stderr)
            return 2
        selected = select_cases(cases, split=args.split)
        planned_case_runs = len(selected) * args.repetitions
        if planned_case_runs > args.max_case_runs and not args.allow_high_volume:
            print(
                f"refusing {planned_case_runs} case runs; safety budget is "
                f"{args.max_case_runs}. Use --allow-high-volume only after "
                "checking the Gemini project quota.",
                file=sys.stderr,
            )
            return 2
        from agent_eval.runner import run_suite
        from edgemind.infrastructure.ai.gemini import GeminiModelGateway
        from edgemind.infrastructure.config import settings

        traces = asyncio.run(
            run_suite(GeminiModelGateway(settings), selected, args.repetitions)
        )
        _write_jsonl(args.traces, traces)
        report = aggregate_scores(selected, traces)
        report["run_configuration"] = {
            "model_id": settings.model_id,
            "fallback_model_id": settings.fallback_model_id,
            "split": args.split,
            "repetitions": args.repetitions,
            "controlled_tool_fixtures": True,
        }
        _write_json(args.report, report)
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
