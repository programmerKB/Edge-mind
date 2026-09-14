"""Execute real Gemini Agent runs against deterministic benchmark fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
import time

from agent_eval.fixtures import FixtureTools
from edgemind.application.agent import AgentService, ModelReply


class RecordingModel:
    """Thin model wrapper that reveals routing without changing model behavior."""

    def __init__(self, model):
        self.model = model
        self.decide_calls = 0

    async def decide(self, message: str) -> ModelReply:
        self.decide_calls += 1
        return await self.model.decide(message)

    async def summarize(self, message: str, tool_results: list[dict]) -> ModelReply:
        return await self.model.summarize(message, tool_results)

    async def close(self) -> None:
        # The shared gateway is closed by the suite runner.
        return None


async def run_case(model, case: dict, run_index: int) -> dict:
    """Run one case and serialize its observable trajectory."""
    recording_model = RecordingModel(model)
    tools = FixtureTools(case["fixture_id"])
    agent = AgentService(recording_model, tools)
    events = []
    started = time.perf_counter()
    async for event in agent.stream(case["prompt"], case["inference_model"]):
        events.append(event)
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    final_event = events[-1] if events else None
    final_answer = final_event.content if final_event is not None else ""
    token_usage = (
        asdict(final_event.token_usage)
        if final_event is not None and final_event.token_usage is not None
        else None
    )
    return {
        "case_id": case["case_id"],
        "run_index": run_index,
        "actual_route": "model" if recording_model.decide_calls else "deterministic",
        "actual_tool_calls": [
            {"name": name, "arguments": arguments} for name, arguments in tools.calls
        ],
        "tool_results": tools.results,
        "final_answer": final_answer,
        "runtime_error": (
            final_answer if final_event is not None and final_event.status == "error" else None
        ),
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "event_statuses": [event.status for event in events],
    }


async def run_suite(model, cases: list[dict], repetitions: int) -> list[dict]:
    """Run cases sequentially to keep API-rate behavior reproducible."""
    traces = []
    try:
        for run_index in range(1, repetitions + 1):
            for case in cases:
                traces.append(await run_case(model, case, run_index))
                await asyncio.sleep(0)
    finally:
        await model.close()
    return traces
