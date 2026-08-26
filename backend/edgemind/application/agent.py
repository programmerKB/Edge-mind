"""Framework-independent Agent orchestration and progress event generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from edgemind.application.intent import temperature_forecast_arguments
from edgemind.application.ports import AgentModelGateway, DiagnosticTools


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-selected tool invocation."""

    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class ModelReply:
    """A direct model answer or a list of requested tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Transport-neutral progress, artifact, success, or error event."""

    status: str
    content: str
    attachments: list[dict[str, str]] = field(default_factory=list)


class AgentService:
    """Coordinate intent routing, diagnostic tools, and grounded summaries."""

    def __init__(
        self,
        model: AgentModelGateway,
        tools: DiagnosticTools,
    ):
        """Inject the model and diagnostic tool ports used by this Agent."""
        self._model = model
        self._tools = tools

    async def stream(self, message: str) -> AsyncIterator[AgentEvent]:
        """Yield one complete Agent interaction without transport encoding."""
        try:
            yield AgentEvent("thought", "Agent 正在分析您的請求...")
            direct_arguments = temperature_forecast_arguments(message)
            if direct_arguments:
                tool_calls = (
                    ToolCall("get_temperature_forecast", direct_arguments),
                )
            else:
                decision = await self._model.decide(message)
                if not decision.tool_calls:
                    yield AgentEvent("success", decision.text)
                    return
                tool_calls = decision.tool_calls

            tool_results: list[dict] = []
            sent_attachment_urls: set[str] = set()
            for call in tool_calls:
                yield AgentEvent(
                    "action",
                    f"正在呼叫工具: {call.name}，參數: {call.arguments}",
                )
                payload = await asyncio.to_thread(
                    self._tools.execute,
                    call.name,
                    call.arguments,
                )
                attachments = [
                    item
                    for item in payload.get("attachments", [])
                    if item["url"] not in sent_attachment_urls
                ]
                sent_attachment_urls.update(item["url"] for item in attachments)
                yield AgentEvent(
                    "observation",
                    _observation_message(call.name, payload),
                )
                if attachments:
                    yield AgentEvent(
                        "artifacts",
                        f"後端已產生 {len(attachments)} 張模型評估圖表。",
                        attachments,
                    )
                tool_results.append({"name": call.name, "result": payload})

            yield AgentEvent(
                "thought",
                "正在統整邊緣感測數據並生成報告...",
            )
            text = await self._model.summarize(message, tool_results)
            yield AgentEvent("success", text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            yield AgentEvent("error", f"Agent 執行失敗：{error}")

    async def close(self) -> None:
        """Release resources owned by the configured model gateway."""
        await self._model.close()


def _observation_message(name: str, payload: dict) -> str:
    """Build a compact progress message from one tool payload."""
    if payload.get("error"):
        return str(payload["error"])
    if name == "get_motor_status":
        return f"已取得 {payload.get('motor_id', '設備')} 的最新感測資料。"
    if name == "get_temperature_forecast":
        return (
            f"已完成 {payload.get('motor_id', '設備')} 的 "
            "30 分鐘溫度預測與模型評估。"
        )
    return "後端資料已取得。"
