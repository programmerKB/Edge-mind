"""Framework-independent Agent orchestration and progress event generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import AsyncIterator

from edgemind.application.intent import (
    motor_status_arguments,
    temperature_forecast_arguments,
)
from edgemind.application.ports import (
    AgentModelGateway,
    AgentUsageGateway,
    DiagnosticTools,
)
from edgemind.domain.forecasting import ForecastError
from edgemind.domain.ridge_experiments import DIRECT_MODEL, MODEL_LABELS


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-selected tool invocation."""

    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Model-reported token counts accumulated for one interaction."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    cached_tokens: int = 0
    tool_prompt_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add usage from another successful model response."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thought_tokens=self.thought_tokens + other.thought_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            tool_prompt_tokens=(
                self.tool_prompt_tokens + other.tool_prompt_tokens
            ),
            total_tokens=self.total_tokens + other.total_tokens,
            model_calls=self.model_calls + other.model_calls,
        )


@dataclass(frozen=True, slots=True)
class ModelReply:
    """A direct model answer or a list of requested tool calls."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    token_usage: TokenUsage | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Transport-neutral progress, artifact, success, or error event."""

    status: str
    content: str
    attachments: list[dict[str, str]] = field(default_factory=list)
    token_usage: TokenUsage | None = None


class AgentService:
    """Coordinate intent routing, diagnostic tools, and grounded summaries."""

    def __init__(
        self,
        model: AgentModelGateway,
        tools: DiagnosticTools,
        usage_gateway: AgentUsageGateway | None = None,
    ):
        """Inject the model and diagnostic tool ports used by this Agent."""
        self._model = model
        self._tools = tools
        self._usage_gateway = usage_gateway

    async def stream(
        self,
        message: str,
        inference_model: str = DIRECT_MODEL,
    ) -> AsyncIterator[AgentEvent]:
        """Yield one complete Agent interaction without transport encoding."""
        try:
            if inference_model not in MODEL_LABELS:
                raise ForecastError(f"不支援的推論模型：{inference_model}")
            yield AgentEvent("thought", "Agent 正在分析您的請求...")
            direct_status_arguments = motor_status_arguments(message)
            direct_forecast_arguments = temperature_forecast_arguments(message)
            token_usage = None
            if direct_status_arguments:
                tool_calls = (
                    ToolCall("get_motor_status", direct_status_arguments),
                )
            elif direct_forecast_arguments:
                tool_calls = (
                    ToolCall(
                        "get_temperature_forecast",
                        direct_forecast_arguments,
                    ),
                )
            else:
                decision = await self._model.decide(message)
                if not decision.tool_calls:
                    yield AgentEvent(
                        "success",
                        decision.text,
                        token_usage=decision.token_usage,
                    )
                    return
                tool_calls = decision.tool_calls
                token_usage = decision.token_usage

            # The UI selection is authoritative. Never let a language-model
            # generated argument silently change the requested Ridge variant.
            tool_calls = tuple(
                ToolCall(
                    call.name,
                    {
                        **call.arguments,
                        **(
                            {"model_name": inference_model}
                            if call.name == "get_temperature_forecast"
                            else {}
                        ),
                    },
                )
                for call in tool_calls
            )

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
                        f"後端已產生 {len(attachments)} 張模型評估圖表。"
                        + (
                            f"\n{payload['evaluation_note']}"
                            if payload.get("evaluation_note") else ""
                        ),
                        attachments,
                    )
                tool_results.append({"name": call.name, "result": payload})

            yield AgentEvent(
                "thought",
                "正在統整邊緣感測數據並撰寫診斷說明...",
            )
            try:
                summary = await self._model.summarize(message, tool_results)
                summary_text = summary.text
                if summary.token_usage is not None:
                    token_usage = (
                        summary.token_usage
                        if token_usage is None
                        else token_usage + summary.token_usage
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Model summary failed; returning grounded local summary",
                    exc_info=True,
                )
                summary_text = _grounded_fallback_summary(tool_results)
            if token_usage is not None and self._usage_gateway is not None:
                await asyncio.to_thread(
                    self._usage_gateway.record_token_usage,
                    tool_results,
                    token_usage,
                )
            yield AgentEvent(
                "success",
                summary_text,
                token_usage=token_usage,
            )
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
            f"已使用 {payload.get('model_label', 'Ridge')} 完成 "
            f"{payload.get('motor_id', '設備')} 的 30 分鐘溫度預測與模型評估。"
        )
    return "後端資料已取得。"


def _display_value(value: object) -> str:
    """Format a tool value without inventing precision or units."""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _append_value(
    lines: list[str],
    payload: dict,
    key: str,
    label: str,
    unit: str = "",
) -> None:
    """Append only fields actually present in a tool observation."""
    value = payload.get(key)
    if value is not None:
        lines.append(f"- {label}：{_display_value(value)}{unit}")


def _status_fallback(payload: dict) -> str:
    """Render one status observation without model-generated diagnoses."""
    motor_id = _display_value(payload.get("motor_id", "設備"))
    lines = [f"### {motor_id} 即時狀態"]
    _append_value(lines, payload, "temperature", "溫度", "°C")
    _append_value(lines, payload, "humidity", "濕度", "%")
    _append_value(lines, payload, "accel_x", "X 軸加速度")
    _append_value(lines, payload, "accel_y", "Y 軸加速度")
    _append_value(lines, payload, "accel_z", "Z 軸加速度")
    _append_value(lines, payload, "vibration", "震動量")
    _append_value(lines, payload, "status", "工具判定狀態")
    _append_value(lines, payload, "recorded_at", "資料時間")
    status = str(payload.get("status", "")).strip().lower()
    if status in {"warning", "critical", "anomaly", "abnormal"}:
        lines.extend(
            [
                "",
                "工具狀態不是 normal，建議依現場 SOP 人工檢查並持續監測。",
                "工具未提供故障原因，因此目前不判定特定故障。",
            ]
        )
    elif status:
        lines.extend(["", "建議依既有巡檢週期持續監測。"])
    return "\n".join(lines)


def _forecast_fallback(payload: dict) -> str:
    """Render one forecast observation using only returned fields."""
    motor_id = _display_value(payload.get("motor_id", "設備"))
    lines = [f"### {motor_id} 溫度預測"]
    _append_value(lines, payload, "predicted_temperature", "預測溫度", "°C")
    _append_value(lines, payload, "current_temperature", "目前溫度", "°C")
    _append_value(lines, payload, "predicted_change", "預測變化", "°C")
    _append_value(lines, payload, "forecast_horizon_minutes", "預測時間", " 分鐘")
    _append_value(lines, payload, "model_label", "推論模型")
    if payload.get("model_label") is None:
        _append_value(lines, payload, "model_name", "推論模型")
    _append_value(lines, payload, "training_motor_id", "訓練設備")
    evaluation = payload.get("evaluation")
    if isinstance(evaluation, dict):
        _append_value(lines, evaluation, "mae", "MAE")
        _append_value(lines, evaluation, "rmse", "RMSE")
        _append_value(lines, evaluation, "max_error", "Max Error")
    note = payload.get("evaluation_note")
    if note:
        lines.extend(["", _display_value(note)])
    lines.extend(
        [
            "",
            "建議持續監測實際溫度，並依設備規格或現場警戒值判讀。",
            "工具未提供警戒門檻或故障原因，因此目前不自行判定是否過熱或特定故障。",
        ]
    )
    return "\n".join(lines)


def _grounded_fallback_summary(tool_results: list[dict]) -> str:
    """Return useful tool data when the optional model summary is unavailable."""
    sections = [
        "模型摘要服務暫時無法使用；以下直接整理已完成的後端工具結果。"
    ]
    for item in tool_results:
        name = item.get("name")
        payload = item.get("result")
        if not isinstance(payload, dict):
            sections.append("工具未回傳可讀取的資料。")
        elif payload.get("error"):
            sections.append(f"### 工具回報\n{payload['error']}")
        elif name == "get_motor_status":
            sections.append(_status_fallback(payload))
        elif name == "get_temperature_forecast":
            sections.append(_forecast_fallback(payload))
        else:
            sections.append("工具已完成，但沒有可顯示的標準欄位。")
    return "\n\n".join(sections)
