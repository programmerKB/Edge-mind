"""Framework-independent Agent orchestration and progress event generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator

from edgemind.application.intent import (
    temperature_forecast_arguments,
    temperature_trajectory_arguments,
)
from edgemind.application.ports import AgentModelGateway, DiagnosticTools
from edgemind.domain.research import DEFAULT_MODEL_NAMES


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

    async def stream(
        self,
        message: str,
        *,
        model_name: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield one complete Agent interaction without transport encoding."""
        try:
            selected_model = _validated_model_name(model_name)
            yield AgentEvent("thought", "Agent 正在分析您的請求...")
            trajectory_arguments = temperature_trajectory_arguments(message)
            direct_arguments = temperature_forecast_arguments(message)
            if trajectory_arguments:
                if selected_model:
                    trajectory_arguments["model_name"] = selected_model
                tool_calls = (
                    ToolCall(
                        "get_temperature_trajectory_forecast",
                        trajectory_arguments,
                    ),
                )
            elif direct_arguments:
                if selected_model:
                    tool_calls = (
                        ToolCall(
                            "get_temperature_trajectory_forecast",
                            {
                                **direct_arguments,
                                "model_name": selected_model,
                                "horizons_minutes": [30],
                            },
                        ),
                    )
                else:
                    tool_calls = (
                        ToolCall("get_temperature_forecast", direct_arguments),
                    )
            else:
                decision = await self._model.decide(message)
                if not decision.tool_calls:
                    yield AgentEvent("success", decision.text)
                    return
                tool_calls = _apply_selected_model(
                    decision.tool_calls,
                    selected_model,
                )

            tool_results: list[dict] = []
            sent_attachment_urls: set[str] = set()
            completed_attachments: list[dict[str, str]] = []
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
                completed_attachments.extend(attachments)
                yield AgentEvent(
                    "observation",
                    _observation_message(call.name, payload),
                )
                tool_results.append({"name": call.name, "result": payload})

            yield AgentEvent(
                "thought",
                "正在統整邊緣感測數據並生成報告...",
            )
            if (
                len(tool_results) == 1
                and tool_results[0]["name"]
                == "get_temperature_trajectory_forecast"
                and not tool_results[0]["result"].get("error")
            ):
                text = _trajectory_forecast_report(tool_results[0]["result"])
            else:
                text = await self._model.summarize(message, tool_results)
            yield AgentEvent("success", text, completed_attachments)
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
    if name == "get_temperature_trajectory_forecast":
        model = payload.get("model") or {}
        model_label = model.get("display_name") or model.get("name")
        model_suffix = f"（模型：{model_label}）" if model_label else ""
        horizons = [
            point.get("horizon_minutes")
            for point in payload.get("trajectory", [])
            if isinstance(point, dict) and point.get("horizon_minutes") is not None
        ]
        horizon_label = (
            f"未來 {horizons[0]} 分鐘"
            if len(horizons) == 1
            else f"未來 {min(horizons)} 至 {max(horizons)} 分鐘"
            if horizons
            else "未來 5 至 30 分鐘"
        )
        return (
            f"已完成 {payload.get('motor_id', '設備')} 的歷史鎖定測試、"
            f"全資料重訓、{horizon_label}溫度軌跡、過熱風險與圖表"
            f"{model_suffix}。"
        )
    return "後端資料已取得。"


def _display_number(value, digits: int = 3) -> str:
    """Format optional numeric report values without leaking Python nulls."""
    if not isinstance(value, (int, float)):
        return "無法計算"
    return f"{float(value):.{digits}f}"


def _trajectory_forecast_report(payload: dict) -> str:
    """Build one stable completed report from the audited forecast payload."""
    model = payload.get("model") or {}
    source = payload.get("source") or {}
    risk = payload.get("risk") or {}
    historical = payload.get("historical_evaluation") or {}
    locked_test = historical.get("locked_test") or {}
    metrics = locked_test.get("overall") or {}
    split = historical.get("split") or {}
    test_sample_count = (split.get("test") or {}).get("sample_count")
    model_label = model.get("display_name") or model.get("name") or "預測模型"
    risk_level = {
        "low": "低",
        "medium": "中",
        "high": "高",
    }.get(risk.get("risk_level"), "未知")
    trajectory_rows = "\n".join(
        "| +{horizon} 分鐘 | {temperature} °C |".format(
            horizon=point.get("horizon_minutes", "-"),
            temperature=_display_number(
                point.get("predicted_temperature_c")
            ),
        )
        for point in payload.get("trajectory", [])
        if isinstance(point, dict)
    )
    if not trajectory_rows:
        trajectory_rows = "| - | 無可用預測 |"

    if risk_level == "高":
        recommendation = "建議立即檢查負載、散熱與潤滑狀態，並安排人工複核。"
    elif risk_level == "中":
        recommendation = "建議提高監測頻率，檢查散熱與負載是否持續升高。"
    else:
        recommendation = "目前可維持運行，並依既定週期持續監測溫度與振動。"

    provenance = payload.get("dataset_provenance") or {}
    evidence_note = (
        "本次使用 Demo／合成資料，結果只代表流程驗證，不代表真實設備效能。"
        if provenance.get("synthetic_or_demo_detected")
        else "資料來源尚未連結不可變稽核清單，正式決策前仍需完成來源稽核。"
    )
    return (
        "## 完整預測流程已完成\n\n"
        f"設備 **{payload.get('motor_id', '設備')}** 已依序完成歷史時間切分、"
        f"鎖定測試集評估、全歷史資料重訓、即時推論、風險分析與圖表產生。"
        f"本次模型為 **{model_label}**。\n\n"
        "### 即時預測\n\n"
        f"目前溫度：**{_display_number(source.get('current_temperature_c'))} °C**\n\n"
        "| 預測時距 | 預測溫度 |\n"
        "|---|---:|\n"
        f"{trajectory_rows}\n\n"
        "### 歷史鎖定測試結果\n\n"
        f"歷史測試樣本：**{test_sample_count if test_sample_count is not None else '無法計算'}**；"
        f"MAE：**{_display_number(metrics.get('mae'))} °C**；"
        f"RMSE：**{_display_number(metrics.get('rmse'))} °C**；"
        f"R²：**{_display_number(metrics.get('r2_score'))}**；"
        f"MAPE：**{_display_number(metrics.get('mape_percent'))}%**。"
        "這些誤差來自已有真實對應值的歷史鎖定測試集，不是用即時未來資料估算。\n\n"
        "### 風險與建議\n\n"
        f"過熱風險：**{risk_level}**；預測最高溫："
        f"**{_display_number(risk.get('max_temperature_c'))} °C**；警戒門檻："
        f"**{_display_number(risk.get('threshold_c'))} °C**。{recommendation}\n\n"
        "> 即時預測的目標時間尚未到達，因此該筆未來真值仍須於資料產生後進行事後驗證；"
        "這不影響上述已完成的歷史模型誤差評估。\n\n"
        f"> {evidence_note}"
    )


def _validated_model_name(model_name: str | None) -> str | None:
    """Normalize and validate an optional UI-selected forecast model."""
    if model_name is None:
        return None
    normalized = str(model_name).strip().lower()
    if normalized not in DEFAULT_MODEL_NAMES:
        raise ValueError(
            f"不支援的預測模型：{model_name}；可用模型："
            f"{', '.join(DEFAULT_MODEL_NAMES)}"
        )
    return normalized


def _apply_selected_model(
    tool_calls: tuple[ToolCall, ...],
    model_name: str | None,
) -> tuple[ToolCall, ...]:
    """Make the explicit UI selection authoritative for every forecast call."""
    if model_name is None:
        return tool_calls
    selected_calls: list[ToolCall] = []
    for call in tool_calls:
        arguments = dict(call.arguments)
        if call.name == "get_temperature_forecast":
            selected_calls.append(
                ToolCall(
                    "get_temperature_trajectory_forecast",
                    {
                        **arguments,
                        "model_name": model_name,
                        "horizons_minutes": [30],
                    },
                )
            )
        elif call.name == "get_temperature_trajectory_forecast":
            selected_calls.append(
                ToolCall(
                    call.name,
                    {**arguments, "model_name": model_name},
                )
            )
        else:
            selected_calls.append(call)
    return tuple(selected_calls)
