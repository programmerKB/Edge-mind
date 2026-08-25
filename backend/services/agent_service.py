"""Gemini orchestration and SSE event generation."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import AsyncIterator

from google import genai
from google.genai import types

from reports.artifacts import build_chart_attachments
from services.agent_tools import get_motor_status, get_temperature_forecast
from services.intent_routing import temperature_forecast_arguments
from services.sse import sse_event as _sse_event
from core.config import AGENT_SYSTEM_INSTRUCTION, settings


_client: genai.Client | None = None


def initialize_agent_client() -> None:
    """Create the SDK client when the application starts serving."""
    global _client
    if _client is None:
        _client = genai.Client()


def _config() -> types.GenerateContentConfig:
    """Build the tool-enabled configuration used for intent resolution."""
    return types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_INSTRUCTION,
        tools=[get_motor_status, get_temperature_forecast],
        temperature=0.2,
    )


def _summary_config() -> types.GenerateContentConfig:
    """Build a tool-free configuration for the final grounded explanation."""
    return types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_INSTRUCTION,
        temperature=0.2,
    )


async def _generate(contents: list[types.Content], config):
    """Call the blocking SDK in a worker thread with a bounded wait time."""
    initialize_agent_client()
    assert _client is not None
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _client.models.generate_content,
                model=settings.model_id,
                contents=contents,
                config=config,
            ),
            timeout=settings.agent_response_timeout_seconds,
        )
    except TimeoutError as error:
        seconds = f"{settings.agent_response_timeout_seconds:g}"
        raise RuntimeError(f"模型服務超過 {seconds} 秒未回應") from error


async def _execute_tool(name: str, arguments: dict) -> str:
    """Dispatch an allow-listed synchronous database tool off the event loop."""
    if name == "get_motor_status":
        return await asyncio.to_thread(
            get_motor_status,
            motor_id=arguments.get("motor_id", ""),
        )
    if name == "get_temperature_forecast":
        return await asyncio.to_thread(
            get_temperature_forecast,
            motor_id=arguments.get("motor_id", ""),
            training_motor_id=arguments.get("training_motor_id"),
        )
    return json.dumps({"error": "未知的工具"}, ensure_ascii=False)


def _observation_message(name: str, payload: dict) -> str:
    """Keep progress events compact while the full payload goes to Gemini."""
    if payload.get("error"):
        return str(payload["error"])
    if name == "get_motor_status":
        motor_id = payload.get("motor_id", "設備")
        return f"已取得 {motor_id} 的最新感測資料。"
    if name == "get_temperature_forecast":
        motor_id = payload.get("motor_id", "設備")
        return f"已完成 {motor_id} 的 30 分鐘溫度預測與模型評估。"
    return "後端資料已取得。"


async def stream_agent_response(
    message: str,
    *,
    ensure_ascii: bool,
) -> AsyncIterator[str]:
    """Yield one agent interaction as server-sent events."""
    try:
        yield _sse_event("thought", "Agent 正在分析您的請求...", ensure_ascii)
        config = _config()
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)],
            )
        ]
        direct_arguments = temperature_forecast_arguments(message)
        if direct_arguments:
            tool_calls = [("get_temperature_forecast", direct_arguments)]
        else:
            response = await _generate(contents, config)
            if not response.function_calls:
                yield _sse_event("success", response.text, ensure_ascii)
                return
            tool_calls = [
                (tool_call.name, dict(tool_call.args or {}))
                for tool_call in response.function_calls
            ]

        tool_results = []
        sent_attachment_urls: set[str] = set()
        for name, arguments in tool_calls:
            yield _sse_event(
                "action",
                f"正在呼叫工具: {name}，參數: {arguments}",
                ensure_ascii,
            )
            tool_result = await _execute_tool(name, arguments)
            tool_payload = json.loads(tool_result)
            tool_attachments = tool_payload.get("attachments") or (
                build_chart_attachments(
                    tool_payload.get("artifacts", {}).get("charts", [])
                )
            )
            tool_attachments = [
                item
                for item in tool_attachments
                if item["url"] not in sent_attachment_urls
            ]
            sent_attachment_urls.update(
                item["url"] for item in tool_attachments
            )
            yield _sse_event(
                "observation",
                _observation_message(name, tool_payload),
                ensure_ascii,
            )
            if tool_attachments:
                yield _sse_event(
                    "artifacts",
                    f"後端已產生 {len(tool_attachments)} 張模型評估圖表。",
                    ensure_ascii,
                    attachments=tool_attachments,
                )
            tool_results.append({"name": name, "result": tool_payload})

        yield _sse_event(
            "thought",
            "正在統整邊緣感測數據並生成報告...",
            ensure_ascii,
        )
        summary_prompt = (
            f"原始問題：{message}\n\n"
            "以下是後端工具取得的真實資料。請只根據這些資料，以繁體中文"
            "直接回答原始問題並提供具體建議；不要聲稱使用未列出的資料。\n"
            + json.dumps(tool_results, ensure_ascii=False)
        )
        final_response = await _generate(
            [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=summary_prompt)],
                )
            ],
            _summary_config(),
        )
        yield _sse_event(
            "success",
            final_response.text,
            ensure_ascii,
        )
    except asyncio.CancelledError:
        # Client disconnected or explicitly stopped the response.
        raise
    except Exception as error:
        yield _sse_event("error", f"Agent 執行失敗：{error}", ensure_ascii)


async def close_agent_client() -> None:
    """Release transports owned by the SDK during application shutdown."""
    global _client
    if _client is None:
        return
    close = getattr(_client, "close", None)
    if close is None:
        _client = None
        return
    result = close()
    if inspect.isawaitable(result):
        await result
    _client = None
