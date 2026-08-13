"""Gemini orchestration and SSE event generation."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import AsyncIterator

from google import genai
from google.genai import types

from settings import AGENT_SYSTEM_INSTRUCTION, MODEL_ID
from tools import get_motor_status, get_temperature_forecast


_client: genai.Client | None = None


def initialize_agent_client() -> None:
    """Create the SDK client when the application starts serving."""
    global _client
    if _client is None:
        _client = genai.Client()


def _sse_event(status: str, content: str, ensure_ascii: bool) -> str:
    payload = json.dumps(
        {"status": status, "content": content},
        ensure_ascii=ensure_ascii,
    )
    return f"data: {payload}\n\n"


def _config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=AGENT_SYSTEM_INSTRUCTION,
        tools=[get_motor_status, get_temperature_forecast],
        temperature=0.2,
    )


async def _generate(contents: list[types.Content], config):
    initialize_agent_client()
    assert _client is not None
    return await asyncio.to_thread(
        _client.models.generate_content,
        model=MODEL_ID,
        contents=contents,
        config=config,
    )


async def _execute_tool(name: str, arguments: dict) -> str:
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
        response = await _generate(contents, config)

        if not response.function_calls:
            yield _sse_event("success", response.text, ensure_ascii)
            return

        contents.append(response.candidates[0].content)
        tool_response_parts = []
        for tool_call in response.function_calls:
            name = tool_call.name
            arguments = dict(tool_call.args or {})
            yield _sse_event(
                "action",
                f"正在呼叫工具: {name}，參數: {arguments}",
                ensure_ascii,
            )
            tool_result = await _execute_tool(name, arguments)
            yield _sse_event(
                "observation",
                f"資料庫回傳: {tool_result}",
                ensure_ascii,
            )
            tool_response_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": json.loads(tool_result)},
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))
        yield _sse_event(
            "thought",
            "正在統整邊緣感測數據並生成報告...",
            ensure_ascii,
        )
        final_response = await _generate(contents, config)
        yield _sse_event("success", final_response.text, ensure_ascii)
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
