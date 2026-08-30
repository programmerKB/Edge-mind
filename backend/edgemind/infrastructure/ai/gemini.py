"""Google Gen AI implementation of the application Agent model port."""

from __future__ import annotations

import asyncio
import inspect
import json

from google import genai
from google.genai import types

from edgemind.application.agent import ModelReply, ToolCall
from edgemind.application.ports import DiagnosticTools
from edgemind.infrastructure.config import AGENT_SYSTEM_INSTRUCTION, Settings


class GeminiModelGateway:
    """Translate application model requests to Google Gen AI SDK calls."""

    def __init__(self, settings: Settings, tools: DiagnosticTools):
        """Bind deployment settings and callable diagnostic tool schemas."""
        self._settings = settings
        self._tools = tools
        self._client: genai.Client | None = None

    def initialize(self) -> None:
        """Create the SDK client once when the application starts."""
        if self._client is None:
            self._client = genai.Client()

    async def _generate(self, contents: list[types.Content], config):
        """Run the blocking SDK outside the event loop with a timeout."""
        self.initialize()
        assert self._client is not None
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self._settings.model_id,
                    contents=contents,
                    config=config,
                ),
                timeout=self._settings.agent_response_timeout_seconds,
            )
        except TimeoutError as error:
            seconds = f"{self._settings.agent_response_timeout_seconds:g}"
            raise RuntimeError(f"模型服務超過 {seconds} 秒未回應") from error

    async def decide(self, message: str) -> ModelReply:
        """Ask Gemini for either a direct response or explicit tool calls."""
        response = await self._generate(
            [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=message)],
                )
            ],
            types.GenerateContentConfig(
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
                tools=[
                    self._tools.get_motor_status,
                    self._tools.get_temperature_forecast,
                    self._tools.get_temperature_trajectory_forecast,
                ],
                temperature=0.2,
            ),
        )
        calls = tuple(
            ToolCall(call.name, dict(call.args or {}))
            for call in (response.function_calls or ())
        )
        return ModelReply(text=response.text or "", tool_calls=calls)

    async def summarize(self, message: str, tool_results: list[dict]) -> str:
        """Generate a grounded Traditional-Chinese answer from tool results."""
        prompt = (
            f"原始問題：{message}\n\n"
            "以下是後端工具取得的真實資料。請只根據這些資料，以繁體中文"
            "直接回答原始問題並提供具體建議；不要聲稱使用未列出的資料。"
            "若即時預測的 truth_status 為 pending，必須說明這只是未來目標"
            "尚未到達；模型誤差應引用 historical_evaluation.locked_test，"
            "不得誤稱整個模型沒有誤差資料。\n"
            + json.dumps(tool_results, ensure_ascii=False)
        )
        response = await self._generate(
            [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            types.GenerateContentConfig(
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        if not response.text:
            raise RuntimeError("模型服務未產生文字摘要")
        return response.text

    async def close(self) -> None:
        """Release transports owned by the SDK during shutdown."""
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result
        self._client = None
