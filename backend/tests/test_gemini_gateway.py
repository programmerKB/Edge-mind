"""Regression tests for Gemini's manual function-calling boundary."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from edgemind.infrastructure.ai.gemini import GeminiModelGateway


class _CapturingModels:
    def __init__(self):
        self.config = None

    def generate_content(self, *, model, contents, config):
        """Mirror the SDK deep copy that rejected bound service methods."""
        self.config = copy.deepcopy(config)
        return SimpleNamespace(
            text="",
            function_calls=[
                SimpleNamespace(
                    name="get_motor_status",
                    args={"motor_id": "M2"},
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=120,
                candidates_token_count=8,
                thoughts_token_count=4,
                cached_content_token_count=20,
                tool_use_prompt_token_count=3,
                total_token_count=135,
            ),
        )


class _FakeClient:
    def __init__(self):
        self.models = _CapturingModels()

    def close(self):
        return None


class _ModelServiceError(RuntimeError):
    def __init__(self, code):
        super().__init__(f"model error {code}")
        self.code = code


class _SequencedModels:
    def __init__(self, results):
        self.results = iter(results)
        self.call_count = 0
        self.model_ids = []

    def generate_content(self, *, model, contents, config):
        self.call_count += 1
        self.model_ids.append(model)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class GeminiGatewayTests(unittest.IsolatedAsyncioTestCase):
    """Keep SDK configuration data-only and tool execution application-owned."""

    async def test_decide_uses_deepcopy_safe_manual_tool_declarations(self):
        settings = SimpleNamespace(
            model_id="gemini-test",
            agent_response_timeout_seconds=1.0,
        )
        gateway = GeminiModelGateway(settings)
        client = _FakeClient()
        gateway._client = client

        reply = await gateway.decide("請查詢馬達 M2 的健康狀態")

        self.assertEqual(reply.tool_calls[0].name, "get_motor_status")
        self.assertEqual(reply.tool_calls[0].arguments, {"motor_id": "M2"})
        self.assertEqual(reply.token_usage.prompt_tokens, 120)
        self.assertEqual(reply.token_usage.output_tokens, 8)
        self.assertEqual(reply.token_usage.thought_tokens, 4)
        self.assertEqual(reply.token_usage.cached_tokens, 20)
        self.assertEqual(reply.token_usage.tool_prompt_tokens, 3)
        self.assertEqual(reply.token_usage.total_tokens, 135)
        self.assertEqual(reply.token_usage.model_calls, 1)
        config = client.models.config
        self.assertTrue(config.automatic_function_calling.disable)
        declarations = config.tools[0].function_declarations
        self.assertEqual(
            [declaration.name for declaration in declarations],
            ["get_motor_status", "get_temperature_forecast"],
        )
        self.assertTrue(
            all(not callable(declaration) for declaration in declarations)
        )
        forecast_schema = declarations[1].parameters_json_schema
        self.assertEqual(
            forecast_schema["properties"]["model_name"]["enum"],
            ["ridge_direct", "ridge_history"],
        )

    async def test_transient_model_error_is_retried(self):
        settings = SimpleNamespace(
            model_id="gemini-test",
            agent_response_timeout_seconds=1.0,
        )
        response = SimpleNamespace(text="服務已恢復", function_calls=[])
        models = _SequencedModels([_ModelServiceError(503), response])
        gateway = GeminiModelGateway(settings)
        gateway._client = SimpleNamespace(models=models)

        with patch(
            "edgemind.infrastructure.ai.gemini.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep:
            reply = await gateway.decide("請檢查設備")

        self.assertEqual(reply.text, "服務已恢復")
        self.assertEqual(models.call_count, 2)
        sleep.assert_awaited_once_with(0.5)

    async def test_exhausted_transient_errors_return_friendly_message(self):
        settings = SimpleNamespace(
            model_id="gemini-test",
            agent_response_timeout_seconds=1.0,
        )
        models = _SequencedModels([_ModelServiceError(503) for _ in range(3)])
        gateway = GeminiModelGateway(settings)
        gateway._client = SimpleNamespace(models=models)

        with patch(
            "edgemind.infrastructure.ai.gemini.asyncio.sleep",
            new_callable=AsyncMock,
        ), self.assertRaisesRegex(RuntimeError, "所有可用模型皆已自動重試"):
            await gateway.decide("請檢查設備")

        self.assertEqual(models.call_count, 3)

    async def test_non_transient_model_error_is_not_retried(self):
        settings = SimpleNamespace(
            model_id="gemini-test",
            agent_response_timeout_seconds=1.0,
        )
        models = _SequencedModels([_ModelServiceError(400)])
        gateway = GeminiModelGateway(settings)
        gateway._client = SimpleNamespace(models=models)

        with patch(
            "edgemind.infrastructure.ai.gemini.asyncio.sleep",
            new_callable=AsyncMock,
        ) as sleep, self.assertRaisesRegex(_ModelServiceError, "400"):
            await gateway.decide("請檢查設備")

        self.assertEqual(models.call_count, 1)
        sleep.assert_not_awaited()

    async def test_fallback_model_is_used_after_primary_stays_busy(self):
        settings = SimpleNamespace(
            model_id="gemini-primary",
            fallback_model_id="gemini-fallback",
            agent_response_timeout_seconds=1.0,
        )
        response = SimpleNamespace(text="備援服務正常", function_calls=[])
        models = _SequencedModels(
            [_ModelServiceError(503) for _ in range(3)] + [response]
        )
        gateway = GeminiModelGateway(settings)
        gateway._client = SimpleNamespace(models=models)

        with patch(
            "edgemind.infrastructure.ai.gemini.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            reply = await gateway.decide("請檢查設備")

        self.assertEqual(reply.text, "備援服務正常")
        self.assertEqual(
            models.model_ids,
            [
                "gemini-primary",
                "gemini-primary",
                "gemini-primary",
                "gemini-fallback",
            ],
        )

    async def test_summarize_returns_text_and_reported_token_usage(self):
        settings = SimpleNamespace(
            model_id="gemini-test",
            agent_response_timeout_seconds=1.0,
        )
        response = SimpleNamespace(
            text="設備目前正常。",
            usage_metadata=SimpleNamespace(
                prompt_token_count=200,
                candidates_token_count=12,
                thoughts_token_count=None,
                cached_content_token_count=None,
                tool_use_prompt_token_count=None,
                total_token_count=212,
            ),
        )
        gateway = GeminiModelGateway(settings)
        gateway._client = SimpleNamespace(models=_SequencedModels([response]))

        reply = await gateway.summarize(
            "設備狀態如何？",
            [{"name": "get_motor_status", "result": {"status": "normal"}}],
        )

        self.assertEqual(reply.text, "設備目前正常。")
        self.assertEqual(reply.token_usage.prompt_tokens, 200)
        self.assertEqual(reply.token_usage.output_tokens, 12)
        self.assertEqual(reply.token_usage.total_tokens, 212)


if __name__ == "__main__":
    unittest.main()
