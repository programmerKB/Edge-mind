"""Regression tests for Gemini's manual function-calling boundary."""

import copy
from types import SimpleNamespace
import unittest

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
        )


class _FakeClient:
    def __init__(self):
        self.models = _CapturingModels()

    def close(self):
        return None


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


if __name__ == "__main__":
    unittest.main()
