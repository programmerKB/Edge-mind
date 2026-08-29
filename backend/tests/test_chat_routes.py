"""Chat transport tests for explicit diagnostic forecast-model selection."""

import unittest

from pydantic import ValidationError

from edgemind.application.agent import AgentEvent
from edgemind.presentation.api.routes.chat import create_router
from edgemind.presentation.schemas import ChatRequest


class FakeAgent:
    def __init__(self):
        self.calls = []

    async def stream(self, message, *, model_name=None):
        self.calls.append((message, model_name))
        yield AgentEvent("success", "ok")


class ChatRouteTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def endpoint(router, path):
        return next(route.endpoint for route in router.routes if route.path == path)

    async def test_utf8_route_forwards_selected_model_to_agent(self):
        agent = FakeAgent()
        endpoint = self.endpoint(create_router(agent), "/api/chat_utf8")

        response = await endpoint(
            ChatRequest(message="預測馬達 M1 未來溫度", model_name="tcn")
        )
        chunks = [chunk async for chunk in response.body_iterator]

        self.assertEqual(agent.calls, [("預測馬達 M1 未來溫度", "tcn")])
        self.assertIn('"status": "success"', "".join(chunks))

    def test_chat_contract_rejects_removed_models_and_extra_fields(self):
        with self.assertRaises(ValidationError):
            ChatRequest(message="預測", model_name="xgboost")
        with self.assertRaises(ValidationError):
            ChatRequest(message="預測", model_name="lstm", unexpected=True)


if __name__ == "__main__":
    unittest.main()
