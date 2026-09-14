"""Tests for deterministic forecast intent extraction."""

import unittest

from edgemind.application.intent import (
    motor_status_arguments,
    temperature_forecast_arguments,
)


class IntentRoutingTests(unittest.TestCase):
    """Verify forecast prompts are routed without false positives."""

    def test_routes_cross_device_forecast(self):
        self.assertEqual(
            temperature_forecast_arguments(
                "請使用 DEMO-1 訓練的模型，推論設備 DEMO-2 在 30 分鐘後的溫度"
            ),
            {"motor_id": "DEMO-2", "training_motor_id": "DEMO-1"},
        )

    def test_routes_same_device_forecast(self):
        self.assertEqual(
            temperature_forecast_arguments("請預測馬達 M1 在 30分鐘後的溫度"),
            {"motor_id": "M1"},
        )

    def test_does_not_route_current_status_query(self):
        self.assertIsNone(
            temperature_forecast_arguments("請查詢馬達 M1 現在的溫度與狀態")
        )

    def test_routes_explicit_current_status_without_model(self):
        self.assertEqual(
            motor_status_arguments("請查詢馬達 M1 現在的溫度與狀態"),
            {"motor_id": "M1"},
        )

    def test_routes_negated_forecast_to_current_status(self):
        self.assertEqual(
            motor_status_arguments("不要預測 M1，只查現在溫度"),
            {"motor_id": "M1"},
        )

    def test_does_not_route_forecast_to_current_status(self):
        self.assertIsNone(
            motor_status_arguments("預測馬達 M1 在 30 分鐘後的溫度")
        )


if __name__ == "__main__":
    unittest.main()
