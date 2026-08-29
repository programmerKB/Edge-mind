"""Tests for deterministic forecast intent extraction."""

import unittest

from edgemind.application.intent import (
    temperature_forecast_arguments,
    temperature_trajectory_arguments,
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

    def test_routes_multi_horizon_risk_before_legacy_forecast(self):
        self.assertEqual(
            temperature_trajectory_arguments(
                "請使用 DEMO-1 訓練的模型，預測 DEMO-2 未來溫度軌跡與過熱風險"
            ),
            {"motor_id": "DEMO-2", "training_motor_id": "DEMO-1"},
        )

    def test_extracts_requested_sequence_model_without_using_it_as_device(self):
        self.assertEqual(
            temperature_trajectory_arguments(
                "請用 TCN 預測馬達 M1 未來 5 到 30 分鐘的溫度軌跡"
            ),
            {
                "motor_id": "M1",
                "model_name": "tcn",
                "horizons_minutes": [5, 10, 15, 20, 25, 30],
            },
        )

    def test_extracts_all_added_sequence_model_names(self):
        for label, identifier in (
            ("DLinear", "dlinear"),
            ("Transformer", "transformer"),
            ("PatchTST", "patchtst"),
        ):
            with self.subTest(model=label):
                arguments = temperature_trajectory_arguments(
                    f"請用 {label} 預測馬達 M1 未來溫度軌跡"
                )
                self.assertEqual(arguments["motor_id"], "M1")
                self.assertEqual(arguments["model_name"], identifier)

    def test_routes_readme_style_training_device_and_named_model(self):
        self.assertEqual(
            temperature_trajectory_arguments(
                "請使用 DEMO-1 訓練的 Ridge History 模型，"
                "預測 DEMO-2 未來 5 到 30 分鐘的溫度軌跡與過熱風險"
            ),
            {
                "motor_id": "DEMO-2",
                "training_motor_id": "DEMO-1",
                "model_name": "ridge_history_trend",
                "horizons_minutes": [5, 10, 15, 20, 25, 30],
            },
        )

    def test_single_point_request_is_not_a_trajectory_request(self):
        self.assertIsNone(
            temperature_trajectory_arguments("請預測馬達 M1 在 30 分鐘後的溫度")
        )

    def test_routes_custom_threshold_and_sixty_minute_horizons(self):
        self.assertEqual(
            temperature_trajectory_arguments(
                "用 40°C 警戒值預測馬達 M1 未來 5 到 60 分鐘風險"
            ),
            {
                "motor_id": "M1",
                "threshold_c": 40.0,
                "horizons_minutes": list(range(5, 61, 5)),
            },
        )

    def test_preserves_invalid_custom_range_for_fail_closed_validation(self):
        self.assertEqual(
            temperature_trajectory_arguments(
                "以 40°C 警戒值預測馬達 M1 未來 5 到 65 分鐘風險"
            )["horizons_minutes"],
            [5, 65],
        )


if __name__ == "__main__":
    unittest.main()
