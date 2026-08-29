"""Availability behavior for research-only heavy model adapters."""

import importlib.util
from datetime import datetime, timedelta, timezone
import math
from types import SimpleNamespace
import unittest

from edgemind.domain.research import (
    BASE_FEATURE_NAMES,
    ResearchConfig,
    build_default_registry,
    build_sequence_dataset,
)
from edgemind.infrastructure.ml import register_optional_research_models
from edgemind.infrastructure.ml.research_models import TorchSequenceAdapter


class OptionalResearchModelTests(unittest.TestCase):
    def test_adapters_are_registered_without_importing_heavy_dependencies(self):
        registry = register_optional_research_models(build_default_registry())
        catalogue = {item["name"]: item for item in registry.describe()}

        self.assertEqual(
            set(catalogue),
            {
                "ridge_direct", "ridge_history_trend", "dlinear",
                "lstm", "tcn", "patchtst",
            },
        )
        for name in ("dlinear", "lstm", "tcn", "patchtst"):
            self.assertIn(catalogue[name]["status"], {"available", "unavailable"})
            self.assertTrue(catalogue[name]["suggested_dependencies"])
            if catalogue[name]["status"] == "unavailable":
                self.assertIn("missing optional modules", catalogue[name]["reason"])

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "NumPy and PyTorch research dependencies are not installed",
    )
    def test_every_torch_architecture_has_the_multi_horizon_shape(self):
        import torch

        config = ResearchConfig()
        inputs = torch.zeros((2, config.history_steps, len(BASE_FEATURE_NAMES)))
        for architecture in ("dlinear", "lstm", "tcn", "patchtst"):
            with self.subTest(architecture=architecture):
                adapter = TorchSequenceAdapter(config, BASE_FEATURE_NAMES, architecture)
                outputs = adapter._network()(inputs)
                self.assertEqual(
                    tuple(outputs.shape),
                    (2, len(config.horizons_minutes)),
                )

    @unittest.skipUnless(
        importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"),
        "Complete research dependencies are not installed",
    )
    def test_every_optional_adapter_can_fit_and_predict(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [
            SimpleNamespace(
                motor_id="SMOKE",
                recorded_at=start + timedelta(minutes=5 * index),
                temperature=25 + 0.04 * index + 0.3 * math.sin(index / 4),
                humidity=55 - 0.02 * index + math.cos(index / 6),
                accel_x=0.08 * math.sin(index / 3),
                accel_y=0.07 * math.cos(index / 4),
                accel_z=1 + 0.02 * math.sin(index / 5),
            )
            for index in range(72)
        ]
        config = ResearchConfig(horizons_minutes=(5, 10), walk_forward_folds=1)
        examples = build_sequence_dataset(rows, config).examples
        registry = register_optional_research_models(build_default_registry())

        for name in ("dlinear", "lstm", "tcn", "patchtst"):
            with self.subTest(model=name):
                registration = registry.get(name)
                model = registration.factory(config, BASE_FEATURE_NAMES)
                model.fit(examples[:24])
                prediction = model.predict(examples[24])
                self.assertEqual(len(prediction), len(config.horizons_minutes))
                self.assertTrue(all(math.isfinite(value) for value in prediction))


if __name__ == "__main__":
    unittest.main()
