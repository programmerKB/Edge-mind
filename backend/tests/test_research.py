"""Unit tests for the framework-independent forecasting research engine."""

from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace
import unittest

from edgemind.domain.research import (
    BASE_FEATURE_NAMES,
    DEFAULT_ABLATIONS,
    DirectRidgeModel,
    HistoryTrendRidgeModel,
    ModelRegistry,
    PersistenceModel,
    ResearchConfig,
    ResearchError,
    SequenceExample,
    build_default_registry,
    build_inference_sequence,
    build_sequence_dataset,
    chronological_split,
    derive_risk,
    evaluate_model,
    horizon_regression_metrics,
    run_feature_ablations,
    run_research_experiment,
    walk_forward_splits,
)


START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make_readings(
    count: int,
    *,
    motor_id: str = "MOTOR-A",
    start: datetime = START,
    temperature_offset: float = 0.0,
) -> list[SimpleNamespace]:
    """Build deterministic but non-collinear five-minute sensor histories."""
    rows = []
    for index in range(count):
        load_cycle = math.sin(index / 5.0)
        temperature = (
            25.0
            + temperature_offset
            + 0.025 * index
            + 0.35 * load_cycle
            + 0.08 * math.cos(index / 11.0)
        )
        rows.append(
            SimpleNamespace(
                motor_id=motor_id,
                recorded_at=start + timedelta(minutes=5 * index),
                temperature=temperature,
                humidity=52.0 + 1.5 * math.cos(index / 7.0) - 0.01 * index,
                accel_x=0.08 * math.sin(index / 3.0),
                accel_y=0.07 * math.cos(index / 4.0),
                accel_z=1.0 + 0.02 * math.sin(index / 6.0),
            )
        )
    return rows


def make_manual_example(
    index: int,
    config: ResearchConfig,
    *,
    device_id: str = "MOTOR-A",
    base_temperature: float | None = None,
    slope_per_step: float | None = None,
) -> SequenceExample:
    """Build an independent sequence whose target follows its history slope."""
    base = float(20 + index * 0.15 if base_temperature is None else base_temperature)
    slope = float(
        ((index % 9) - 4) * 0.035
        if slope_per_step is None
        else slope_per_step
    )
    anchor = START + timedelta(days=index)
    history_times = tuple(
        anchor
        - timedelta(
            minutes=(config.history_steps - 1 - step) * config.sampling_minutes
        )
        for step in range(config.history_steps)
    )
    history = []
    for step in range(config.history_steps):
        distance = step - config.history_steps + 1
        temperature = base + slope * distance
        history.append(
            (
                temperature,
                48 + (index % 7) * 0.4 + distance * 0.01,
                math.sin(index * 0.3 + step * 0.1),
                math.cos(index * 0.2 + step * 0.15),
                1 + 0.01 * math.sin(index + step),
            )
        )
    target_times = tuple(
        anchor + timedelta(minutes=horizon)
        for horizon in config.horizons_minutes
    )
    targets = tuple(
        base + slope * (horizon // config.sampling_minutes)
        for horizon in config.horizons_minutes
    )
    return SequenceExample(
        device_id=device_id,
        anchor_time=anchor,
        history_times=history_times,
        history=tuple(history),
        target_times=target_times,
        targets=targets,
    )


class ResearchConfigurationTests(unittest.TestCase):
    def test_defaults_describe_the_recommended_contract(self):
        config = ResearchConfig()

        self.assertEqual(config.sampling_minutes, 5)
        self.assertEqual(config.history_steps, 12)
        self.assertEqual(config.history_minutes, 60)
        self.assertEqual(config.horizons_minutes, (5, 10, 15, 20, 25, 30))
        self.assertEqual(config.effective_gap_steps, 6)
        self.assertEqual(config.to_dict()["gap_minutes"], 30)
        self.assertEqual(config.event_merge_gap_minutes, 15)
        self.assertEqual(config.warning_lookback_minutes, 30)
        self.assertEqual(config.bootstrap_repetitions, 10_000)
        self.assertEqual(config.random_seed, 42)
        self.assertEqual(
            config.ridge_alpha_candidates,
            (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
        )
        self.assertEqual(config.max_training_epochs, 160)
        self.assertEqual(config.early_stopping_patience, 16)

    def test_horizons_extend_to_sixty_minutes_without_code_changes(self):
        config = ResearchConfig(horizons_minutes=tuple(range(5, 61, 5)))
        dataset = build_sequence_dataset(make_readings(80), config)

        self.assertEqual(config.effective_gap_steps, 12)
        self.assertEqual(len(dataset.examples), 57)
        self.assertEqual(len(dataset.examples[0].targets), 12)

    def test_rejects_an_unsafe_gap_or_misaligned_horizon(self):
        with self.assertRaises(ResearchError):
            ResearchConfig(horizons_minutes=(5, 10, 60), gap_steps=6)
        with self.assertRaises(ResearchError):
            ResearchConfig(horizons_minutes=(5, 12))
        with self.assertRaisesRegex(ResearchError, "cannot exceed 60"):
            ResearchConfig(horizons_minutes=(5, 65))


class SequenceDatasetTests(unittest.TestCase):
    def test_builds_twelve_by_five_inputs_and_six_exact_targets(self):
        rows = make_readings(36)
        dataset = build_sequence_dataset(rows)
        example = dataset.examples[0]

        self.assertEqual(len(dataset.examples), 19)
        self.assertEqual(len(example.history), 12)
        self.assertTrue(all(len(row) == 5 for row in example.history))
        self.assertEqual(len(example.targets), 6)
        self.assertEqual(example.anchor_time, rows[11].recorded_at)
        self.assertEqual(example.target_times[-1], rows[17].recorded_at)
        self.assertEqual(example.targets[-1], rows[17].temperature)
        self.assertEqual(dataset.quality.dropped_unaligned_anchors, 17)

    def test_reports_and_excludes_bad_rows_without_target_substitution(self):
        rows = make_readings(40)
        rows.pop(20)  # A missing five-minute slot.
        rows.append(SimpleNamespace(**vars(rows[10])))  # Duplicate timestamp.
        rows.extend(
            (
                SimpleNamespace(**{**vars(rows[0]), "recorded_at": "bad-time"}),
                SimpleNamespace(**{**vars(rows[1]), "humidity": None}),
                SimpleNamespace(**{**vars(rows[2]), "temperature": math.inf}),
            )
        )

        dataset = build_sequence_dataset(rows)
        quality = dataset.quality

        self.assertEqual(quality.invalid_timestamp_rows, 1)
        self.assertEqual(quality.incomplete_feature_rows, 1)
        self.assertEqual(quality.nonfinite_feature_rows, 1)
        self.assertEqual(quality.duplicate_timestamp_count, 1)
        self.assertEqual(quality.duplicate_rows_excluded, 2)
        self.assertEqual(quality.irregular_interval_count, 2)
        self.assertEqual(quality.missing_interval_steps, 2)
        self.assertEqual(len(dataset.examples), 2)
        # No example may jump over either unavailable timestamp.
        unavailable = {
            START + timedelta(minutes=5 * 10),
            START + timedelta(minutes=5 * 20),
        }
        self.assertTrue(
            all(
                unavailable.isdisjoint(example.history_times + example.target_times)
                for example in dataset.examples
            )
        )

    def test_sequences_never_cross_device_boundaries(self):
        rows = make_readings(36, motor_id="A") + make_readings(
            36,
            motor_id="B",
            temperature_offset=3.0,
        )
        dataset = build_sequence_dataset(rows)

        self.assertEqual(dataset.device_ids, ("A", "B"))
        self.assertEqual(len(dataset.examples), 38)
        self.assertEqual(
            {example.device_id for example in dataset.examples},
            {"A", "B"},
        )

    def test_device_filter_is_reported(self):
        rows = make_readings(36, motor_id="A") + make_readings(36, motor_id="B")
        dataset = build_sequence_dataset(rows, device_id="B")

        self.assertEqual(dataset.device_ids, ("B",))
        self.assertEqual(dataset.quality.filtered_out_rows, 36)
        self.assertEqual(len(dataset.examples), 19)

    def test_builds_latest_live_window_without_future_truth(self):
        rows = make_readings(36)

        sequence = build_inference_sequence(rows)

        self.assertEqual(sequence.anchor_time, rows[-1].recorded_at)
        self.assertEqual(sequence.history_times[0], rows[-12].recorded_at)
        self.assertEqual(len(sequence.history), 12)
        self.assertEqual(len(sequence.target_times), 6)
        self.assertEqual(
            sequence.target_times[-1],
            rows[-1].recorded_at + timedelta(minutes=30),
        )
        self.assertEqual(sequence.to_dict()["truth_status"], "pending")

    def test_live_window_fails_closed_for_gap_or_duplicate(self):
        missing = make_readings(36)
        missing.pop(-4)
        with self.assertRaisesRegex(ResearchError, "incomplete"):
            build_inference_sequence(missing)

        duplicated = make_readings(36)
        duplicated.append(SimpleNamespace(**vars(duplicated[-1])))
        with self.assertRaisesRegex(ResearchError, "duplicated"):
            build_inference_sequence(duplicated)


class LeakageSafeSplitTests(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig()
        self.dataset = build_sequence_dataset(make_readings(150), self.config)

    def test_train_validation_test_are_chronological_and_purged(self):
        split = chronological_split(self.dataset, self.config)
        audit = split.to_dict(self.config.sampling_minutes)["leakage_audit"]

        self.assertEqual(split.gap_steps, 6)
        self.assertEqual(split.skipped_sequence_count, 12)
        self.assertLess(
            max(time for item in split.train for time in item.target_times),
            split.validation[0].anchor_time,
        )
        self.assertLess(
            max(time for item in split.validation for time in item.target_times),
            split.test[0].anchor_time,
        )
        self.assertTrue(all(audit.values()))
        self.assertEqual(
            len(split.train) + len(split.validation) + len(split.test) + 12,
            len(self.dataset.examples),
        )

    def test_walk_forward_uses_expanding_training_and_a_safe_gap(self):
        folds = walk_forward_splits(self.dataset, self.config, n_splits=3)

        self.assertEqual(len(folds), 3)
        self.assertLess(len(folds[0].train), len(folds[1].train))
        self.assertLess(len(folds[1].train), len(folds[2].train))
        for fold in folds:
            self.assertEqual(fold.gap_steps, 6)
            self.assertLess(
                max(time for item in fold.train for time in item.target_times),
                fold.test[0].anchor_time,
            )
            self.assertTrue(
                fold.to_dict()["leakage_audit"][
                    "train_targets_end_before_test_anchor"
                ]
            )

    def test_split_rejects_a_mixed_device_training_dataset(self):
        mixed = build_sequence_dataset(
            make_readings(40, motor_id="A") + make_readings(40, motor_id="B")
        )
        with self.assertRaises(ResearchError):
            chronological_split(mixed)


class ExecutableBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig(ridge_alpha=1e-6)
        self.examples = tuple(
            make_manual_example(index, self.config) for index in range(80)
        )

    def test_persistence_repeats_the_anchor_temperature(self):
        model = PersistenceModel(self.config)
        model.fit(self.examples[:20])

        prediction = model.predict(self.examples[20])

        self.assertEqual(len(prediction), 6)
        self.assertTrue(
            all(value == self.examples[20].current_temperature for value in prediction)
        )
        self.assertEqual(model.parameter_count(), 0)

    def test_direct_ridge_is_multioutput_and_uses_training_statistics_only(self):
        model = DirectRidgeModel(self.config, ("temperature",))
        model.fit(self.examples[:50])
        state = model.state_dict()
        evaluation = evaluate_model(model, self.examples[50:], self.config)

        expected_training_mean = sum(
            example.current_temperature for example in self.examples[:50]
        ) / 50
        self.assertAlmostEqual(state["means"][0], expected_training_mean)
        self.assertEqual(
            state["target_transform"], "delta_from_current_temperature"
        )
        self.assertEqual(len(state["coefficients"]), 6)
        self.assertEqual(set(evaluation["by_horizon"]), {"5", "10", "15", "20", "25", "30"})
        self.assertTrue(math.isfinite(evaluation["overall"]["mae"]))
        records = evaluation["prediction_records"]
        self.assertEqual(len(records), len(self.examples[50:]) * 6)
        self.assertEqual(
            set(records[0]),
            {
                "sample_id",
                "device_id",
                "origin_time",
                "target_time",
                "horizon_minutes",
                "actual",
                "predicted",
                "error",
            },
        )
        self.assertEqual(len(records[0]["sample_id"]), 64)
        self.assertAlmostEqual(
            records[0]["error"],
            records[0]["predicted"] - records[0]["actual"],
        )
        self.assertEqual(
            {record["sample_id"] for record in records[:6]},
            {records[0]["sample_id"]},
        )

    def test_history_trend_ridge_learns_slope_dependent_targets(self):
        model = HistoryTrendRidgeModel(
            self.config,
            ("temperature",),
        )
        model.fit(self.examples[:60])
        evaluation = evaluate_model(model, self.examples[60:], self.config)

        self.assertEqual(model.representation, "flattened_history_plus_statistics_and_slope")
        self.assertIn("temperature_slope", model.vector_feature_names)
        self.assertLess(evaluation["overall"]["mae"], 0.03)
        self.assertGreater(model.parameter_count(), 6)


class MetricsAndRiskTests(unittest.TestCase):
    def test_horizon_metrics_include_pooled_and_per_horizon_errors(self):
        metrics = horizon_regression_metrics(
            [(1, 2), (3, 4)],
            [(2, 2), (4, 5)],
            (5, 10),
        )

        self.assertAlmostEqual(metrics["overall"]["mae"], 0.75)
        self.assertAlmostEqual(metrics["by_horizon"]["5"]["mae"], 1.0)
        self.assertAlmostEqual(metrics["by_horizon"]["10"]["mae"], 0.5)
        self.assertIn("stddev_c", metrics["target_distribution"])

    def test_risk_derivation_interpolates_threshold_lead_time(self):
        risk = derive_risk(
            (33.5, 34.0, 35.5, 36.0, 35.8, 35.6),
            (5, 10, 15, 20, 25, 30),
            threshold_c=35.0,
            current_temperature=33.0,
        )

        self.assertEqual(risk.max_temperature_c, 36.0)
        self.assertTrue(risk.threshold_crossing)
        self.assertAlmostEqual(risk.time_to_threshold_minutes, 13.3333333)
        self.assertGreater(risk.heating_rate_c_per_minute, 0)
        self.assertGreater(risk.max_adjacent_heating_rate_c_per_minute, 0)
        self.assertEqual(risk.risk_level, "high")

    def test_risk_levels_distinguish_near_threshold_and_stable_cases(self):
        medium = derive_risk(
            (33.0, 33.2, 33.4, 33.6, 33.8, 34.0),
            (5, 10, 15, 20, 25, 30),
            threshold_c=35.0,
            current_temperature=32.8,
        )
        low = derive_risk(
            (29.9, 30.0, 30.0, 30.1, 30.0, 30.0),
            (5, 10, 15, 20, 25, 30),
            threshold_c=35.0,
            current_temperature=30.0,
        )

        self.assertEqual(medium.risk_level, "medium")
        self.assertFalse(medium.threshold_crossing)
        self.assertIsNone(medium.time_to_threshold_minutes)
        self.assertEqual(low.risk_level, "low")

    def test_risk_evaluation_includes_continuous_roc_and_pr_auc(self):
        config = ResearchConfig(threshold_c=35.0)
        examples = (
            make_manual_example(0, config, base_temperature=30.0, slope_per_step=0),
            make_manual_example(1, config, base_temperature=34.0, slope_per_step=0.3),
        )

        class OrderedRiskModel:
            def predict(self, example):
                value = 31.0 if example.anchor_time == examples[0].anchor_time else 37.0
                return (value,) * len(config.horizons_minutes)

        metrics = evaluate_model(OrderedRiskModel(), examples, config)
        crossing = metrics["risk"]["threshold_crossing"]

        self.assertEqual(crossing["roc_auc"], 1.0)
        self.assertEqual(crossing["average_precision"], 1.0)
        self.assertGreater(crossing["pr_auc"], 0.0)
        self.assertEqual(crossing["sample_count"], 2)
        self.assertEqual(crossing["excluded_ongoing_origin_count"], 0)
        self.assertEqual(
            metrics["risk"]["event_evaluation"]["status"],
            "available",
        )

    def test_ongoing_hot_origins_do_not_inflate_early_warning_metrics(self):
        config = ResearchConfig(threshold_c=35.0)
        ongoing = make_manual_example(
            0,
            config,
            base_temperature=40.0,
            slope_per_step=0,
        )

        class CoolingPredictionModel:
            def predict(self, _example):
                return (33.0,) * len(config.horizons_minutes)

        metrics = evaluate_model(CoolingPredictionModel(), (ongoing,), config)
        risk = metrics["risk"]

        self.assertEqual(risk["threshold_crossing"]["sample_count"], 0)
        self.assertEqual(
            risk["threshold_crossing"]["excluded_ongoing_origin_count"],
            1,
        )
        self.assertEqual(risk["ongoing_origin_detection"]["sample_count"], 1)
        self.assertEqual(risk["ongoing_origin_detection"]["FN"], 1)
        self.assertEqual(risk["time_to_threshold"]["sample_count"], 0)


class RegistryAndExperimentTests(unittest.TestCase):
    def test_optional_models_are_explicitly_unavailable_not_fabricated(self):
        registry = build_default_registry()

        self.assertEqual(registry.availability("persistence")["status"], "available")
        for name in (
            "xgboost", "gru", "lstm", "tcn", "dlinear", "transformer", "patchtst",
        ):
            availability = registry.availability(name)
            self.assertEqual(availability["status"], "unavailable")
            self.assertIn("adapter", availability["reason"])
            self.assertNotIn("metrics", availability)

    def test_registry_accepts_an_outward_adapter_factory(self):
        registry = ModelRegistry()
        registry.register(
            "external",
            lambda config, features: PersistenceModel(config, features),
            display_name="External",
            description="Test adapter",
        )

        self.assertEqual(registry.availability("external")["status"], "available")
        with self.assertRaises(ResearchError):
            registry.register(
                "external",
                lambda config, features: PersistenceModel(config, features),
                display_name="Duplicate",
                description="Duplicate",
            )

    def test_experiment_is_json_serializable_and_reports_real_efficiency(self):
        config = ResearchConfig(walk_forward_folds=2)
        training = make_readings(115, motor_id="TRAIN")
        evaluation = make_readings(
            70,
            motor_id="EVAL",
            start=START + timedelta(days=10),
            temperature_offset=1.2,
        )

        result = run_research_experiment(
            training,
            evaluation,
            config=config,
            model_names=(
                "persistence",
                "ridge_direct",
                "ridge_history_trend",
                "xgboost",
                "lstm",
            ),
            include_ablations=False,
        )

        json.dumps(result, allow_nan=False)
        self.assertTrue(all(result["dataset"]["split"]["leakage_audit"].values()))
        walk_forward_scope = result["dataset"]["walk_forward_scope"]
        locked_test_anchor = datetime.fromisoformat(
            walk_forward_scope["locked_test_first_anchor_time"]
        )
        self.assertEqual(walk_forward_scope["scope"], "development_only")
        self.assertTrue(
            walk_forward_scope[
                "all_fold_targets_end_before_locked_test_anchor"
            ]
        )
        for fold in result["dataset"]["walk_forward_splits"]:
            self.assertLess(
                datetime.fromisoformat(fold["train"]["last_target_time"]),
                locked_test_anchor,
            )
            self.assertLess(
                datetime.fromisoformat(fold["test"]["last_target_time"]),
                locked_test_anchor,
            )
        for name in ("persistence", "ridge_direct", "ridge_history_trend"):
            model = result["models"][name]
            self.assertEqual(model["status"], "available")
            self.assertEqual(model["walk_forward"]["fold_count"], 2)
            self.assertGreater(model["efficiency"]["model_size_bytes"], 0)
            self.assertEqual(
                model["cross_device"]["evaluation_device_ids"],
                ["EVAL"],
            )
            latest = model["cross_device"]["latest_forecast"]
            self.assertEqual(len(latest["trajectory"]), 6)
            self.assertIn(latest["risk"]["risk_level"], {"low", "medium", "high"})
            threshold_metrics = model["test"]["risk"]["threshold_crossing"]
            self.assertIn("roc_auc", threshold_metrics)
            self.assertIn("pr_auc", threshold_metrics)
            self.assertEqual(
                threshold_metrics["score_definition"],
                "predicted_max_temperature_c_minus_threshold_c",
            )
            self.assertNotIn("lead_time_minutes", threshold_metrics)
            self.assertEqual(
                threshold_metrics["lead_time_status"],
                "not_computed_without_event_matching",
            )
            self.assertEqual(
                len(model["validation"]["prediction_records"]),
                model["validation"]["overall"]["sample_count"],
            )
            self.assertEqual(
                len(model["test"]["prediction_records"]),
                model["test"]["overall"]["sample_count"],
            )
            self.assertEqual(
                len(model["cross_device"]["prediction_records"]),
                model["cross_device"]["overall"]["sample_count"],
            )
            self.assertNotIn(
                "prediction_records",
                model["walk_forward"]["aggregate"],
            )
            self.assertTrue(
                all(
                    "prediction_records" not in fold
                    for fold in model["walk_forward"]["folds"]
                )
            )
            self.assertIn("skill_score_vs_persistence", model["test"])
            self.assertFalse(model["training"]["locked_test_used_for_selection"])
        split_records = result["dataset"]["split_manifest_records"]
        self.assertEqual(
            len(split_records),
            result["dataset"]["training"]["sequence_count"]
            + result["dataset"]["evaluation"]["sequence_count"],
        )
        self.assertEqual(
            {record["split"] for record in split_records},
            {"train", "validation", "test", "purged_gap", "external_test"},
        )
        self.assertEqual(result["models"]["xgboost"]["status"], "unavailable")
        self.assertEqual(result["models"]["lstm"]["status"], "unavailable")
        comparisons = {
            item["candidate_model"]: item
            for item in result["statistical_comparisons"]
            if item.get("candidate_model")
        }
        self.assertEqual(
            comparisons["ridge_direct"]["status"],
            "insufficient_blocks",
        )
        self.assertEqual(
            comparisons["ridge_history_trend"]["block_definition"],
            "device_id_x_origin_utc_date",
        )

    def test_cross_device_rows_are_never_used_by_fit(self):
        fitted_device_sets: list[set[str]] = []

        class SpyAdapter:
            feature_names = BASE_FEATURE_NAMES
            representation = "spy_persistence"

            def __init__(self, config, _features):
                self.config = config

            def fit(self, examples):
                fitted_device_sets.append({item.device_id for item in examples})

            def predict(self, example):
                return (example.current_temperature,) * len(
                    self.config.horizons_minutes
                )

            def state_dict(self):
                return {"algorithm": "spy"}

        registry = ModelRegistry()
        registry.register(
            "spy",
            lambda config, features: SpyAdapter(config, features),
            display_name="Spy",
            description="Records fit devices",
        )
        result = run_research_experiment(
            make_readings(90, motor_id="TRAIN"),
            make_readings(60, motor_id="EVAL", start=START + timedelta(days=4)),
            config=ResearchConfig(walk_forward_folds=2),
            model_names=("spy",),
            registry=registry,
            include_ablations=False,
        )

        self.assertEqual(result["models"]["spy"]["status"], "available")
        self.assertTrue(fitted_device_sets)
        self.assertTrue(all(devices == {"TRAIN"} for devices in fitted_device_sets))

    def test_default_feature_ablation_matrix_is_executable(self):
        config = ResearchConfig()
        split = chronological_split(build_sequence_dataset(make_readings(100)), config)

        results = run_feature_ablations(split, config)

        self.assertEqual([item["id"] for item in results], ["A", "B", "C", "D", "E"])
        self.assertEqual(len(DEFAULT_ABLATIONS), 5)
        self.assertTrue(all(item["status"] == "available" for item in results))
        self.assertEqual(results[0]["features"], ["temperature"])
        self.assertEqual(results[-1]["model"], "ridge_history_trend")
        self.assertIn("by_horizon", results[-1]["test"])


if __name__ == "__main__":
    unittest.main()
