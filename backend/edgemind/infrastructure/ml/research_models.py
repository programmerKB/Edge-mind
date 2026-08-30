"""Executable recurrent, convolutional, linear, and patch research adapters.

Imports of NumPy and PyTorch are intentionally lazy.  The production
edge image can therefore keep running the dependency-free baselines, while a
research environment installed from ``requirements-research.txt`` activates
the heavier comparison models automatically.
"""

from __future__ import annotations

from typing import Any, Sequence

from edgemind.domain.research import (
    BASE_FEATURE_NAMES,
    ModelRegistry,
    ResearchConfig,
    ResearchError,
    SequenceExample,
)


def _feature_indexes(feature_names: Sequence[str]) -> tuple[int, ...]:
    indexes = []
    for name in feature_names:
        try:
            indexes.append(BASE_FEATURE_NAMES.index(name))
        except ValueError as error:
            raise ResearchError(f"unknown feature name: {name}") from error
    if not indexes:
        raise ResearchError("at least one feature is required")
    return tuple(indexes)


def _project_history(
    example: SequenceExample,
    indexes: Sequence[int],
) -> list[list[float]]:
    return [
        [float(row[index]) for index in indexes]
        for row in example.history
    ]


class TorchSequenceAdapter:
    """Shared deterministic loop for all PyTorch sequence architectures."""

    representation = "sequence_to_direct_multi_horizon"

    def __init__(
        self,
        config: ResearchConfig,
        feature_names: tuple[str, ...],
        architecture: str,
    ):
        import numpy as np
        import torch

        self.config = config
        self.feature_names = tuple(feature_names)
        self.architecture = architecture
        self.representation = {
            "lstm": "recurrent_sequence_to_direct_multi_horizon",
            "tcn": "causal_convolution_to_direct_multi_horizon",
            "dlinear": "decomposition_linear_to_direct_multi_horizon",
            "patchtst": "channel_independent_patches_to_direct_multi_horizon",
        }.get(architecture, "sequence_to_direct_multi_horizon")
        self._indexes = _feature_indexes(self.feature_names)
        self._np = np
        self._torch = torch
        self._model: Any | None = None
        self._x_mean: Any | None = None
        self._x_scale: Any | None = None
        self._y_mean: Any | None = None
        self._y_scale: Any | None = None
        self._selected_epochs: int | None = None
        self._selection_validation_mae: float | None = None
        self._used_early_stopping = False

    def _network(self):
        torch = self._torch
        nn = torch.nn
        input_size = len(self._indexes)
        output_size = len(self.config.horizons_minutes)
        history_steps = self.config.history_steps
        architecture = self.architecture

        class LSTMNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(input_size, 32, batch_first=True)
                self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, output_size))

            def forward(self, values):
                encoded, _ = self.encoder(values)
                return self.head(encoded[:, -1, :])

        class TemporalConvolutionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.Sequential(
                    nn.ConstantPad1d((2, 0), 0.0),
                    nn.Conv1d(input_size, 32, kernel_size=3, dilation=1),
                    nn.ReLU(),
                    nn.ConstantPad1d((4, 0), 0.0),
                    nn.Conv1d(32, 32, kernel_size=3, dilation=2),
                    nn.ReLU(),
                    nn.ConstantPad1d((8, 0), 0.0),
                    nn.Conv1d(32, 32, kernel_size=3, dilation=4),
                    nn.ReLU(),
                )
                self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, output_size))

            def forward(self, values):
                encoded = self.blocks(values.transpose(1, 2))
                return self.head(encoded[:, :, -1])

        class DLinearNetwork(nn.Module):
            """DLinear-style moving-average decomposition with direct heads."""

            def __init__(self):
                super().__init__()
                self.seasonal = nn.Linear(history_steps, output_size)
                self.trend = nn.Linear(history_steps, output_size)
                self.feature_projection = nn.Linear(input_size, 1)

            def forward(self, values):
                # Replication padding keeps the moving average causal-neutral at
                # both history boundaries without changing sequence length.
                channels = values.transpose(1, 2)
                padded = torch.nn.functional.pad(channels, (1, 1), mode="replicate")
                trend = torch.nn.functional.avg_pool1d(padded, kernel_size=3, stride=1)
                seasonal = channels - trend
                per_feature = self.seasonal(seasonal) + self.trend(trend)
                return self.feature_projection(per_feature.transpose(1, 2)).squeeze(-1)

        class PatchTSTNetwork(nn.Module):
            """Channel-independent patch Transformer adapted to a 12-step input."""

            def __init__(self):
                super().__init__()
                width = 32
                self.patch_length = min(4, history_steps)
                self.patch_stride = max(1, self.patch_length // 2)
                self.patch_count = 1 + (
                    history_steps - self.patch_length
                ) // self.patch_stride
                self.patch_projection = nn.Linear(self.patch_length, width)
                self.position = nn.Parameter(
                    torch.zeros(1, self.patch_count, width)
                )
                layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=4,
                    dim_feedforward=64,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    layer,
                    num_layers=2,
                    enable_nested_tensor=False,
                )
                self.head = nn.Sequential(
                    nn.LayerNorm(input_size * width),
                    nn.Linear(input_size * width, output_size),
                )

            def forward(self, values):
                patches = values.unfold(
                    dimension=1,
                    size=self.patch_length,
                    step=self.patch_stride,
                ).permute(0, 2, 1, 3)
                batch, channels, patch_count, patch_length = patches.shape
                patches = patches.reshape(batch * channels, patch_count, patch_length)
                encoded = self.patch_projection(patches) + self.position
                encoded = self.encoder(encoded).mean(dim=1)
                return self.head(encoded.reshape(batch, channels * encoded.shape[-1]))

        networks = {
            "lstm": LSTMNetwork,
            "tcn": TemporalConvolutionNetwork,
            "dlinear": DLinearNetwork,
            "patchtst": PatchTSTNetwork,
        }
        try:
            return networks[architecture]()
        except KeyError as error:
            raise ResearchError(f"unsupported PyTorch architecture: {architecture}") from error

    def _x(self, examples: Sequence[SequenceExample]):
        return self._np.asarray(
            [_project_history(example, self._indexes) for example in examples],
            dtype=self._np.float32,
        )

    def _arrays(self, examples: Sequence[SequenceExample]):
        x = self._x(examples)
        y = self._np.asarray(
            [
                [float(value) - example.current_temperature for value in example.targets]
                for example in examples
            ],
            dtype=self._np.float32,
        )
        return x, y

    def _fit_core(
        self,
        examples: Sequence[SequenceExample],
        validation: Sequence[SequenceExample],
        epochs: int,
    ) -> tuple[int, float | None]:
        if not examples:
            raise ResearchError(f"{self.architecture.upper()} requires training examples")
        torch = self._torch
        torch.manual_seed(self.config.random_seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        x, y = self._arrays(examples)
        self._x_mean = x.mean(axis=(0, 1), keepdims=True)
        self._x_scale = x.std(axis=(0, 1), keepdims=True)
        self._x_scale[self._x_scale < 1e-6] = 1.0
        self._y_mean = y.mean(axis=0, keepdims=True)
        self._y_scale = y.std(axis=0, keepdims=True)
        self._y_scale[self._y_scale < 1e-6] = 1.0
        x_tensor = torch.from_numpy((x - self._x_mean) / self._x_scale)
        y_tensor = torch.from_numpy((y - self._y_mean) / self._y_scale)
        self._model = self._network().cpu()
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=0.003, weight_decay=1e-4
        )
        # The preregistered primary objective weights every horizon equally and
        # optimizes MAE; RMSE remains a reported secondary evaluation metric.
        loss_function = torch.nn.L1Loss()
        validation_x_tensor = None
        validation_y = None
        if validation:
            validation_x, validation_y = self._arrays(validation)
            validation_x_tensor = torch.from_numpy(
                (validation_x - self._x_mean) / self._x_scale
            )
        best_epoch = epochs
        best_validation_mae = float("inf")
        best_state = None
        stale_epochs = 0
        batch_size = min(self.config.training_batch_size, len(x_tensor))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.random_seed)
        for epoch in range(1, epochs + 1):
            self._model.train()
            order = torch.randperm(len(x_tensor), generator=generator)
            for start in range(0, len(order), batch_size):
                indexes = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(
                    self._model(x_tensor[indexes]), y_tensor[indexes]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            if validation_x_tensor is None:
                continue
            self._model.eval()
            with torch.inference_mode():
                normalized_output = self._model(validation_x_tensor).cpu().numpy()
            residual_output = normalized_output * self._y_scale + self._y_mean
            validation_mae = float(self._np.mean(self._np.abs(residual_output - validation_y)))
            if validation_mae < best_validation_mae - 1e-7:
                best_validation_mae = validation_mae
                best_epoch = epoch
                stale_epochs = 0
                best_state = {
                    name: tensor.detach().clone()
                    for name, tensor in self._model.state_dict().items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= self.config.early_stopping_patience:
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return best_epoch, (
            best_validation_mae if validation_x_tensor is not None else None
        )

    def fit(self, examples: Sequence[SequenceExample]) -> None:
        fixed_epochs = min(80, self.config.max_training_epochs)
        self._selected_epochs, _ = self._fit_core(examples, (), fixed_epochs)

    def fit_with_validation(
        self,
        examples: Sequence[SequenceExample],
        validation: Sequence[SequenceExample],
    ) -> None:
        if not validation:
            self.fit(examples)
            return
        selected, validation_mae = self._fit_core(
            examples,
            validation,
            self.config.max_training_epochs,
        )
        self._selected_epochs = selected
        self._selection_validation_mae = validation_mae
        self._used_early_stopping = True

    def refit_on_development(self, examples: Sequence[SequenceExample]) -> None:
        selected_epochs = self._selected_epochs or min(
            80, self.config.max_training_epochs
        )
        self._fit_core(examples, (), selected_epochs)

    def predict(self, example: SequenceExample) -> tuple[float, ...]:
        if self._model is None:
            raise ResearchError(f"{self.architecture.upper()} adapter has not been fitted")
        x = self._x((example,))
        normalized = (x - self._x_mean) / self._x_scale
        with self._torch.inference_mode():
            output = self._model(self._torch.from_numpy(normalized)).cpu().numpy()
        values = output * self._y_scale + self._y_mean
        return tuple(
            example.current_temperature + float(value) for value in values[0]
        )

    def state_dict(self) -> dict[str, Any]:
        if self._model is None:
            raise ResearchError(f"{self.architecture.upper()} adapter has not been fitted")
        return {
            "architecture": self.architecture,
            "seed": self.config.random_seed,
            "feature_names": list(self.feature_names),
            "horizons_minutes": list(self.config.horizons_minutes),
            "target_transform": "delta_from_current_temperature",
            "selected_epochs": self._selected_epochs,
            "selection_validation_mae": self._selection_validation_mae,
            "normalization": {
                "x_mean": self._x_mean.tolist(),
                "x_scale": self._x_scale.tolist(),
                "y_mean": self._y_mean.tolist(),
                "y_scale": self._y_scale.tolist(),
            },
            "weights": {
                name: tensor.detach().cpu().tolist()
                for name, tensor in self._model.state_dict().items()
            },
        }

    def training_metadata(self) -> dict[str, Any]:
        return {
            "selection_method": (
                "validation_mae_early_stopping"
                if self._used_early_stopping
                else "fixed_preregistered_epochs"
            ),
            "selected_epochs": self._selected_epochs,
            "max_epochs": self.config.max_training_epochs,
            "patience": self.config.early_stopping_patience,
            "batch_size": self.config.training_batch_size,
            "selection_validation_mae": self._selection_validation_mae,
            "target_transform": "delta_from_current_temperature",
        }

    def parameter_count(self) -> int:
        if self._model is None:
            return 0
        return sum(parameter.numel() for parameter in self._model.parameters())


def _torch_factory(architecture: str):
    return lambda config, features: TorchSequenceAdapter(config, features, architecture)


def register_optional_research_models(registry: ModelRegistry) -> ModelRegistry:
    """Replace placeholder entries with lazy, dependency-audited adapters."""
    for name, label in (
        ("dlinear", "DLinear"),
        ("lstm", "LSTM"),
        ("tcn", "TCN"),
        ("patchtst", "PatchTST"),
    ):
        registry.register(
            name,
            _torch_factory(name),
            display_name=label,
            description=(
                f"Deterministic CPU {label} direct multi-horizon sequence model."
            ),
            required_modules=("numpy", "torch"),
            suggested_dependencies=("numpy==2.4.6", "torch==2.13.0+cpu"),
            replace=True,
        )
    return registry


__all__ = [
    "TorchSequenceAdapter",
    "register_optional_research_models",
]
