from __future__ import annotations

import math

import torch
from torch import nn

from .config import ExperimentConfig


class SparseAttentionRegressor(nn.Module):
    """Single-head attention, ReLU value map, sum pooling, linear readout."""

    def __init__(
        self,
        config: ExperimentConfig,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.config = config
        self.WQ = nn.Parameter(
            self._weight(
                config.d_input, config.d_qk, config.d_input, generator, device, dtype
            )
        )
        self.WK = nn.Parameter(
            self._weight(
                config.d_input, config.d_qk, config.d_input, generator, device, dtype
            )
        )
        self.WV = nn.Parameter(
            self._weight(
                config.d_input,
                config.d_value,
                config.d_input,
                generator,
                device,
                dtype,
            )
        )
        self.Wout = nn.Parameter(
            self._weight(
                config.d_output,
                config.d_value,
                config.d_value,
                generator,
                device,
                dtype,
            )
        )

    def _weight(
        self,
        rows: int,
        columns: int,
        fan_in: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return (
            torch.randn(
                rows,
                columns,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            * self.config.init_scale
            / math.sqrt(fan_in)
        )

    def forward(
        self, X: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = X @ self.WQ
        K = X @ self.WK
        logits = Q @ K.transpose(-1, -2) / math.sqrt(self.config.d_qk)
        attention = torch.softmax(logits, dim=-1)
        X_tilde = attention @ X
        values = torch.relu(X_tilde @ self.WV)
        pooled = values.sum(dim=1)
        prediction = pooled @ self.Wout.T
        return prediction, attention, X_tilde

    def named_matrix_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, getattr(self, name)) for name in ("WQ", "WK", "WV", "Wout")]


def l2_regression_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((prediction - target) ** 2).sum(dim=1).mean()

