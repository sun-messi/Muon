from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one K/method run of the focused experiment."""

    d_input: int = 100
    d_signal: int = 20
    n_m1_features: int = 10
    n_m2_features: int = 10
    d_qk: int = 100
    d_value: int = 100
    d_output: int = 50
    sequence_length: int = 23

    signal_tokens: int = 5
    m1_group_probability: float = 0.8
    signal_scale: float = 1.0
    binary_teacher_density: float = 0.1

    steps: int = 1000
    batch_size: int = 32
    eval_size: int = 5000
    eval_every: int = 1
    learning_rate: float = 0.03
    init_scale: float = 0.1

    method: str = "adaptive_wv_wout"
    momentum: float = 0.95
    rank_tau: float = 5.0
    rank_tail_fraction: float = 0.5
    rank_ema_alpha: float = 0.005
    rank_floor: int = 1
    warmup_steps: int = 0

    seed: int = 0
    device: str = "cuda"

    def validate(self) -> None:
        if self.d_signal != self.n_m1_features + self.n_m2_features:
            raise ValueError("d_signal must equal n_m1_features + n_m2_features")
        if self.d_input <= self.d_signal:
            raise ValueError("d_input must be larger than d_signal")
        if self.sequence_length < self.signal_tokens:
            raise ValueError("sequence_length must be at least signal_tokens")
        if self.signal_tokens > min(self.n_m1_features, self.n_m2_features):
            raise ValueError(
                "signal_tokens must fit in either feature group for only-M1/only-M2 eval"
            )
        if self.method not in {"muon", "adaptive_wv_wout"}:
            raise ValueError("method must be 'muon' or 'adaptive_wv_wout'")
        if not 0.0 <= self.m1_group_probability <= 1.0:
            raise ValueError("m1_group_probability must be in [0, 1]")
        if not 0.0 < self.binary_teacher_density <= 1.0:
            raise ValueError("binary_teacher_density must be in (0, 1]")
        if self.eval_every < 1:
            raise ValueError("eval_every must be positive")
        if not 0.0 < self.rank_tail_fraction <= 1.0:
            raise ValueError("rank_tail_fraction must be in (0, 1]")

    @property
    def noise_tokens(self) -> int:
        return self.sequence_length - self.signal_tokens

    @property
    def adaptive_parameter_names(self) -> tuple[str, ...]:
        return ("WV", "Wout") if self.method == "adaptive_wv_wout" else ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

