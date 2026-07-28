from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig


def rank_from_singular_values(
    singular_values: torch.Tensor,
    tau: float,
    tail_fraction: float,
    rank_floor: int,
) -> int:
    if singular_values.numel() <= 1:
        return 1
    tail_start = min(
        singular_values.numel() - 1,
        max(0, int(singular_values.numel() * (1.0 - tail_fraction))),
    )
    noise_floor = torch.median(singular_values[tail_start:])
    threshold = tau * (
        noise_floor + singular_values.max() * 1e-30 + 1e-30
    )
    rank = int((singular_values > threshold).sum().item())
    return max(rank_floor, min(singular_values.numel(), rank))


def hard_topk_muon_update(
    matrix: torch.Tensor,
    rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    U, singular_values, Vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    update = hard_topk_from_svd(U, Vh, matrix.dtype, rank)
    return update, singular_values


def hard_topk_from_svd(
    U: torch.Tensor,
    Vh: torch.Tensor,
    dtype: torch.dtype,
    rank: int | None = None,
) -> torch.Tensor:
    max_rank = min(U.shape[1], Vh.shape[0])
    selected_rank = max_rank if rank is None else int(rank)
    selected_rank = max(0, min(selected_rank, max_rank))
    if selected_rank == 0:
        return torch.zeros(
            U.shape[0], Vh.shape[1], device=U.device, dtype=dtype
        )
    return (U[:, :selected_rank] @ Vh[:selected_rank]).to(dtype)


@dataclass
class StepDiagnostics:
    selected_ranks: dict[str, int]
    raw_ranks: dict[str, int]


class AdaptiveMuon:
    """Full Muon or hard-top-k adaptive Muon using momentum-buffer spectra."""

    def __init__(
        self,
        named_parameters: list[tuple[str, torch.nn.Parameter]],
        config: ExperimentConfig,
    ) -> None:
        self.named_parameters = named_parameters
        self.config = config
        self.buffers = {
            name: torch.zeros_like(parameter)
            for name, parameter in named_parameters
        }
        self.max_ranks = {
            name: min(parameter.shape) for name, parameter in named_parameters
        }
        self.rank_ema = {
            name: float(self.max_ranks[name]) for name, _ in named_parameters
        }

    @torch.no_grad()
    def step(
        self,
        gradients: dict[str, torch.Tensor],
        step_index: int,
    ) -> StepDiagnostics:
        selected_ranks: dict[str, int] = {}
        raw_ranks: dict[str, int] = {}
        adaptive_names = set(self.config.adaptive_parameter_names)

        for name, parameter in self.named_parameters:
            gradient = gradients[name]
            buffer = self.buffers[name]
            buffer.mul_(self.config.momentum).add_(gradient)
            full_rank = self.max_ranks[name]

            is_adaptive = (
                self.config.method == "adaptive_wv_wout"
                and name in adaptive_names
                and step_index > self.config.warmup_steps
            )
            U, singular_values, Vh = torch.linalg.svd(
                buffer.float(), full_matrices=False
            )
            if is_adaptive:
                raw_rank = rank_from_singular_values(
                    singular_values,
                    self.config.rank_tau,
                    self.config.rank_tail_fraction,
                    min(self.config.rank_floor, full_rank),
                )
                self.rank_ema[name] = (
                    self.config.rank_ema_alpha * raw_rank
                    + (1.0 - self.config.rank_ema_alpha) * self.rank_ema[name]
                )
                selected_rank = round(self.rank_ema[name])
                selected_rank = max(
                    min(self.config.rank_floor, full_rank),
                    min(full_rank, selected_rank),
                )
            else:
                raw_rank = full_rank
                selected_rank = full_rank
                self.rank_ema[name] = float(full_rank)

            update = hard_topk_from_svd(
                U, Vh, buffer.dtype, selected_rank
            )
            parameter.add_(update, alpha=-self.config.learning_rate)
            raw_ranks[name] = raw_rank
            selected_ranks[name] = selected_rank

        return StepDiagnostics(selected_ranks, raw_ranks)
