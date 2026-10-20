"""Online Mode-SNR controller for matched Muon/Pion spectral transforms.

The controller observes the incremental matrix gradients from the two updates
immediately preceding a decision boundary.  It never changes the optimizer
shell: it only supplies a per-parameter :class:`spectral.Segment` override.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch

from spectral import Segment, coefficient_sequence


FORMAT = "adaptive_pion_controller_v1"
OPERATOR_ROLES = ("qkv", "attention_output", "mlp_expand", "mlp_contract")


@dataclass(frozen=True)
class Candidate:
    name: str
    backend: str
    promotion_steps: int
    suppression_steps: int
    alpha: float

    @property
    def segment(self) -> Segment:
        return Segment(0, self.backend, self.promotion_steps, self.suppression_steps, self.alpha)


DEFAULT_CANDIDATES = (
    Candidate("muon5ns", "muon5ns", 0, 0, 0.0),
    Candidate("6p1s_a1", "pion", 6, 1, 1.0),
    Candidate("5p4s_a075", "pion", 5, 4, 0.75),
    Candidate("3p2s_a045", "pion", 3, 2, 0.45),
    Candidate("2p4s_a1", "pion", 2, 4, 1.0),
)


def canonical_sha256(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parameter_role(name: str) -> str:
    if name.endswith(".attention.query_key_value.weight"):
        return "qkv"
    if name.endswith(".attention.dense.weight"):
        return "attention_output"
    if name.endswith(".mlp.dense_h_to_4h.weight"):
        return "mlp_expand"
    if name.endswith(".mlp.dense_4h_to_h.weight"):
        return "mlp_contract"
    raise ValueError(f"unsupported adaptive matrix parameter: {name!r}")


def scalar_polynomial_response(
    singular: torch.Tensor, candidate: Candidate, epsilon: float = 1e-7
) -> torch.Tensor:
    """Return the absolute singular response produced by a candidate.

    ``apply_polynomial`` first divides the whole matrix by its Frobenius norm.
    Polynomial iterations preserve singular vectors, so applying the scalar
    recurrence to normalized singular values is exact up to SVD roundoff.
    """

    values = singular.float() / (singular.float().norm() + epsilon)
    for a, b, c in coefficient_sequence(
        candidate.backend,
        candidate.promotion_steps,
        candidate.suppression_steps,
        candidate.alpha,
    ):
        square = values.square()
        values = a * values + b * square * values + c * square.square() * values
    return values.abs()


def mode_snr(projections: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    if projections.ndim != 2 or projections.shape[0] < 2:
        raise ValueError("Mode-SNR requires at least two projection groups")
    mean = projections.mean(0)
    std = projections.std(0, unbiased=True)
    return mean.abs() / (std + epsilon)


def soft_reliability(snr: torch.Tensor, tau: float) -> torch.Tensor:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    square = snr.square()
    return square / (square + tau * tau)


def response_rank(response: torch.Tensor, threshold: float = 0.5) -> int:
    top = response.max().clamp_min(1e-30)
    return int((response / top > threshold).sum())


def energy_rank(singular: torch.Tensor, fraction: float = 0.95) -> int:
    energy = singular.float().square()
    if float(energy.sum()) == 0.0:
        return 0
    cumulative = energy.cumsum(0) / energy.sum()
    return int(torch.searchsorted(cumulative, torch.tensor(fraction, device=cumulative.device)).item() + 1)


def candidate_metrics(
    singular: torch.Tensor,
    target: torch.Tensor,
    candidates: Iterable[Candidate] = DEFAULT_CANDIDATES,
) -> list[dict]:
    values = []
    muon_norm = None
    for level, candidate in enumerate(candidates):
        response = scalar_polynomial_response(singular, candidate)
        normalized = response / response.max().clamp_min(1e-30)
        norm = float(response.norm())
        if level == 0:
            muon_norm = max(norm, 1e-30)
        values.append(
            {
                "level": level,
                "candidate": candidate.name,
                "mismatch": float((normalized - target).square().mean()),
                "response_rank_gt_0p5": response_rank(response),
                "response_e95": energy_rank(response, 0.95),
                "response_norm": norm,
            }
        )
    assert muon_norm is not None
    for value in values:
        value["norm_ratio_muon"] = value["response_norm"] / muon_norm
    return values


def load_controller_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("format") != FORMAT:
        raise ValueError(f"adaptive config must use format {FORMAT!r}")
    required = {
        "signal",
        "granularity",
        "warmup_updates",
        "decision_interval",
        "observation_updates",
        "tau",
        "hysteresis_relative_improvement",
        "max_level_change",
        "candidates",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"adaptive config missing fields: {sorted(missing)}")
    if payload["signal"] not in ("mode_snr", "spectrum_e95"):
        raise ValueError("unsupported adaptive signal")
    if payload["granularity"] not in ("global", "operator", "matrix"):
        raise ValueError("unsupported adaptive granularity")
    if int(payload["warmup_updates"]) <= 0 or int(payload["decision_interval"]) <= 0:
        raise ValueError("warmup and decision interval must be positive")
    if int(payload["observation_updates"]) != 2:
        raise ValueError("formal controller requires exactly two observation updates")
    candidates = []
    for item in payload["candidates"]:
        candidate = Candidate(
            str(item["name"]),
            str(item["backend"]),
            int(item["promotion_steps"]),
            int(item["suppression_steps"]),
            float(item["alpha"]),
        )
        coefficient_sequence(
            candidate.backend,
            candidate.promotion_steps,
            candidate.suppression_steps,
            candidate.alpha,
        )
        candidates.append(candidate)
    if not candidates or candidates[0].backend != "muon5ns":
        raise ValueError("candidate level 0 must be Muon 5NS")
    payload["candidates"] = [asdict(candidate) for candidate in candidates]
    return payload, hashlib.sha256(raw).hexdigest()


def write_controller_config(
    path: Path,
    *,
    signal: str,
    granularity: str,
    tau: float,
    norm_guard: tuple[float, float] | None = (0.20, 1.15),
    warmup_updates: int = 100,
    decision_interval: int = 100,
    observation_updates: int = 2,
) -> dict:
    payload = {
        "format": FORMAT,
        "signal": signal,
        "granularity": granularity,
        "warmup_updates": warmup_updates,
        "decision_interval": decision_interval,
        "observation_updates": observation_updates,
        "tau": tau,
        "norm_guard": None if norm_guard is None else list(norm_guard),
        "hysteresis_relative_improvement": 0.10,
        "max_level_change": 1,
        "candidates": [asdict(candidate) for candidate in DEFAULT_CANDIDATES],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


class AdaptivePionController:
    """Stateful controller whose formal decisions occur after optimizer steps."""

    def __init__(self, config: dict, parameter_names: Iterable[str]):
        self.config = dict(config)
        self.candidates = tuple(Candidate(**item) for item in config["candidates"])
        self.parameter_names = tuple(parameter_names)
        for name in self.parameter_names:
            parameter_role(name)
        self.current_levels: dict[str, int] = {}
        self.decision_count = 0
        self.last_decision_step = 0
        self._observations: dict[str, list[torch.Tensor]] = {
            name: [] for name in self.parameter_names
        }
        self.last_event: dict | None = None

    @property
    def warmup_updates(self) -> int:
        return int(self.config["warmup_updates"])

    @property
    def decision_interval(self) -> int:
        return int(self.config["decision_interval"])

    @property
    def observation_updates(self) -> int:
        return int(self.config["observation_updates"])

    def decision_steps(self, total_updates: int) -> set[int]:
        return set(range(self.warmup_updates, total_updates, self.decision_interval))

    def observation_update_indices(self, total_updates: int) -> set[int]:
        result: set[int] = set()
        for step in self.decision_steps(total_updates):
            result.update(range(step - self.observation_updates, step))
        return result

    def should_observe(self, update: int, total_updates: int) -> bool:
        return update in self.observation_update_indices(total_updates)

    def observe_microbatch(self, gradients: Mapping[str, torch.Tensor]) -> None:
        if set(gradients) != set(self.parameter_names):
            raise ValueError("adaptive microbatch gradient names differ from matrix parameters")
        for name, gradient in gradients.items():
            self._observations[name].append(gradient.detach().float().cpu().clone())

    def _group_key(self, name: str) -> str:
        granularity = self.config["granularity"]
        if granularity == "global":
            return "global"
        if granularity == "operator":
            return parameter_role(name)
        return name

    def _target(self, singular: torch.Tensor, projections: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.config["signal"] == "mode_snr":
            snr = mode_snr(projections)
            target = soft_reliability(snr, float(self.config["tau"]))
            return target, {
                "snr_median": float(snr.median()),
                "snr_mean": float(snr.mean()),
                "snr_gt_tau": int((snr > float(self.config["tau"])).sum()),
                "target_soft_rank": float(target.sum()),
            }
        e95 = energy_rank(singular, 0.95)
        target = torch.zeros_like(singular)
        target[:e95] = 1
        return target, {
            "snr_median": None,
            "snr_mean": None,
            "snr_gt_tau": None,
            "target_soft_rank": float(e95),
        }

    def decide(self, model_step: int, optimizer) -> dict:
        if model_step < self.warmup_updates or (
            model_step - self.warmup_updates
        ) % self.decision_interval:
            raise ValueError(f"step {model_step} is not an adaptive decision boundary")
        expected_groups = self.observation_updates * 4
        counts = {name: len(values) for name, values in self._observations.items()}
        if set(counts.values()) != {expected_groups}:
            raise RuntimeError(f"expected {expected_groups} microbatch gradients per matrix, found {counts}")
        started = time.perf_counter()
        named_parameters = optimizer.named_parameters()
        matrix_rows: list[dict] = []
        group_metrics: dict[str, list[list[dict]]] = {}
        for name in self.parameter_names:
            parameter = named_parameters[name]
            momentum = optimizer.state[parameter].get("momentum_buffer")
            if momentum is None:
                raise RuntimeError(f"missing momentum buffer at adaptive decision: {name}")
            u, singular, vh = torch.linalg.svd(momentum.detach().float(), full_matrices=False)
            v = vh.mT
            projection_rows = []
            for gradient in self._observations[name]:
                local = gradient.to(parameter.device)
                projection_rows.append((u * (local @ v)).sum(0))
            projections = torch.stack(projection_rows)
            target, target_summary = self._target(singular, projections)
            metrics = candidate_metrics(singular, target, self.candidates)
            key = self._group_key(name)
            group_metrics.setdefault(key, []).append(metrics)
            matrix_rows.append(
                {
                    "parameter": name,
                    "operator": parameter_role(name),
                    "group": key,
                    "input_rank": len(singular),
                    "input_e95": energy_rank(singular, 0.95),
                    "input_stable_rank": float(singular.square().sum() / singular[0].square().clamp_min(1e-30)),
                    **target_summary,
                    "candidates": metrics,
                }
            )

        decisions = []
        group_levels: dict[str, int] = {}
        guard = self.config.get("norm_guard")
        for key, matrix_metrics in sorted(group_metrics.items()):
            aggregated = []
            for level, candidate in enumerate(self.candidates):
                rows = [values[level] for values in matrix_metrics]
                aggregated.append(
                    {
                        "level": level,
                        "candidate": candidate.name,
                        "mismatch": statistics.median(row["mismatch"] for row in rows),
                        "norm_ratio_muon": statistics.median(row["norm_ratio_muon"] for row in rows),
                        "response_rank_gt_0p5": statistics.median(
                            row["response_rank_gt_0p5"] for row in rows
                        ),
                        "response_e95": statistics.median(row["response_e95"] for row in rows),
                    }
                )
            eligible = aggregated
            if guard is not None:
                low, high = map(float, guard)
                eligible = [row for row in aggregated if low <= row["norm_ratio_muon"] <= high]
            if not eligible:
                eligible = [aggregated[0]]
            suggested = min(eligible, key=lambda row: (row["mismatch"], row["level"]))["level"]
            current = int(self.current_levels.get(key, 0))
            improvement = (
                aggregated[current]["mismatch"] - aggregated[suggested]["mismatch"]
            ) / max(aggregated[current]["mismatch"], 1e-30)
            applied = current
            reason = "hysteresis_keep"
            if suggested != current and improvement >= float(
                self.config["hysteresis_relative_improvement"]
            ):
                max_change = int(self.config["max_level_change"])
                direction = 1 if suggested > current else -1
                applied = current + direction * min(max_change, abs(suggested - current))
                reason = "move_toward_suggestion"
            # A one-level move can itself cross the norm boundary on spectra
            # whose ordering differs from the offline estimate.
            if guard is not None:
                low, high = map(float, guard)
                while applied > 0 and not low <= aggregated[applied]["norm_ratio_muon"] <= high:
                    applied -= 1
                    reason = "norm_guard_backoff"
            self.current_levels[key] = applied
            group_levels[key] = applied
            decisions.append(
                {
                    "group": key,
                    "previous_level": current,
                    "suggested_level": suggested,
                    "applied_level": applied,
                    "improvement": improvement,
                    "reason": reason,
                    "candidates": aggregated,
                }
            )

        overrides = {
            name: self.candidates[group_levels[self._group_key(name)]].segment
            for name in self.parameter_names
        }
        optimizer.set_dynamic_segments(overrides)
        self.decision_count += 1
        self.last_decision_step = model_step
        self._observations = {name: [] for name in self.parameter_names}
        event = {
            "event": "adaptive_pion_decision",
            "step": model_step,
            "decision_index": self.decision_count,
            "signal": self.config["signal"],
            "granularity": self.config["granularity"],
            "tau": float(self.config["tau"]),
            "elapsed_seconds": time.perf_counter() - started,
            "decisions": decisions,
            "matrices": matrix_rows,
        }
        self.last_event = event
        return event

    def state_dict(self) -> dict:
        if any(self._observations.values()):
            raise RuntimeError("formal checkpoints must be saved with an empty adaptive observation buffer")
        return {
            "format": FORMAT,
            "config_sha256": canonical_sha256(self.config),
            "current_levels": dict(self.current_levels),
            "decision_count": self.decision_count,
            "last_decision_step": self.last_decision_step,
        }

    def load_state_dict(self, state: dict, optimizer) -> None:
        if state.get("format") != FORMAT:
            raise ValueError("unsupported adaptive controller checkpoint format")
        if state.get("config_sha256") != canonical_sha256(self.config):
            raise ValueError("adaptive controller config differs from checkpoint")
        self.current_levels = {str(key): int(value) for key, value in state["current_levels"].items()}
        self.decision_count = int(state["decision_count"])
        self.last_decision_step = int(state["last_decision_step"])
        overrides = {
            name: self.candidates[self.current_levels[self._group_key(name)]].segment
            for name in self.parameter_names
        } if self.current_levels else {}
        optimizer.set_dynamic_segments(overrides)

