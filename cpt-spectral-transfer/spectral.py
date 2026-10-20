"""Matched Muon/Pion spectral transforms and matrix optimizer.

The optimizer shell is intentionally shared.  Muon and Pion differ only in
the coefficient sequence passed to :func:`apply_polynomial`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


MUON = (3.4445, -4.7750, 2.0315)
PROMOTION = (1.875, -1.25, 0.375)


def _psspp_blend_fraction(backend: str) -> float | None:
    """Decode a norm-controlled Muon/PSSPP interpolation backend.

    ``b000`` is the original PSSPP transform and ``b100`` is the Muon
    direction.  Every interpolation is rescaled per matrix to the natural
    Frobenius norm of PSSPP, so the suffix changes spectral shape without
    changing matrix-update magnitude.
    """

    prefix = "pion_blend_psspp_norm_b"
    if not backend.startswith(prefix):
        return None
    encoded = backend.removeprefix(prefix)
    if len(encoded) != 3 or not encoded.isdigit():
        raise ValueError(f"invalid norm-controlled PSSPP blend backend: {backend!r}")
    percent = int(encoded)
    if not 0 <= percent <= 100:
        raise ValueError(f"PSSPP blend percentage must be in [0, 100]: {backend!r}")
    return percent / 100.0


def _psspp_signal_fraction(backend: str) -> float | None:
    """Decode an equal-norm raw-spectrum residual added to PSSPP.

    Unlike :func:`_psspp_blend_fraction`, whose residual endpoint is the
    whitened Muon direction, this endpoint is the Frobenius-normalized raw
    momentum direction.  It therefore restores large/mid singular modes in
    proportion to their input singular values while leaving the weakest tail
    comparatively suppressed.  Both endpoints are normalized before mixing,
    and the final transform is rescaled to natural PSSPP Frobenius norm.
    """

    prefix = "pion_signal_psspp_norm_b"
    if not backend.startswith(prefix):
        return None
    encoded = backend.removeprefix(prefix)
    if len(encoded) != 3 or not encoded.isdigit():
        raise ValueError(f"invalid norm-controlled PSSPP signal backend: {backend!r}")
    percent = int(encoded)
    if not 0 <= percent <= 100:
        raise ValueError(f"PSSPP signal percentage must be in [0, 100]: {backend!r}")
    return percent / 100.0


def _psspp_pulse_pattern(backend: str) -> str | None:
    """Decode a temporal PSSPP/Muon-residual pulse policy name."""

    prefix = "pion_pulse_psspp_"
    if not backend.startswith(prefix):
        return None
    pattern = backend.removeprefix(prefix)
    supported = {
        "alt20",
        "block20p10",
        "pulse40p4",
        "pulse50p5",
        "pulse100p10",
        "burst100p100",
    }
    if pattern not in supported:
        raise ValueError(f"unknown PSSPP pulse pattern: {backend!r}")
    return pattern


def _psspp_pulse_percent(pattern: str, step: int) -> int:
    """Return the Muon residual percentage for a supported pulse policy."""

    if pattern == "alt20":
        return 20 if step % 2 == 0 else 0
    if pattern == "block20p10":
        return 20 if step % 10 < 5 else 0
    if pattern == "pulse40p4":
        return 40 if step % 4 == 0 else 0
    if pattern == "pulse50p5":
        return 50 if step % 5 == 0 else 0
    if pattern == "pulse100p10":
        return 100 if step % 10 == 0 else 0
    if pattern == "burst100p100":
        return 100 if step % 100 < 10 else 0
    raise ValueError(f"unknown PSSPP pulse pattern: {pattern!r}")


def suppression(alpha: float) -> tuple[float, float, float]:
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.875:
        raise ValueError("suppression alpha must be finite and in [0, 1.875]")
    return alpha, 2.5 - 2.0 * alpha, alpha - 1.5


def coefficient_sequence(
    backend: str,
    promotion_steps: int = 0,
    suppression_steps: int = 0,
    alpha: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    if backend == "muon5ns":
        if (promotion_steps, suppression_steps, alpha) != (0, 0, 0.0):
            raise ValueError("muon5ns requires P=0, S=0, alpha=0")
        return (MUON,) * 5
    blend_fraction = _psspp_blend_fraction(backend)
    signal_fraction = _psspp_signal_fraction(backend)
    pulse_pattern = _psspp_pulse_pattern(backend)
    if blend_fraction is not None or signal_fraction is not None or pulse_pattern is not None:
        if (promotion_steps, suppression_steps) != (3, 2):
            raise ValueError(
                f"norm-controlled PSSPP residual {backend!r} requires P=3, S=2"
            )
        suppressed = suppression(alpha)
        return (PROMOTION, suppressed, suppressed, PROMOTION, PROMOTION)
    ordered_backend = backend
    if backend.startswith(("pion_policy_", "pion_profile_")) and "_order_" in backend:
        prefix = "pion_policy_" if backend.startswith("pion_policy_") else "pion_profile_"
        _, order = backend.removeprefix(prefix).split("_order_", 1)
        ordered_backend = f"pion_order_{order}"
    if ordered_backend.startswith("pion_order_"):
        order = ordered_backend.removeprefix("pion_order_")
        if not order or any(stage not in "ps" for stage in order):
            raise ValueError(f"invalid ordered Pion backend: {backend!r}")
        if order.count("p") != promotion_steps or order.count("s") != suppression_steps:
            raise ValueError(
                f"ordered Pion backend {backend!r} disagrees with P={promotion_steps}, S={suppression_steps}"
            )
        suppressed = suppression(alpha)
        return tuple(PROMOTION if stage == "p" else suppressed for stage in order)
    if backend != "pion" and not backend.startswith(("pion_policy_", "pion_profile_")):
        raise ValueError(f"unsupported spectral backend: {backend!r}")
    if promotion_steps < 0 or suppression_steps < 0:
        raise ValueError("P and S must be non-negative")
    if promotion_steps + suppression_steps == 0:
        raise ValueError("Pion requires at least one iteration")
    return (PROMOTION,) * promotion_steps + (suppression(alpha),) * suppression_steps


@torch.no_grad()
def apply_polynomial(
    matrix: torch.Tensor,
    coefficients: Iterable[tuple[float, float, float]],
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Apply a normalized quintic sequence to 2D or batched matrices."""

    if matrix.ndim < 2 or not matrix.is_floating_point():
        raise TypeError("matrix must be a floating-point tensor with ndim >= 2")
    result = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + epsilon)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.mT
    for a, b, c in coefficients:
        gram = result @ result.mT
        gram_result = gram @ result
        result = a * result + b * gram_result + c * (gram @ gram_result)
    return result.mT if transposed else result


def spectral_transform(
    matrix: torch.Tensor,
    backend: str,
    promotion_steps: int = 0,
    suppression_steps: int = 0,
    alpha: float = 0.0,
) -> torch.Tensor:
    matrix_fp32 = matrix.float()
    blend_fraction = _psspp_blend_fraction(backend)
    signal_fraction = _psspp_signal_fraction(backend)
    if blend_fraction is None and signal_fraction is None:
        return apply_polynomial(
            matrix_fp32,
            coefficient_sequence(backend, promotion_steps, suppression_steps, alpha),
        ).to(matrix.dtype)

    psspp = apply_polynomial(
        matrix_fp32,
        coefficient_sequence(backend, promotion_steps, suppression_steps, alpha),
    )
    residual_fraction = blend_fraction if blend_fraction is not None else signal_fraction
    assert residual_fraction is not None
    if residual_fraction == 0.0:
        return psspp.to(matrix.dtype)
    target_norm = psspp.norm(dim=(-2, -1), keepdim=True)
    if blend_fraction is not None:
        residual = apply_polynomial(matrix_fp32, (MUON,) * 5)
    else:
        raw_norm = matrix_fp32.norm(dim=(-2, -1), keepdim=True)
        residual = torch.where(
            raw_norm > 0,
            matrix_fp32 * (target_norm / raw_norm.clamp_min(1e-30)),
            torch.zeros_like(matrix_fp32),
        )
    blended = psspp.lerp(residual, residual_fraction)
    blended_norm = blended.norm(dim=(-2, -1), keepdim=True)
    rescaled = torch.where(
        target_norm > 0,
        blended * (target_norm / blended_norm.clamp_min(1e-30)),
        torch.zeros_like(blended),
    )
    return rescaled.to(matrix.dtype)


@dataclass(frozen=True)
class Segment:
    start: int
    backend: str
    promotion_steps: int
    suppression_steps: int
    alpha: float


def parse_schedule(value: str) -> tuple[Segment, ...]:
    segments: list[Segment] = []
    for index, raw in enumerate(value.split(",")):
        fields = raw.strip().split(":")
        if len(fields) != 5:
            raise ValueError(f"schedule segment {index} must be start:backend:P:S:alpha")
        segment = Segment(int(fields[0]), fields[1], int(fields[2]), int(fields[3]), float(fields[4]))
        coefficient_sequence(
            segment.backend,
            segment.promotion_steps,
            segment.suppression_steps,
            segment.alpha,
        )
        segments.append(segment)
    if not segments or segments[0].start != 0:
        raise ValueError("schedule must start at update 0")
    if any(a.start >= b.start for a, b in zip(segments, segments[1:])):
        raise ValueError("schedule start updates must be strictly increasing")
    return tuple(segments)


def segment_at(schedule: tuple[Segment, ...], step: int) -> Segment:
    active = schedule[0]
    for segment in schedule[1:]:
        if segment.start > step:
            break
        active = segment
    return active


class MatchedMuon(torch.optim.Optimizer):
    """Momentum + matched spectral transform for hidden 2D matrices only."""

    def __init__(
        self,
        params,
        *,
        lr: float = 3e-3,
        momentum: float = 0.95,
        schedule: tuple[Segment, ...],
        diagnostic_param: torch.nn.Parameter | None = None,
        diagnostic_name: str = "",
        parameter_names: list[str] | None = None,
    ):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.schedule = schedule
        self.update_step = 0
        self.diagnostic_param = diagnostic_param
        self.diagnostic_name = diagnostic_name
        parameters = [parameter for group in self.param_groups for parameter in group["params"]]
        if parameter_names is not None and len(parameter_names) != len(parameters):
            raise ValueError("parameter_names must match optimizer parameter count")
        self.parameter_names = {
            id(parameter): name
            for parameter, name in zip(parameters, parameter_names or [""] * len(parameters))
        }
        self._named_parameters = {
            name: parameter
            for parameter, name in zip(parameters, parameter_names or [""] * len(parameters))
            if name
        }
        self.dynamic_segments: dict[str, Segment] = {}
        self.capture_diagnostic = False
        self.raw_diagnostic_gradient: torch.Tensor | None = None
        self.last_diagnostic: dict | None = None

    def set_schedule(self, schedule: tuple[Segment, ...]) -> None:
        self.schedule = schedule

    def named_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self._named_parameters)

    def set_dynamic_segments(self, segments: dict[str, Segment]) -> None:
        unknown = set(segments) - set(self._named_parameters)
        if unknown:
            raise ValueError(f"dynamic spectral overrides contain unknown parameters: {sorted(unknown)}")
        for segment in segments.values():
            coefficient_sequence(
                segment.backend,
                segment.promotion_steps,
                segment.suppression_steps,
                segment.alpha,
            )
        self.dynamic_segments = dict(segments)

    def request_diagnostic(self, raw_gradient: torch.Tensor) -> None:
        self.capture_diagnostic = True
        self.raw_diagnostic_gradient = raw_gradient.detach().float().cpu().clone()
        self.last_diagnostic = None

    def _segment_for_parameter(self, segment: Segment, parameter: torch.nn.Parameter) -> Segment:
        name = self.parameter_names.get(id(parameter), "")
        if name in self.dynamic_segments:
            return self.dynamic_segments[name]
        pulse_pattern = _psspp_pulse_pattern(segment.backend)
        if pulse_pattern is not None:
            percent = _psspp_pulse_percent(pulse_pattern, self.update_step)
            return Segment(
                segment.start,
                f"pion_blend_psspp_norm_b{percent:03d}",
                segment.promotion_steps,
                segment.suppression_steps,
                segment.alpha,
            )
        if segment.backend.startswith("pion_profile_"):
            if not name:
                raise RuntimeError("parameter-aware Pion backend requires parameter_names")
            profile = segment.backend.removeprefix("pion_profile_")
            order = ""
            if "_order_" in profile:
                profile, order = profile.split("_order_", 1)
            if profile.startswith("residual_"):
                if order != "psspp" or (segment.promotion_steps, segment.suppression_steps) != (3, 2):
                    raise ValueError("residual profiles require order PSSPP with P=3, S=2")
                profile_fields = profile.removeprefix("residual_").rsplit("_b", 1)
                if len(profile_fields) != 2 or not profile_fields[1].isdigit():
                    raise ValueError(f"invalid residual profile: {profile!r}")
                operator, encoded_fraction = profile_fields
                if len(encoded_fraction) != 3 or not 0 <= int(encoded_fraction) <= 100:
                    raise ValueError(f"invalid residual fraction in profile: {profile!r}")
                selected = {
                    "qkv": name.endswith(".attention.query_key_value.weight"),
                    "attention_output": name.endswith(".attention.dense.weight"),
                    "mlp_expand": name.endswith(".mlp.dense_h_to_4h.weight"),
                    "mlp_contract": name.endswith(".mlp.dense_4h_to_h.weight"),
                }.get(operator)
                if selected is None:
                    raise ValueError(f"unknown residual-profile operator: {operator!r}")
                fraction = encoded_fraction if selected else "010"
                return Segment(
                    segment.start,
                    f"pion_blend_psspp_norm_b{fraction}",
                    segment.promotion_steps,
                    segment.suppression_steps,
                    segment.alpha,
                )
            alpha_overrides = {
                "contract_a000": (None, 0.0),
                "contract_a015": (None, 0.15),
                "contract_a025": (None, 0.25),
                "contract_a035": (None, 0.35),
                "contract_a065": (None, 0.65),
                "qkv_a025_contract_a025": (0.25, 0.25),
                "qkv_a075_contract_a025": (0.75, 0.25),
            }
            if profile in alpha_overrides:
                qkv_alpha, contract_alpha = alpha_overrides[profile]
                active_alpha = segment.alpha
                if qkv_alpha is not None and name.endswith(
                    ".attention.query_key_value.weight"
                ):
                    active_alpha = qkv_alpha
                if contract_alpha is not None and name.endswith(
                    ".mlp.dense_4h_to_h.weight"
                ):
                    active_alpha = contract_alpha
                return Segment(
                    segment.start,
                    f"pion_order_{order}" if order else "pion",
                    segment.promotion_steps,
                    segment.suppression_steps,
                    active_alpha,
                )
            layer_token = name.split("gpt_neox.layers.", 1)[-1].split(".", 1)[0]
            layer = int(layer_token) if layer_token.isdigit() else -1
            if profile == "all_qkv_l4_pppss":
                active_order = (
                    "pppss"
                    if name.endswith(".attention.query_key_value.weight") and layer == 4
                    else order
                )
                return Segment(
                    segment.start,
                    f"pion_order_{active_order}" if active_order else "pion",
                    segment.promotion_steps,
                    segment.suppression_steps,
                    segment.alpha,
                )
            if not name.endswith(".attention.query_key_value.weight") or not 0 <= layer < 6:
                return Segment(segment.start, "muon5ns", 0, 0, 0.0)

            active_order = order
            active_alpha = segment.alpha
            if profile == "qkv_l4_a075":
                active_alpha = .75 if layer == 4 else .5
            elif profile == "qkv_l4_a100":
                active_alpha = 1.0 if layer == 4 else .5
            elif profile == "qkv_early_a075":
                active_alpha = .75 if layer < 3 else .5
            elif profile == "qkv_early_a100":
                active_alpha = 1.0 if layer < 3 else .5
            elif profile == "qkv_alpha_desc":
                active_alpha = (.75, .75, .625, .625, .5, .5)[layer]
            elif profile == "qkv_splitorder":
                active_order = "ppps" if layer < 3 else "ppsp"
                active_alpha = .5
            else:
                raise ValueError(f"unknown Pion parameter profile: {profile!r}")
            return Segment(
                segment.start,
                f"pion_order_{active_order}" if active_order else "pion",
                segment.promotion_steps,
                segment.suppression_steps,
                active_alpha,
            )
        if not segment.backend.startswith("pion_policy_"):
            return segment
        if not name:
            raise RuntimeError("parameter-aware Pion backend requires parameter_names")
        policy = segment.backend.removeprefix("pion_policy_")
        order = ""
        if "_order_" in policy:
            policy, order = policy.split("_order_", 1)
        layer_token = name.split("gpt_neox.layers.", 1)[-1].split(".", 1)[0]
        layer = int(layer_token) if layer_token.isdigit() else -1
        selected = {
            "mlp": ".mlp." in name,
            "attention": ".attention." in name,
            "output": name.endswith(".attention.dense.weight")
            or name.endswith(".mlp.dense_4h_to_h.weight"),
            "input": name.endswith(".attention.query_key_value.weight")
            or name.endswith(".mlp.dense_h_to_4h.weight"),
            "qkv": name.endswith(".attention.query_key_value.weight"),
            "qkv_early2": name.endswith(".attention.query_key_value.weight") and 0 <= layer < 2,
            "qkv_middle2": name.endswith(".attention.query_key_value.weight") and 2 <= layer < 4,
            "qkv_late2": name.endswith(".attention.query_key_value.weight") and 4 <= layer < 6,
            "qkv_first3": name.endswith(".attention.query_key_value.weight") and 0 <= layer < 3,
            "qkv_last3": name.endswith(".attention.query_key_value.weight") and 3 <= layer < 6,
            "qkv_even": name.endswith(".attention.query_key_value.weight") and layer in (0, 2, 4),
            "qkv_odd": name.endswith(".attention.query_key_value.weight") and layer in (1, 3, 5),
            "qkv_exclude_layer0": name.endswith(".attention.query_key_value.weight") and layer != 0,
            "qkv_exclude_layer1": name.endswith(".attention.query_key_value.weight") and layer != 1,
            "qkv_exclude_layer2": name.endswith(".attention.query_key_value.weight") and layer != 2,
            "qkv_exclude_layer3": name.endswith(".attention.query_key_value.weight") and layer != 3,
            "qkv_exclude_layer4": name.endswith(".attention.query_key_value.weight") and layer != 4,
            "qkv_exclude_layer5": name.endswith(".attention.query_key_value.weight") and layer != 5,
            "all_except_qkv_layer0": not (name.endswith(".attention.query_key_value.weight") and layer == 0),
            "all_except_qkv_layer1": not (name.endswith(".attention.query_key_value.weight") and layer == 1),
            "all_except_qkv_layer2": not (name.endswith(".attention.query_key_value.weight") and layer == 2),
            "all_except_qkv_layer3": not (name.endswith(".attention.query_key_value.weight") and layer == 3),
            "all_except_qkv_layer4": not (name.endswith(".attention.query_key_value.weight") and layer == 4),
            "all_except_qkv_layer5": not (name.endswith(".attention.query_key_value.weight") and layer == 5),
            "attention_output": name.endswith(".attention.dense.weight"),
            "mlp_expand": name.endswith(".mlp.dense_h_to_4h.weight"),
            "mlp_contract": name.endswith(".mlp.dense_4h_to_h.weight"),
            "qkv_mlp_contract": name.endswith(".attention.query_key_value.weight")
            or name.endswith(".mlp.dense_4h_to_h.weight"),
            "qkv_mlp_expand": name.endswith(".attention.query_key_value.weight")
            or name.endswith(".mlp.dense_h_to_4h.weight"),
            "attention_output_mlp_expand": name.endswith(".attention.dense.weight")
            or name.endswith(".mlp.dense_h_to_4h.weight"),
            "exclude_qkv": not name.endswith(".attention.query_key_value.weight"),
            "exclude_attention_output": not name.endswith(".attention.dense.weight"),
            "exclude_mlp_expand": not name.endswith(".mlp.dense_h_to_4h.weight"),
            "exclude_mlp_contract": not name.endswith(".mlp.dense_4h_to_h.weight"),
            "early": 0 <= layer < 4,
            "late": layer >= 4,
        }.get(policy)
        if selected is None:
            raise ValueError(f"unknown Pion parameter policy: {policy!r}")
        if selected:
            return Segment(
                segment.start,
                f"pion_order_{order}" if order else "pion",
                segment.promotion_steps,
                segment.suppression_steps,
                segment.alpha,
            )
        return Segment(segment.start, "muon5ns", 0, 0, 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        segment = segment_at(self.schedule, self.update_step)
        for group in self.param_groups:
            by_shape: dict[tuple[tuple[int, ...], Segment], list[torch.nn.Parameter]] = {}
            for parameter in group["params"]:
                if parameter.grad is not None:
                    active = self._segment_for_parameter(segment, parameter)
                    by_shape.setdefault((tuple(parameter.shape), active), []).append(parameter)
            for (shape, active), parameters in by_shape.items():
                gradients = torch.stack([p.grad.float() for p in parameters])
                buffers = []
                for parameter, gradient in zip(parameters, gradients):
                    state = self.state[parameter]
                    buffer = state.get("momentum_buffer")
                    if buffer is None:
                        buffer = torch.zeros_like(gradient)
                        state["momentum_buffer"] = buffer
                    buffer.mul_(group["momentum"]).add_(gradient)
                    buffers.append(buffer)
                backend_input = torch.stack(buffers)
                transformed = spectral_transform(
                    backend_input,
                    active.backend,
                    active.promotion_steps,
                    active.suppression_steps,
                    active.alpha,
                )
                scale = math.sqrt(max(1.0, shape[-2] / shape[-1]))
                for index, parameter in enumerate(parameters):
                    if self.capture_diagnostic and parameter is self.diagnostic_param:
                        self.last_diagnostic = {
                            "update_step": self.update_step,
                            "param": self.diagnostic_name,
                            "route_backend": segment.backend,
                            "backend": active.backend,
                            "promotion_steps": active.promotion_steps,
                            "suppression_steps": active.suppression_steps,
                            "alpha": active.alpha,
                            "raw_gradient": self.raw_diagnostic_gradient,
                            "backend_input": backend_input[index].detach().cpu().clone(),
                            "transformed": transformed[index].detach().float().cpu().clone(),
                            "rectangular_scale": scale,
                            "lr": float(group["lr"]),
                        }
                    parameter.add_(transformed[index].to(parameter.dtype), alpha=-group["lr"] * scale)
        self.capture_diagnostic = False
        self.raw_diagnostic_gradient = None
        self.update_step += 1

    def state_dict(self):
        payload = super().state_dict()
        payload["matched_muon_update_step"] = self.update_step
        payload["matched_muon_dynamic_segments"] = {
            name: {
                "start": segment.start,
                "backend": segment.backend,
                "promotion_steps": segment.promotion_steps,
                "suppression_steps": segment.suppression_steps,
                "alpha": segment.alpha,
            }
            for name, segment in self.dynamic_segments.items()
        }
        return payload

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.update_step = int(state_dict.pop("matched_muon_update_step", 0))
        serialized = state_dict.pop("matched_muon_dynamic_segments", {})
        result = super().load_state_dict(state_dict)
        self.set_dynamic_segments(
            {
                name: Segment(
                    int(value["start"]),
                    str(value["backend"]),
                    int(value["promotion_steps"]),
                    int(value["suppression_steps"]),
                    float(value["alpha"]),
                )
                for name, value in serialized.items()
            }
        )
        return result


def summarize_spectrum(matrix: torch.Tensor) -> dict[str, float | int | list[float]]:
    singular = torch.linalg.svdvals(matrix.float()).cpu()
    singular = torch.sort(singular, descending=True).values
    top = max(float(singular[0]), 1e-30)
    energy = singular.square()
    total_energy = energy.sum()
    cumulative = energy.cumsum(0) / total_energy.clamp_min(1e-30)

    def energy_rank(level: float) -> int:
        if float(total_energy) == 0.0:
            return 0
        return int(torch.searchsorted(cumulative, torch.tensor(level)).item() + 1)

    return {
        "singular_values": singular.tolist(),
        "fro": float(torch.linalg.vector_norm(singular)),
        "stable_rank": float(total_energy / singular[0].square().clamp_min(1e-30)),
        "rank_gt_0p9": int((singular / top > 0.9).sum()),
        "rank_gt_0p5": int((singular / top > 0.5).sum()),
        "rank_gt_0p1": int((singular / top > 0.1).sum()),
        "energy_rank_90": energy_rank(0.90),
        "energy_rank_95": energy_rank(0.95),
        "energy_rank_99": energy_rank(0.99),
    }
