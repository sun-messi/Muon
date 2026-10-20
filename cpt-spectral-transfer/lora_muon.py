"""Minimal factor-wise LoRA support for the matched Pythia Muon experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_PREFIX = "gpt_neox.layers."
DIAGNOSTIC_MODULE = "gpt_neox.layers.0.mlp.dense_4h_to_h"


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable ``scaling * B @ A`` adapter."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must be in [1, min(in_features, out_features)]")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.base.weight, self.base.bias)
        adapter = F.linear(F.linear(inputs, self.lora_A), self.lora_B)
        return base + self.scaling * adapter

    def effective_delta(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


@dataclass
class InjectedLoRA:
    modules: dict[str, LoRALinear]
    parameters: list[nn.Parameter]
    parameter_names: list[str]
    diagnostic: LoRALinear


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    fields = dotted_name.split(".")
    parent = model
    for field in fields[:-1]:
        parent = getattr(parent, field)
    return parent, fields[-1]


def inject_hidden_lora(model: nn.Module, rank: int, alpha: float) -> InjectedLoRA:
    """Replace all 24 Pythia hidden Linear modules, excluding the LM head."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith(TARGET_PREFIX) and isinstance(module, nn.Linear)
    ]
    modules: dict[str, LoRALinear] = {}
    parameters: list[nn.Parameter] = []
    parameter_names: list[str] = []
    for name, module in targets:
        parent, attribute = _resolve_parent(model, name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, attribute, wrapped)
        modules[name] = wrapped
        parameters.extend((wrapped.lora_A, wrapped.lora_B))
        parameter_names.extend((f"{name}.lora_A", f"{name}.lora_B"))
    if len(modules) != 24:
        raise RuntimeError(f"expected 24 hidden Linear targets, found {len(modules)}")
    if DIAGNOSTIC_MODULE not in modules:
        raise RuntimeError(f"diagnostic LoRA module not found: {DIAGNOSTIC_MODULE}")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in trainable} != {id(parameter) for parameter in parameters}:
        raise RuntimeError("LoRA trainable parameter partition is incomplete or overlapping")
    return InjectedLoRA(modules, parameters, parameter_names, modules[DIAGNOSTIC_MODULE])


def frozen_state_hash(model: nn.Module) -> str:
    """Hash every frozen parameter and buffer to prove the base stayed unchanged."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    for name, buffer in sorted(model.named_buffers()):
        digest.update(name.encode())
        digest.update(buffer.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def expected_trainable_parameters(rank: int) -> int:
    # Per layer: qkv 1024r + attn-out 512r + MLP-expand 1280r +
    # MLP-contract 1280r = 4096r; Pythia-31M has six layers.
    return 6 * 4096 * rank
