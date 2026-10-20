#!/usr/bin/env python3
"""Matched Pythia-31M CPT/midtraining with a polynomial-only spectral backend."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_pion import AdaptivePionController, canonical_sha256, load_controller_config
from lora_muon import (
    DIAGNOSTIC_MODULE,
    expected_trainable_parameters,
    frozen_state_hash,
    inject_hidden_lora,
)
from smoltalk_data import MONITOR_ROWS, TokenWindowDataset, dataset_fingerprint, sha256_file
from ord_full_dataset import ORDFullSequenceDataset
from spectral import MatchedMuon, parse_schedule, segment_at, spectral_transform, summarize_spectrum
from train_mid import (
    DIAGNOSTIC_PARAM,
    append_event,
    atomic_save,
    config_hash,
    evaluate_ce,
    json_dump,
    model_hash,
    optimizer_hash,
    parameter_partition,
    tensor_hash,
)


FORMAT = "pythia31m_smoltalk_full_state_v1"


class CoreBTokenDataset(torch.utils.data.Dataset):
    """Fixed 256-slot + 256-context + 512-continuation examples."""

    def __init__(self, cache_dir: Path, prefix: str, *, k: int, max_rows: int = 0):
        self.body = np.load(cache_dir / f"{prefix}_body.npy", mmap_mode="r")
        self.distractor = np.load(cache_dir / f"{prefix}_distractor.npy", mmap_mode="r")
        source_path = cache_dir / f"{prefix}_source_ids.npy"
        self.source = np.load(source_path, mmap_mode="r") if source_path.is_file() else np.zeros(len(self.body), dtype=np.uint8)
        self.k = int(k)
        self.rows = min(len(self.body), max_rows) if max_rows else len(self.body)
        if self.k not in (0, 32, 64, 128, 256):
            raise ValueError(f"unsupported interference K={self.k}")
        if self.body.shape != (len(self.body), 768) or self.distractor.shape != (len(self.body), 256):
            raise ValueError("invalid Core B cache geometry")
        if self.source.shape != (len(self.body),):
            raise ValueError("invalid Core B source-id geometry")

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        body = torch.as_tensor(np.asarray(self.body[index], dtype=np.int64))
        x = torch.zeros(1024, dtype=torch.long)
        x[256:] = body
        attention = torch.zeros(1024, dtype=torch.long)
        attention[256:] = 1
        effective_k = self.k if int(self.source[index]) == 0 else 0
        if effective_k:
            x[256 - effective_k:256] = torch.as_tensor(np.asarray(self.distractor[index, 256 - effective_k:256], dtype=np.int64))
            attention[256 - effective_k:256] = 1
        y = torch.full((1024,), -100, dtype=torch.long)
        y[511:1023] = body[256:768]
        position = torch.arange(1024, dtype=torch.long)
        position[attention == 0] = 0
        return x, y, attention, position


def coreb_fingerprint(dataset: CoreBTokenDataset) -> dict:
    return {
        "rows": len(dataset), "k": dataset.k,
        "body_shape": list(dataset.body.shape),
        "distractor_shape": list(dataset.distractor.shape),
        "first_body_hash": tensor_hash(torch.as_tensor(np.asarray(dataset.body[0], dtype=np.int64))),
        "first_distractor_hash": tensor_hash(torch.as_tensor(np.asarray(dataset.distractor[0], dtype=np.int64))),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/smoltalk_only_32m_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--spectral-schedule", required=True)
    parser.add_argument(
        "--adaptive-pion-config",
        type=Path,
        help="JSON controller config; omitted preserves the existing fixed spectral schedule.",
    )
    parser.add_argument(
        "--optimizer-family", choices=("matched", "pythia_adam"), default="matched"
    )
    parser.add_argument("--rho-label", default="")
    parser.add_argument("--interference-k", type=int, default=0)
    parser.add_argument("--total-updates", type=int, default=1000)
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=None,
        help="Fixed warmup length; omitted preserves the historical 10%%-of-horizon rule.",
    )
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-spectral-fork", action="store_true")
    parser.add_argument("--expected-fork-step", type=int, default=0)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--matrix-lr", type=float, default=3e-3)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--matrix-momentum", type=float, default=.95)
    parser.add_argument("--aux-weight-decay", type=float, default=.01)
    parser.add_argument("--adam-lr", type=float, default=1e-3)
    parser.add_argument("--adam-beta1", type=float, default=.9)
    parser.add_argument("--adam-beta2", type=float, default=.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-weight-decay", type=float, default=.1)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--monitor-rows", type=int, default=MONITOR_ROWS)
    parser.add_argument("--monitor-retention-every-validation", action="store_true")
    parser.add_argument("--full-validation-steps", default="")
    parser.add_argument("--retention-steps", default="0,100,500,1000")
    parser.add_argument("--save-steps", default="100,250,500,750,1000")
    parser.add_argument("--diagnostic-updates", default="0,50,99,100")
    parser.add_argument("--training-mode", choices=("full", "lora"), default="full")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=0.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def parse_int_set(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def schedule_lr(
    update: int, total_updates: int, warmup_updates: int | None = None
) -> float:
    warmup = (
        max(1, round(total_updates * .10))
        if warmup_updates is None
        else int(warmup_updates)
    )
    if total_updates <= 0 or warmup <= 0:
        raise ValueError("total_updates and warmup_updates must be positive")
    if update < warmup:
        return (update + 1) / warmup
    progress = (update - warmup) / max(1, total_updates - warmup - 1)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1.0, progress)))


def invariant_config(args, manifest: dict, adaptive_config: dict | None = None) -> dict:
    warmup_updates = (
        max(1, round(args.total_updates * .10))
        if args.warmup_updates is None
        else int(args.warmup_updates)
    )
    payload = {
        "format": (
            "smoltalk_pair_invariants_v1"
            if manifest["format"] == "pythia31m_smoltalk_only_cache_v1"
            else "pythia31m_matched_cpt_invariants_v2"
        ),
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "tokenizer_id": manifest["tokenizer_id"],
        "tokenizer_revision": manifest["tokenizer_revision"],
        "dataset_id": manifest["dataset_id"],
        "dataset_revision": manifest["dataset_revision"],
        "cache_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "seed": args.seed,
        "context": manifest["context"],
        "total_updates": args.total_updates,
        "device_batch_size": args.device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "matrix_lr": args.matrix_lr,
        "aux_lr": args.aux_lr,
        "matrix_momentum": args.matrix_momentum,
        "matrix_weight_decay": 0.0,
        "aux_weight_decay": args.aux_weight_decay,
        "warmup_updates": warmup_updates,
        "min_lr_ratio": .1,
        "dtype": "bfloat16",
        "spectral_dtype": "float32",
        "clip_norm": 1.0,
        "compile": args.compile,
        "loss_mask": manifest["loss_mask"],
        "optimizer_family": args.optimizer_family,
        "rho_label": args.rho_label,
        "interference_k": args.interference_k,
        "adaptive_pion_config_sha256": (
            canonical_sha256(adaptive_config) if adaptive_config is not None else "none"
        ),
    }
    if manifest["format"] != "pythia31m_smoltalk_only_cache_v1":
        payload.update(
            cache_format=manifest["format"],
            primary_validation_name=manifest.get("primary_validation_name", "primary"),
        )
    if args.training_mode == "lora":
        payload.update(
            training_mode="lora",
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_scaling=args.lora_alpha / args.lora_rank,
            lora_dropout=0.0,
            lora_bias="none",
            lora_targets="all_24_hidden_linear",
        )
    if args.optimizer_family == "pythia_adam":
        payload.update(
            adam_lr=args.adam_lr,
            adam_betas=[args.adam_beta1, args.adam_beta2],
            adam_eps=args.adam_eps,
            adam_weight_decay=args.adam_weight_decay,
        )
    payload["invariant_hash"] = config_hash(payload)
    return payload


def save_state(
    path: Path,
    model,
    matrix_optimizer,
    aux_optimizer,
    next_update: int,
    config: dict,
    adaptive_controller: AdaptivePionController | None = None,
):
    payload = {
        "format": FORMAT,
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "matrix_optimizer": matrix_optimizer.state_dict(),
        "aux_optimizer": None if aux_optimizer is None else aux_optimizer.state_dict(),
        "next_update": next_update,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "config": config,
        "adaptive_controller": (
            adaptive_controller.state_dict() if adaptive_controller is not None else None
        ),
    }
    atomic_save(path, payload)


def load_state(
    path: Path,
    model,
    matrix_optimizer,
    aux_optimizer,
    *,
    invariants: dict,
    requested_schedule,
    allow_spectral_fork: bool,
    expected_fork_step: int,
    adaptive_controller: AdaptivePionController | None = None,
) -> tuple[int, dict]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')}")
    source = payload["config"]
    source_invariants = dict(source["invariants"])
    claimed_hash = source_invariants.pop("invariant_hash")
    if config_hash(source_invariants) != claimed_hash:
        raise ValueError("checkpoint invariant hash is invalid")
    requested_invariants = dict(invariants)
    requested_invariants.pop("invariant_hash")
    if source_invariants != requested_invariants or claimed_hash != invariants["invariant_hash"]:
        raise ValueError("resume invariants differ")
    next_update = int(payload["next_update"])
    if expected_fork_step and next_update != expected_fork_step:
        raise ValueError(f"expected fork step {expected_fork_step}, found {next_update}")
    source_family = source.get("optimizer_family", "matched")
    requested_family = invariants.get("optimizer_family", "matched")
    if source_family != requested_family:
        raise ValueError("checkpoint optimizer family differs")
    source_schedule = None if source_family == "pythia_adam" else parse_schedule(source["spectral_schedule"])
    if allow_spectral_fork and source_family == "matched":
        for update in range(next_update):
            if segment_at(source_schedule, update).backend != "muon5ns":
                raise ValueError("fork source prefix was not entirely Muon 5NS")
            if segment_at(requested_schedule, update).backend != "muon5ns":
                raise ValueError("fork target schedule differs before the fork step")
    elif source_schedule != requested_schedule:
        raise ValueError("resume schedule differs without --allow-spectral-fork")
    model.load_state_dict(payload["model"])
    matrix_optimizer.load_state_dict(payload["matrix_optimizer"])
    if (aux_optimizer is None) != (payload["aux_optimizer"] is None):
        raise ValueError("checkpoint auxiliary optimizer presence differs")
    if aux_optimizer is not None:
        aux_optimizer.load_state_dict(payload["aux_optimizer"])
    if source_family == "matched":
        matrix_optimizer.set_schedule(requested_schedule)
    source_controller = payload.get("adaptive_controller")
    if (adaptive_controller is None) != (source_controller is None):
        raise ValueError("checkpoint adaptive-controller presence differs")
    if adaptive_controller is not None:
        adaptive_controller.load_state_dict(source_controller, matrix_optimizer)
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    return next_update, source


def stack_rows(dataset, indices, device):
    rows = [dataset[index] for index in indices]
    x = torch.stack([row[0] for row in rows]).to(device, non_blocking=True)
    y = torch.stack([row[1] for row in rows]).to(device, non_blocking=True)
    extras = [] if not isinstance(dataset, CoreBTokenDataset) else [
        torch.stack([row[column] for row in rows]).to(device, non_blocking=True)
        for column in range(2, 4)
    ]
    return (x, y, *extras)


@torch.inference_mode()
def evaluate_ce(model, dataset, device, batch_size=8, max_batches=0):
    model.eval()
    loss_sum = 0.0
    valid_tokens = 0
    batches = math.ceil(len(dataset) / batch_size)
    if max_batches:
        batches = min(batches, max_batches)
    for batch_index in range(batches):
        indices = range(batch_index * batch_size, min(len(dataset), (batch_index + 1) * batch_size))
        batch = stack_rows(dataset, indices, device)
        x, y = batch[:2]
        kwargs = {"attention_mask": batch[2], "position_ids": batch[3]} if len(batch) == 4 else {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x, **kwargs).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")
        loss_sum += float(loss)
        valid_tokens += int((y != -100).sum())
    model.train()
    return loss_sum / max(1, valid_tokens), valid_tokens


def maybe_optimizer_hash(optimizer) -> str:
    return "none" if optimizer is None else optimizer_hash(optimizer)


def lora_factor_diagnostic(parameter, optimizer, raw_gradient, segment):
    gradient = parameter.grad.detach().float()
    state = optimizer.state[parameter]
    previous = state.get("momentum_buffer")
    if previous is None:
        previous = torch.zeros_like(gradient)
    momentum = float(optimizer.param_groups[0]["momentum"])
    backend_input = previous * momentum + gradient
    transformed = spectral_transform(
        backend_input,
        segment.backend,
        segment.promotion_steps,
        segment.suppression_steps,
        segment.alpha,
    )
    return {
        "raw_gradient": raw_gradient,
        "backend_input": backend_input.detach().cpu().clone(),
        "transformed": transformed.detach().float().cpu().clone(),
    }


def main():
    args = parse_args()
    args.cache_dir = args.cache_dir.resolve()
    if args.adaptive_pion_config is not None:
        args.adaptive_pion_config = args.adaptive_pion_config.resolve()
    output = args.output_dir.resolve()
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    supported_formats = {
        "pythia31m_smoltalk_only_cache_v1",
        "pythia31m_openwebmath_cpt_cache_v1",
        "pythia31m_openwebmath_wikitext_replay_cache_v1",
        "pythia31m_multidataset_cpt_cache_v1",
        "pythia31m_openwebmath_pile_replay_rho_cache_v1",
        "pythia31m_coreb_interference_cache_v1",
        "pythia31m_ord_full_sequence_v1",
        "pythia31m_paired_real_task_domain_v1",
    }
    if manifest.get("format") not in supported_formats:
        raise ValueError(f"unsupported cache format: {manifest.get('format')!r}")
    is_coreb = manifest.get("format") == "pythia31m_coreb_interference_cache_v1"
    is_ord = manifest.get("format") == "pythia31m_ord_full_sequence_v1"
    if is_ord and manifest.get("loss_mask") != "all_1023_targets":
        raise ValueError("ORD cache must use all 1023 causal targets")
    if not is_coreb and not is_ord and manifest.get("loss_mask") != "none_all_tokens":
        raise ValueError("cache must use all-token causal language modeling")
    if is_coreb and manifest.get("loss_mask") != "continuation_512_only":
        raise ValueError("Core B cache must use exactly 512 continuation targets")
    train_template = manifest.get("train_file_template", "train_seed{seed}_tokens.npy")
    if "{rho_label}" in train_template and not args.rho_label:
        raise ValueError("this cache requires --rho-label")
    train_filename = train_template.format(seed=args.seed, rho_label=args.rho_label)
    required_cache_files = {"validation_tokens.npy", "retention_tokens.npy"}
    if is_coreb:
        required_cache_files.update(f"{train_filename}_{suffix}.npy" for suffix in ("body", "distractor", "source_ids"))
        for prefix in ("clean_validation", "matched_validation"):
            required_cache_files.update(f"{prefix}_{suffix}.npy" for suffix in ("body", "distractor"))
    elif is_ord:
        required_cache_files.update((train_filename, "train_seed42_source_ids.npy", "clean_new_tokens.npy"))
        for prefix in ("ord_train", "ord_validation"):
            required_cache_files.update((f"{prefix}_tokens.npy", f"{prefix}_lengths.npy"))
    else:
        required_cache_files.add(train_filename)
    for template_key in ("train_source_ids_template", "train_component_ids_template"):
        if template_key in manifest:
            required_cache_files.add(
                manifest[template_key].format(seed=args.seed, rho_label=args.rho_label)
            )
    required_cache_files.update(manifest.get("secondary_validations", {}).values())
    for name in sorted(required_cache_files):
        expected = manifest["file_sha256"].get(name)
        if expected is None or sha256_file(args.cache_dir / name) != expected:
            raise RuntimeError(f"cache hash mismatch or missing manifest entry: {name}")
    if args.seed not in manifest["seeds"]:
        raise ValueError(f"seed {args.seed} is absent from cache")
    if args.device_batch_size * args.grad_accum_steps != 32:
        raise ValueError("global sequence batch must remain exactly 32")
    if args.warmup_updates is not None and args.warmup_updates <= 0:
        raise ValueError("--warmup-updates must be positive")
    if (
        args.warmup_updates is not None
        and args.warmup_updates > args.total_updates
        and not args.smoke
    ):
        raise ValueError("formal --warmup-updates cannot exceed --total-updates")
    expected_updates = int(manifest.get("training_updates", 1000))
    if args.total_updates != expected_updates and not args.smoke:
        raise ValueError(f"formal runs require exactly {expected_updates} updates")
    stop_after = args.stop_after or args.total_updates
    if not 0 < stop_after <= args.total_updates:
        raise ValueError("stop-after must be in (0, total-updates]")
    if args.optimizer_family == "matched":
        schedule = parse_schedule(args.spectral_schedule)
        if any(segment.start >= args.total_updates for segment in schedule[1:]):
            raise ValueError("spectral switch lies outside training horizon")
    else:
        if args.spectral_schedule != "none":
            raise ValueError("Pythia Adam requires --spectral-schedule none")
        schedule = None
    adaptive_config = None
    adaptive_config_file_sha256 = ""
    if args.adaptive_pion_config is not None:
        if args.optimizer_family != "matched" or args.training_mode != "full":
            raise ValueError("Adaptive Pion requires full-parameter matched-optimizer training")
        if any(segment.backend != "muon5ns" for segment in schedule):
            raise ValueError("Adaptive Pion base schedule must remain Muon 5NS")
        adaptive_config, adaptive_config_file_sha256 = load_controller_config(
            args.adaptive_pion_config
        )
        if adaptive_config["warmup_updates"] >= args.total_updates:
            raise ValueError("adaptive warmup must end before the training horizon")
    if args.training_mode == "lora":
        if args.optimizer_family != "matched":
            raise ValueError("LoRA mode is only supported by the matched optimizer")
        if args.lora_rank <= 0 or args.lora_alpha <= 0:
            raise ValueError("LoRA mode requires positive --lora-rank and --lora-alpha")
        if len(schedule) != 1 or schedule[0].backend != "muon5ns":
            raise ValueError("factor-wise LoRA experiment requires full-run Muon 5NS")
    elif args.lora_rank != 0 or args.lora_alpha != 0.0:
        raise ValueError("LoRA arguments require --training-mode lora")

    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "checkpoints"
    spectra = output / "spectra"
    checkpoints.mkdir()
    spectra.mkdir()
    events = output / "events.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    train_data = (CoreBTokenDataset(args.cache_dir, train_filename, k=args.interference_k)
                  if is_coreb else ORDFullSequenceDataset(args.cache_dir, train_filename.removesuffix("_tokens.npy"), k=args.interference_k, seed=args.seed)
                  if is_ord else TokenWindowDataset(args.cache_dir / train_filename, context=manifest["context"]))
    monitor_data = TokenWindowDataset(
        args.cache_dir / "validation_tokens.npy",
        context=manifest["context"],
        max_rows=args.monitor_rows,
    )
    full_validation = monitor_data if args.smoke else TokenWindowDataset(
        args.cache_dir / "validation_tokens.npy", context=manifest["context"]
    )
    retention_data = TokenWindowDataset(
        args.cache_dir / "retention_tokens.npy",
        context=manifest["context"],
        max_rows=(16 if args.smoke else 0),
    )
    retention_monitor_data = (
        TokenWindowDataset(
            args.cache_dir / "retention_tokens.npy",
            context=manifest["context"],
            max_rows=(16 if args.smoke else args.monitor_rows),
        )
        if args.monitor_retention_every_validation
        else None
    )
    secondary_data = {
        name: TokenWindowDataset(
            args.cache_dir / filename,
            context=manifest["context"],
            max_rows=(16 if args.smoke else 0),
        )
        for name, filename in manifest.get("secondary_validations", {}).items()
    }
    if is_coreb:
        secondary_data.update(
            clean_continuation=CoreBTokenDataset(args.cache_dir, "clean_validation", k=0, max_rows=(16 if args.smoke else 0)),
            matched_interference=CoreBTokenDataset(args.cache_dir, "matched_validation", k=args.interference_k, max_rows=(16 if args.smoke else 0)),
        )
    if is_ord:
        secondary_data.update(
            clean_new=ORDFullSequenceDataset(args.cache_dir, "clean_new", k=0, validation=True, max_rows=(16 if args.smoke else 0)),
            noisy_new=ORDFullSequenceDataset(args.cache_dir, "clean_new", k=args.interference_k, validation=True, max_rows=(16 if args.smoke else 0)),
            clean_continuation=ORDFullSequenceDataset(args.cache_dir, "clean_new", k=0, validation=True, continuation_only=True, max_rows=(16 if args.smoke else 0)),
        )
    expected_rows = args.total_updates * args.device_batch_size * args.grad_accum_steps
    if len(train_data) != expected_rows and not args.smoke:
        raise ValueError(f"expected {expected_rows} training rows, found {len(train_data)}")

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["tokenizer_id"], revision=manifest["tokenizer_revision"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_id"],
        revision=manifest["model_revision"],
        torch_dtype=torch.float32,
    ).to(device)
    pretrained_model_hash = model_hash(model)
    lora = None
    initial_frozen_hash = ""
    if args.training_mode == "lora":
        lora = inject_hidden_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        matrix = lora.parameters
        matrix_names = lora.parameter_names
        auxiliary = []
        auxiliary_names = []
        diagnostic_param = None
        initial_frozen_hash = frozen_state_hash(model)
        actual_trainable = sum(parameter.numel() for parameter in matrix)
        expected_trainable = expected_trainable_parameters(args.lora_rank)
        if actual_trainable != expected_trainable:
            raise RuntimeError(
                f"LoRA trainable count mismatch: expected {expected_trainable}, found {actual_trainable}"
            )
    else:
        matrix, auxiliary, matrix_names, auxiliary_names, diagnostic_param = parameter_partition(model)
    if args.optimizer_family == "matched":
        matrix_optimizer = MatchedMuon(
            matrix,
            lr=args.matrix_lr,
            momentum=args.matrix_momentum,
            schedule=schedule,
            diagnostic_param=diagnostic_param,
            diagnostic_name=DIAGNOSTIC_PARAM,
            parameter_names=matrix_names,
        )
        aux_optimizer = (
            torch.optim.AdamW(auxiliary, lr=args.aux_lr, weight_decay=args.aux_weight_decay)
            if auxiliary
            else None
        )
    else:
        named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        matrix_names = [name for name, _ in named_trainable]
        matrix = [parameter for _, parameter in named_trainable]
        auxiliary_names = []
        auxiliary = []
        matrix_optimizer = torch.optim.AdamW(
            matrix,
            lr=args.adam_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.adam_weight_decay,
        )
        aux_optimizer = None
    base_model_hash = pretrained_model_hash if lora is not None else model_hash(model)
    adaptive_controller = (
        AdaptivePionController(adaptive_config, matrix_names)
        if adaptive_config is not None
        else None
    )
    invariants = invariant_config(args, manifest, adaptive_config)
    run_config = {
        "event": "config",
        "invariants": invariants,
        "spectral_schedule": args.spectral_schedule,
        "stop_after": stop_after,
        "gpu_binding": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "eval_every": args.eval_every,
        "validation_steps": sorted(set(range(0, args.total_updates + 1, args.eval_every)) | {args.total_updates}),
        "monitor_rows": args.monitor_rows,
        "monitor_retention_every_validation": args.monitor_retention_every_validation,
        "full_validation_steps": sorted(parse_int_set(args.full_validation_steps)),
        "retention_steps": sorted(parse_int_set(args.retention_steps)),
        "save_steps": sorted(parse_int_set(args.save_steps)),
        "diagnostic_updates": sorted(parse_int_set(args.diagnostic_updates)),
        "base_model_hash": base_model_hash,
        "train_fingerprint": coreb_fingerprint(train_data) if is_coreb else dataset_fingerprint(train_data),
        "monitor_fingerprint": dataset_fingerprint(monitor_data),
        "full_validation_fingerprint": dataset_fingerprint(full_validation),
        "retention_fingerprint": dataset_fingerprint(retention_data),
        "retention_monitor_fingerprint": (
            dataset_fingerprint(retention_monitor_data)
            if retention_monitor_data is not None
            else None
        ),
        "secondary_validation_fingerprints": {
            name: (coreb_fingerprint(dataset) if isinstance(dataset, CoreBTokenDataset) else dataset_fingerprint(dataset)) for name, dataset in secondary_data.items()
        },
        "matrix_param_names": matrix_names,
        "auxiliary_param_names": auxiliary_names,
        "matrix_param_count": sum(parameter.numel() for parameter in matrix),
        "auxiliary_param_count": sum(parameter.numel() for parameter in auxiliary),
        "tokenizer_vocab_size": len(tokenizer),
        "training_mode": args.training_mode,
        "optimizer_family": args.optimizer_family,
        "rho_label": args.rho_label,
        "interference_k": args.interference_k,
        "trainable_param_count": sum(parameter.numel() for parameter in matrix + auxiliary),
        "pretrained_model_hash": pretrained_model_hash,
        "initial_frozen_state_hash": initial_frozen_hash,
        "adaptive_pion_config": adaptive_config,
        "adaptive_pion_config_path": (
            str(args.adaptive_pion_config) if args.adaptive_pion_config is not None else None
        ),
        "adaptive_pion_config_file_sha256": adaptive_config_file_sha256 or None,
    }
    if lora is not None:
        run_config.update(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_scaling=args.lora_alpha / args.lora_rank,
            lora_target_module_count=len(lora.modules),
            lora_parameter_count=len(lora.parameters),
            lora_diagnostic_module=DIAGNOSTIC_MODULE,
        )
    append_event(events, run_config)
    json_dump(output / "config.json", run_config)

    start_update = 0
    source_config = None
    if args.resume:
        start_update, source_config = load_state(
            args.resume.resolve(),
            model,
            matrix_optimizer,
            aux_optimizer,
            invariants=invariants,
            requested_schedule=schedule,
            allow_spectral_fork=args.allow_spectral_fork,
            expected_fork_step=args.expected_fork_step,
            adaptive_controller=adaptive_controller,
        )
        append_event(
            events,
            {
                "event": "resume",
                "next_update": start_update,
                "source_checkpoint": str(args.resume.resolve()),
                "source_checkpoint_sha256": sha256_file(args.resume.resolve()),
                "source_spectral_schedule": source_config["spectral_schedule"],
                "requested_spectral_schedule": args.spectral_schedule,
                "model_hash": model_hash(model),
                "matrix_optimizer_hash": optimizer_hash(matrix_optimizer),
                "aux_optimizer_hash": maybe_optimizer_hash(aux_optimizer),
            },
        )

    train_model = torch.compile(model, dynamic=False) if args.compile else model
    retention_steps = parse_int_set(args.retention_steps)
    full_validation_steps = parse_int_set(args.full_validation_steps)
    save_steps = parse_int_set(args.save_steps)
    diagnostic_updates = parse_int_set(args.diagnostic_updates)
    validation_steps = set(range(0, args.total_updates + 1, args.eval_every)) | {args.total_updates}

    primary_name = manifest.get("primary_validation_name", "smoltalk")
    adaptive_dir = output / "adaptive"
    if adaptive_controller is not None:
        adaptive_dir.mkdir(exist_ok=True)

    def validate(step: int, event: str = "validation"):
        monitor_ce, monitor_tokens = evaluate_ce(train_model, monitor_data, device)
        payload = {
            "event": event,
            "step": step,
            "primary_name": primary_name,
            "primary_ce": monitor_ce,
            "primary_tokens": monitor_tokens,
            "model_hash": model_hash(model),
        }
        if primary_name == "smoltalk":
            payload.update(smoltalk_monitor_ce=monitor_ce, smoltalk_monitor_tokens=monitor_tokens)
        elif primary_name == "openwebmath":
            payload.update(openwebmath_ce=monitor_ce, openwebmath_tokens=monitor_tokens)
        else:
            payload.update(**{f"{primary_name}_ce": monitor_ce, f"{primary_name}_tokens": monitor_tokens})
        if retention_monitor_data is not None:
            retention_monitor_ce, retention_monitor_tokens = evaluate_ce(
                train_model, retention_monitor_data, device
            )
            payload.update(
                retention_monitor_ce=retention_monitor_ce,
                retention_monitor_tokens=retention_monitor_tokens,
            )
        if step in full_validation_steps:
            full_primary_ce, full_primary_tokens = evaluate_ce(
                train_model, full_validation, device
            )
            full_retention_ce, full_retention_tokens = evaluate_ce(
                train_model, retention_data, device
            )
            payload.update(
                full_primary_ce=full_primary_ce,
                full_primary_tokens=full_primary_tokens,
                full_retention_ce=full_retention_ce,
                full_retention_tokens=full_retention_tokens,
                retention_ce=full_retention_ce,
                retention_tokens=full_retention_tokens,
            )
            for name, dataset in secondary_data.items():
                secondary_ce, secondary_tokens = evaluate_ce(train_model, dataset, device)
                payload.update(
                    **{
                        f"full_{name}_ce": secondary_ce,
                        f"full_{name}_tokens": secondary_tokens,
                        f"{name}_ce": secondary_ce,
                        f"{name}_tokens": secondary_tokens,
                    }
                )
        elif step in retention_steps:
            retention_ce, retention_tokens = evaluate_ce(train_model, retention_data, device)
            payload.update(retention_ce=retention_ce, retention_tokens=retention_tokens)
            for name, dataset in secondary_data.items():
                secondary_ce, secondary_tokens = evaluate_ce(train_model, dataset, device)
                payload.update(
                    **{f"{name}_ce": secondary_ce, f"{name}_tokens": secondary_tokens}
                )
        append_event(events, payload)
        return payload

    if start_update == 0:
        validate(0)
        if 0 in save_steps:
            checkpoint_config = {
                "invariants": invariants,
                "spectral_schedule": args.spectral_schedule,
                "optimizer_family": args.optimizer_family,
            }
            save_state(
                checkpoints / "state_step000000.pt",
                model,
                matrix_optimizer,
                aux_optimizer,
                0,
                checkpoint_config,
                adaptive_controller,
            )
    else:
        validate(start_update, "fork_validation")

    current_segment = None
    started = time.perf_counter()
    cumulative_tokens = 0
    for update in range(start_update, stop_after):
        segment = (
            segment_at(schedule, update)
            if args.optimizer_family == "matched"
            else SimpleNamespace(
                backend="adamw", promotion_steps=0, suppression_steps=0, alpha=0.0
            )
        )
        if segment != current_segment:
            append_event(
                events,
                {
                    "event": "spectral_switch",
                    "update": update,
                    "backend": segment.backend,
                    "promotion_steps": segment.promotion_steps,
                    "suppression_steps": segment.suppression_steps,
                    "alpha": segment.alpha,
                },
            )
            current_segment = segment
        multiplier = schedule_lr(update, args.total_updates, args.warmup_updates)
        matrix_optimizer.param_groups[0]["lr"] = (
            args.matrix_lr if args.optimizer_family == "matched" else args.adam_lr
        ) * multiplier
        if aux_optimizer is not None:
            for group in aux_optimizer.param_groups:
                group["lr"] = args.aux_lr * multiplier
        matrix_optimizer.zero_grad(set_to_none=True)
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)
        first_row = update * 32
        row_indices = list(range(first_row, first_row + 32))
        objective = 0.0
        observe_adaptive = adaptive_controller is not None and adaptive_controller.should_observe(
            update, args.total_updates
        )
        previous_matrix_grads = (
            {
                name: torch.zeros_like(parameter, dtype=torch.float32)
                for name, parameter in zip(matrix_names, matrix)
            }
            if observe_adaptive
            else None
        )
        for micro_start in range(0, 32, args.device_batch_size):
            indices = row_indices[micro_start : micro_start + args.device_batch_size]
            batch = stack_rows(train_data, indices, device)
            x, y = batch[:2]
            model_kwargs = {"attention_mask": batch[2], "position_ids": batch[3]} if len(batch) == 4 else {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(input_ids=x, **model_kwargs).logits
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="sum",
                )
                loss = loss_sum / (32 * int(manifest.get("supervised_tokens_per_row", manifest["context"])))
            loss.backward()
            objective += float(loss_sum.detach())
            if observe_adaptive:
                incremental = {}
                for name, parameter in zip(matrix_names, matrix):
                    current = parameter.grad.detach().float()
                    incremental[name] = (current - previous_matrix_grads[name]).cpu()
                    previous_matrix_grads[name] = current.clone()
                adaptive_controller.observe_microbatch(incremental)

        capture = update in diagnostic_updates
        raw_gradient = None
        lora_capture = None
        if capture:
            if lora is None:
                raw_gradient = diagnostic_param.grad.detach().float().cpu().clone()
                if args.optimizer_family == "matched":
                    matrix_optimizer.request_diagnostic(raw_gradient)
            else:
                adapter = lora.diagnostic
                lora_capture = {
                    "raw_A": adapter.lora_A.grad.detach().float().cpu().clone(),
                    "raw_B": adapter.lora_B.grad.detach().float().cpu().clone(),
                    "before_A": adapter.lora_A.detach().float().clone(),
                    "before_B": adapter.lora_B.detach().float().clone(),
                }
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0))
        if capture and lora is not None:
            lora_capture["factor_A"] = lora_factor_diagnostic(
                lora.diagnostic.lora_A,
                matrix_optimizer,
                lora_capture["raw_A"],
                segment,
            )
            lora_capture["factor_B"] = lora_factor_diagnostic(
                lora.diagnostic.lora_B,
                matrix_optimizer,
                lora_capture["raw_B"],
                segment,
            )
        before = diagnostic_param.detach().float().clone() if capture and lora is None else None
        matrix_optimizer.step()
        if aux_optimizer is not None:
            aux_optimizer.step()
        model_step = update + 1
        if (
            adaptive_controller is not None
            and model_step in adaptive_controller.decision_steps(args.total_updates)
        ):
            adaptive_event = adaptive_controller.decide(model_step, matrix_optimizer)
            adaptive_path = adaptive_dir / f"decision_step{model_step:06d}.json"
            json_dump(adaptive_path, adaptive_event)
            append_event(
                events,
                {
                    "event": "adaptive_pion_decision",
                    "step": model_step,
                    "decision_index": adaptive_event["decision_index"],
                    "signal": adaptive_event["signal"],
                    "granularity": adaptive_event["granularity"],
                    "tau": adaptive_event["tau"],
                    "elapsed_seconds": adaptive_event["elapsed_seconds"],
                    "group_applied_levels": {
                        row["group"]: row["applied_level"]
                        for row in adaptive_event["decisions"]
                    },
                    "artifact": str(adaptive_path),
                },
            )
        valid_tokens = 32 * int(manifest.get("supervised_tokens_per_row", manifest["context"]))
        cumulative_tokens += valid_tokens
        append_event(
            events,
            {
                "event": "train",
                "update": update,
                "step": model_step,
                "ce": objective / valid_tokens,
                "valid_tokens": valid_tokens,
                "matrix_lr": args.matrix_lr * multiplier if args.optimizer_family == "matched" else None,
                "optimizer_lr": (
                    args.matrix_lr if args.optimizer_family == "matched" else args.adam_lr
                ) * multiplier,
                "aux_lr": args.aux_lr * multiplier if aux_optimizer is not None else None,
                "aux_optimizer_active": aux_optimizer is not None,
                "grad_norm": grad_norm,
                "backend": segment.backend,
                "promotion_steps": segment.promotion_steps,
                "suppression_steps": segment.suppression_steps,
                "alpha": segment.alpha,
                "adaptive_active_levels": (
                    dict(adaptive_controller.current_levels)
                    if adaptive_controller is not None
                    else None
                ),
                "tokens_per_second": cumulative_tokens / max(1e-9, time.perf_counter() - started),
            },
        )
        if capture:
            artifact_path = spectra / f"spectrum_update{update:06d}.pt"
            if lora is None:
                actual_update = (diagnostic_param.detach().float() - before).cpu()
                if args.optimizer_family == "matched":
                    diagnostic = matrix_optimizer.last_diagnostic
                    artifact = {
                        **diagnostic,
                        "actual_update": actual_update,
                        # Diagnostics are stored on CPU, but doing concurrent SVDs
                        # there oversubscribes the host. Move the matrix route-locally.
                        "raw_summary": summarize_spectrum(diagnostic["raw_gradient"].to(device)),
                        "input_summary": summarize_spectrum(diagnostic["backend_input"].to(device)),
                        "output_summary": summarize_spectrum(diagnostic["transformed"].to(device)),
                        "actual_update_norm": float(actual_update.norm()),
                    }
                else:
                    state = matrix_optimizer.state[diagnostic_param]
                    artifact = {
                        "update_step": update,
                        "param": DIAGNOSTIC_PARAM,
                        "backend": "adamw",
                        "raw_gradient": raw_gradient,
                        "exp_avg": state["exp_avg"].detach().float().cpu().clone(),
                        "exp_avg_sq": state["exp_avg_sq"].detach().float().cpu().clone(),
                        "actual_update": actual_update,
                        "raw_summary": summarize_spectrum(raw_gradient.to(device)),
                        "actual_update_summary": summarize_spectrum(actual_update.to(device)),
                        "actual_update_norm": float(actual_update.norm()),
                    }
                torch.save(artifact, artifact_path)
                spectral_event = {
                        "event": "spectral",
                        "update": update,
                        "model_step": model_step,
                        "route_backend": segment.backend,
                        "backend": artifact["backend"],
                        "promotion_steps": artifact.get("promotion_steps", 0),
                        "suppression_steps": artifact.get("suppression_steps", 0),
                        "alpha": artifact.get("alpha", 0.0),
                        "actual_update_norm": artifact["actual_update_norm"],
                        "raw_gradient_hash": tensor_hash(raw_gradient),
                        "artifact": str(artifact_path),
                }
                if args.optimizer_family == "matched":
                    spectral_event.update(
                        input_rank_gt_0p5=artifact["input_summary"]["rank_gt_0p5"],
                        output_rank_gt_0p5=artifact["output_summary"]["rank_gt_0p5"],
                        input_energy_rank_95=artifact["input_summary"]["energy_rank_95"],
                        output_energy_rank_95=artifact["output_summary"]["energy_rank_95"],
                        input_stable_rank=artifact["input_summary"]["stable_rank"],
                        output_stable_rank=artifact["output_summary"]["stable_rank"],
                        input_fro=artifact["input_summary"]["fro"],
                        output_fro=artifact["output_summary"]["fro"],
                        backend_input_hash=tensor_hash(artifact["backend_input"]),
                    )
                else:
                    spectral_event.update(
                        actual_update_rank_gt_0p5=artifact["actual_update_summary"]["rank_gt_0p5"],
                        actual_update_energy_rank_95=artifact["actual_update_summary"]["energy_rank_95"],
                        actual_update_stable_rank=artifact["actual_update_summary"]["stable_rank"],
                    )
                append_event(events, spectral_event)
            else:
                adapter = lora.diagnostic
                after_A = adapter.lora_A.detach().float()
                after_B = adapter.lora_B.detach().float()
                actual_A = (after_A - lora_capture["before_A"]).cpu()
                actual_B = (after_B - lora_capture["before_B"]).cpu()
                effective_before = adapter.scaling * (
                    lora_capture["before_B"] @ lora_capture["before_A"]
                )
                effective_after = adapter.scaling * (after_B @ after_A)
                effective_update = (effective_after - effective_before).cpu()
                for factor in ("factor_A", "factor_B"):
                    values = lora_capture[factor]
                    values["raw_summary"] = summarize_spectrum(values["raw_gradient"].to(device))
                    values["input_summary"] = summarize_spectrum(values["backend_input"].to(device))
                    values["output_summary"] = summarize_spectrum(values["transformed"].to(device))
                effective_summary = summarize_spectrum(effective_update.to(device))
                algebraic_rank = int(torch.linalg.matrix_rank(effective_update.to(device)).item())
                artifact = {
                    "update_step": update,
                    "param": DIAGNOSTIC_MODULE,
                    "backend": segment.backend,
                    "promotion_steps": segment.promotion_steps,
                    "suppression_steps": segment.suppression_steps,
                    "alpha": segment.alpha,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "lora_scaling": adapter.scaling,
                    "factor_A": lora_capture["factor_A"],
                    "factor_B": lora_capture["factor_B"],
                    "actual_A_update": actual_A,
                    "actual_B_update": actual_B,
                    "effective_weight_update": effective_update,
                    "effective_summary": effective_summary,
                    "effective_algebraic_rank": algebraic_rank,
                    "actual_update_norm": float(effective_update.norm()),
                }
                torch.save(artifact, artifact_path)
                append_event(
                    events,
                    {
                        "event": "spectral",
                        "update": update,
                        "model_step": model_step,
                        "route_backend": segment.backend,
                        "backend": segment.backend,
                        "promotion_steps": segment.promotion_steps,
                        "suppression_steps": segment.suppression_steps,
                        "alpha": segment.alpha,
                        "factor_A_input_rank_gt_0p5": artifact["factor_A"]["input_summary"]["rank_gt_0p5"],
                        "factor_A_output_rank_gt_0p5": artifact["factor_A"]["output_summary"]["rank_gt_0p5"],
                        "factor_B_input_rank_gt_0p5": artifact["factor_B"]["input_summary"]["rank_gt_0p5"],
                        "factor_B_output_rank_gt_0p5": artifact["factor_B"]["output_summary"]["rank_gt_0p5"],
                        "effective_rank_gt_0p5": effective_summary["rank_gt_0p5"],
                        "effective_energy_rank_95": effective_summary["energy_rank_95"],
                        "effective_stable_rank": effective_summary["stable_rank"],
                        "effective_algebraic_rank": algebraic_rank,
                        "actual_update_norm": artifact["actual_update_norm"],
                        "raw_gradient_A_hash": tensor_hash(artifact["factor_A"]["raw_gradient"]),
                        "raw_gradient_B_hash": tensor_hash(artifact["factor_B"]["raw_gradient"]),
                        "backend_input_A_hash": tensor_hash(artifact["factor_A"]["backend_input"]),
                        "backend_input_B_hash": tensor_hash(artifact["factor_B"]["backend_input"]),
                        "artifact": str(artifact_path),
                    },
                )
        if model_step in validation_steps:
            validate(model_step)
        if model_step in save_steps or model_step == stop_after:
            checkpoint_config = {
                "invariants": invariants,
                "spectral_schedule": args.spectral_schedule,
                "optimizer_family": args.optimizer_family,
            }
            save_state(
                checkpoints / f"state_step{model_step:06d}.pt",
                model,
                matrix_optimizer,
                aux_optimizer,
                model_step,
                checkpoint_config,
                adaptive_controller,
            )

    if stop_after < args.total_updates:
        append_event(
            events,
            {
                "event": "prefix_complete",
                "step": stop_after,
                "model_hash": model_hash(model),
                "matrix_optimizer_hash": optimizer_hash(matrix_optimizer),
                "aux_optimizer_hash": maybe_optimizer_hash(aux_optimizer),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return

    full_ce, full_tokens = evaluate_ce(train_model, full_validation, device)
    retention_ce, retention_tokens = evaluate_ce(train_model, retention_data, device)
    secondary_complete = {}
    for name, dataset in secondary_data.items():
        secondary_ce, secondary_tokens = evaluate_ce(train_model, dataset, device)
        secondary_complete[f"full_{name}_ce"] = secondary_ce
        secondary_complete[f"full_{name}_tokens"] = secondary_tokens
    complete = {
        "event": "complete",
        "step": args.total_updates,
        "primary_name": primary_name,
        "full_primary_ce": full_ce,
        "full_primary_tokens": full_tokens,
        "full_retention_ce": retention_ce,
        "full_retention_tokens": retention_tokens,
        "final_model_hash": model_hash(model),
        "matrix_optimizer_hash": optimizer_hash(matrix_optimizer),
        "aux_optimizer_hash": maybe_optimizer_hash(aux_optimizer),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "adaptive_decision_count": (
            adaptive_controller.decision_count if adaptive_controller is not None else 0
        ),
        **secondary_complete,
    }
    if lora is not None:
        final_frozen_hash = frozen_state_hash(model)
        complete.update(
            initial_frozen_state_hash=initial_frozen_hash,
            final_frozen_state_hash=final_frozen_hash,
            frozen_base_unchanged=final_frozen_hash == initial_frozen_hash,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_scaling=args.lora_alpha / args.lora_rank,
        )
    if primary_name == "smoltalk":
        complete.update(full_smoltalk_ce=full_ce, full_smoltalk_tokens=full_tokens)
    elif primary_name == "openwebmath":
        complete.update(full_openwebmath_ce=full_ce, full_openwebmath_tokens=full_tokens)
    else:
        complete.update(**{f"full_{primary_name}_ce": full_ce, f"full_{primary_name}_tokens": full_tokens})
    append_event(events, complete)
    json_dump(output / "complete.json", complete)


if __name__ == "__main__":
    main()
