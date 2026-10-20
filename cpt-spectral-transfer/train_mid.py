#!/usr/bin/env python3
"""One-epoch Pythia-31M midtraining with a polynomial-only Muon/Pion fork."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from middata import ContiguousDataset, PackedDataset, dataset_fingerprint, sha256_file
from spectral import MatchedMuon, parse_schedule, segment_at, summarize_spectrum


DIAGNOSTIC_PARAM = "gpt_neox.layers.0.mlp.dense_4h_to_h.weight"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_v1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="EleutherAI/pythia-31m")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-id", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--spectral-schedule", required=True)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--matrix-lr", type=float, default=3e-3)
    parser.add_argument("--aux-lr", type=float, default=1e-3)
    parser.add_argument("--matrix-momentum", type=float, default=0.95)
    parser.add_argument("--aux-weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-updates", type=int, default=0, help="0 means one complete epoch")
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--diagnostic-every", type=int, default=0)
    parser.add_argument("--audit-steps", default="")
    parser.add_argument("--monitor-retention-batches", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(path: Path, payload: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def config_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parameter_partition(model):
    matrix = []
    auxiliary = []
    matrix_names = []
    auxiliary_names = []
    diagnostic = None
    for name, parameter in model.named_parameters():
        is_hidden_matrix = (
            parameter.ndim == 2
            and name.startswith("gpt_neox.layers.")
            and name.endswith(".weight")
        )
        if is_hidden_matrix:
            matrix.append(parameter)
            matrix_names.append(name)
            if name == DIAGNOSTIC_PARAM:
                diagnostic = parameter
        else:
            auxiliary.append(parameter)
            auxiliary_names.append(name)
    all_ids = [id(p) for p in matrix + auxiliary]
    if len(all_ids) != len(set(all_ids)) or len(all_ids) != len(list(model.parameters())):
        raise RuntimeError("optimizer parameter partition is incomplete or overlapping")
    if diagnostic is None:
        raise RuntimeError(f"diagnostic parameter not found: {DIAGNOSTIC_PARAM}")
    return matrix, auxiliary, matrix_names, auxiliary_names, diagnostic


def rows_per_update(args) -> int:
    return args.device_batch_size * args.grad_accum_steps


def update_count(dataset, args) -> int:
    return math.ceil(len(dataset) / rows_per_update(args))


def schedule_lr(step: int, total: int) -> float:
    warmup = max(1, round(total * 0.10))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def stack_batch(dataset, indices, device):
    examples = [dataset[index] for index in indices]
    x = torch.stack([item[0] for item in examples]).to(device, non_blocking=True)
    y = torch.stack([item[1] for item in examples]).to(device, non_blocking=True)
    return x, y


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
        x, y = stack_batch(dataset, indices, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        loss_sum += float(loss)
        valid_tokens += int((y != -100).sum())
    model.train()
    return loss_sum / max(1, valid_tokens), valid_tokens


def format_prompt(question: str, eos: str) -> str:
    return f"{eos}User: {question}\nAssistant:"


@torch.inference_mode()
def evaluate_mmlu(model, tokenizer, records, device, context):
    model.eval()
    letters = "ABCD"
    correct = 0
    for record in records:
        prompt_ids = tokenizer(
            format_prompt(record["prompt"], tokenizer.eos_token),
            add_special_tokens=False,
        )["input_ids"][-(context - 4) :]
        scores = []
        for letter in letters:
            answer_ids = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
            ids = torch.tensor([prompt_ids + answer_ids], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids[:, :-1]).logits.float()
            target = ids[:, 1:]
            token_loss = F.cross_entropy(
                logits[:, -len(answer_ids) :].reshape(-1, logits.shape[-1]),
                target[:, -len(answer_ids) :].reshape(-1),
                reduction="mean",
            )
            scores.append(-float(token_loss))
        correct += letters[int(np.argmax(scores))] == record["answer"]
    model.train()
    return correct / max(1, len(records))


GSM_RE = re.compile(r"####\s*(-?[0-9.,]+)")


@torch.inference_mode()
def evaluate_gsm8k(model, tokenizer, records, device, context, max_new_tokens=128):
    model.eval()
    correct = 0
    for record in records:
        prompt = format_prompt(record["question"], tokenizer.eos_token)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context - max_new_tokens)
        input_ids = inputs.input_ids.to(device)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=False)
        expected = GSM_RE.search(record["answer"])
        predicted = GSM_RE.search(response)
        if expected and predicted:
            correct += expected.group(1).replace(",", "") == predicted.group(1).replace(",", "")
    model.train()
    return correct / max(1, len(records))


def optimizer_hash(optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    for key in sorted(state["state"]):
        digest.update(str(key).encode())
        for item_key, value in sorted(state["state"][key].items()):
            digest.update(item_key.encode())
            if torch.is_tensor(value):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(str(value).encode())
    return digest.hexdigest()


def atomic_save(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, model, matrix_optimizer, aux_optimizer, next_update, config):
    payload = {
        "format": "pythia31m_midtraining_full_state_v1",
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "matrix_optimizer": matrix_optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "next_update": next_update,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
        "config": config,
    }
    atomic_save(path, payload)


def load_checkpoint(path, model, matrix_optimizer, aux_optimizer, expected_config_hash):
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "pythia31m_midtraining_full_state_v1":
        raise ValueError("unsupported checkpoint format")
    if payload["config"]["semantic_config_hash"] != expected_config_hash:
        raise ValueError("resume semantic config differs")
    model.load_state_dict(payload["model"])
    matrix_optimizer.load_state_dict(payload["matrix_optimizer"])
    aux_optimizer.load_state_dict(payload["aux_optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state(payload["cuda_rng"])
    return int(payload["next_update"])


def main():
    args = parse_args()
    cache = args.cache_dir.resolve()
    output = args.output_dir.resolve()
    manifest = json.loads((cache / "manifest.json").read_text())
    for name, expected in manifest["file_sha256"].items():
        if sha256_file(cache / name) != expected:
            raise RuntimeError(f"cache hash mismatch: {name}")
    schedule = parse_schedule(args.spectral_schedule)
    train_data = PackedDataset(cache, "train")
    val_data = PackedDataset(cache, "validation")
    val_sources = {
        source: PackedDataset(cache, f"validation_{source}")
        for source in ("smoltalk", "mmlu", "gsm8k")
    }
    retention_data = ContiguousDataset(cache / "retention_tokens.npy", manifest["context"])
    total_updates = update_count(train_data, args)
    if args.max_updates:
        total_updates = min(total_updates, args.max_updates)
    if any(segment.start >= total_updates for segment in schedule[1:]):
        raise ValueError("spectral schedule contains a switch beyond the training horizon")

    if output.exists() and any(output.iterdir()) and args.resume is None:
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    spectra_dir = output / "spectra"
    checkpoint_dir.mkdir(exist_ok=True)
    spectra_dir.mkdir(exist_ok=True)
    events_path = output / "events.jsonl"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model_revision = args.model_revision or manifest["model_revision"]
    tokenizer_id = args.tokenizer_id or manifest["model_id"]
    tokenizer_revision = args.tokenizer_revision or manifest["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, revision=tokenizer_revision
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=model_revision,
        torch_dtype=torch.float32,
    ).to(device)
    matrix, auxiliary, matrix_names, auxiliary_names, diagnostic_param = parameter_partition(model)
    matrix_optimizer = MatchedMuon(
        matrix,
        lr=args.matrix_lr,
        momentum=args.matrix_momentum,
        schedule=schedule,
        diagnostic_param=diagnostic_param,
        diagnostic_name=DIAGNOSTIC_PARAM,
    )
    aux_optimizer = torch.optim.AdamW(
        auxiliary, lr=args.aux_lr, weight_decay=args.aux_weight_decay
    )
    initial_model_hash = model_hash(model)
    semantic_config = {
        "model_id": args.model_id,
        "model_revision": model_revision,
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "cache_manifest_sha256": sha256_file(cache / "manifest.json"),
        "context": manifest["context"],
        "device_batch_size": args.device_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "total_updates": total_updates,
        "matrix_lr": args.matrix_lr,
        "aux_lr": args.aux_lr,
        "matrix_momentum": args.matrix_momentum,
        "matrix_weight_decay": 0.0,
        "aux_weight_decay": args.aux_weight_decay,
        "seed": args.seed,
        "dtype": "bfloat16",
        "clip_norm": 1.0,
        "compile": args.compile,
        "gpu_binding": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "spectral_schedule": args.spectral_schedule,
        "audit_steps": args.audit_steps,
    }
    semantic_config["semantic_config_hash"] = config_hash(semantic_config)
    config_event = {
        "event": "config",
        **semantic_config,
        "initial_model_hash": initial_model_hash,
        "train_fingerprint": dataset_fingerprint(train_data),
        "validation_fingerprint": dataset_fingerprint(val_data),
        "matrix_param_names": matrix_names,
        "auxiliary_param_names": auxiliary_names,
        "matrix_param_count": sum(p.numel() for p in matrix),
        "auxiliary_param_count": sum(p.numel() for p in auxiliary),
    }
    if not events_path.exists():
        append_event(events_path, config_event)
        json_dump(output / "config.json", config_event)

    start_update = 0
    if args.resume:
        start_update = load_checkpoint(
            args.resume,
            model,
            matrix_optimizer,
            aux_optimizer,
            semantic_config["semantic_config_hash"],
        )
        append_event(events_path, {"event": "resume", "next_update": start_update})

    train_model = torch.compile(model, dynamic=False) if args.compile else model
    eval_every = args.eval_every or max(1, round(total_updates * 0.10))
    save_every = args.save_every or max(1, round(total_updates * 0.25))
    diagnostic_every = args.diagnostic_every or save_every
    validation_steps = set(range(0, total_updates + 1, eval_every)) | {total_updates}
    save_steps = set(range(0, total_updates + 1, save_every)) | {total_updates}
    diagnostic_steps = set(range(0, total_updates, diagnostic_every))
    audit_steps = {int(value) for value in args.audit_steps.split(",") if value.strip()}
    if any(step < 0 or step >= total_updates for step in audit_steps):
        raise ValueError("audit steps must be valid update indices")

    if start_update == 0:
        clean_ce, clean_tokens = evaluate_ce(train_model, val_data, device)
        retention_ce, retention_tokens = evaluate_ce(
            train_model,
            retention_data,
            device,
            max_batches=args.monitor_retention_batches,
        )
        source_ce = {source: evaluate_ce(train_model, ds, device)[0] for source, ds in val_sources.items()}
        append_event(
            events_path,
            {
                "event": "validation",
                "step": 0,
                "mixture_ce": clean_ce,
                "retention_ce": retention_ce,
                "mixture_tokens": clean_tokens,
                "retention_tokens": retention_tokens,
                **{f"{source}_ce": value for source, value in source_ce.items()},
            },
        )
        save_checkpoint(
            checkpoint_dir / "state_step000000.pt",
            model,
            matrix_optimizer,
            aux_optimizer,
            0,
            semantic_config,
        )

    current_segment = None
    started = time.perf_counter()
    cumulative_tokens = 0
    for update in range(start_update, total_updates):
        if update in audit_steps:
            append_event(
                events_path,
                {
                    "event": "audit_state",
                    "step": update,
                    "model_hash": model_hash(model),
                    "matrix_optimizer_hash": optimizer_hash(matrix_optimizer),
                    "aux_optimizer_hash": optimizer_hash(aux_optimizer),
                },
            )
        segment = segment_at(schedule, update)
        if segment != current_segment:
            append_event(
                events_path,
                {
                    "event": "spectral_switch",
                    "step": update,
                    "backend": segment.backend,
                    "promotion_steps": segment.promotion_steps,
                    "suppression_steps": segment.suppression_steps,
                    "alpha": segment.alpha,
                },
            )
            current_segment = segment
        lr_multiplier = schedule_lr(update, total_updates)
        matrix_optimizer.param_groups[0]["lr"] = args.matrix_lr * lr_multiplier
        for group in aux_optimizer.param_groups:
            group["lr"] = args.aux_lr * lr_multiplier
        matrix_optimizer.zero_grad(set_to_none=True)
        aux_optimizer.zero_grad(set_to_none=True)
        row_start = update * rows_per_update(args)
        row_stop = min(len(train_data), row_start + rows_per_update(args))
        indices = list(range(row_start, row_stop))
        valid_total = sum(int((train_data[index][1] != -100).sum()) for index in indices)
        objective = 0.0
        for micro_start in range(0, len(indices), args.device_batch_size):
            micro_indices = indices[micro_start : micro_start + args.device_batch_size]
            x, y = stack_batch(train_data, micro_indices, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = train_model(input_ids=x).logits
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss = loss_sum / valid_total
            loss.backward()
            objective += float(loss_sum)
        raw_diagnostic = None
        if update in diagnostic_steps:
            raw_diagnostic = diagnostic_param.grad.detach().float().cpu().clone()
            matrix_optimizer.request_diagnostic(raw_diagnostic)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        diagnostic_before = diagnostic_param.detach().float().clone() if raw_diagnostic is not None else None
        matrix_optimizer.step()
        aux_optimizer.step()
        model_step = update + 1
        cumulative_tokens += valid_total
        elapsed = time.perf_counter() - started
        append_event(
            events_path,
            {
                "event": "train",
                "update": update,
                "step": model_step,
                "ce": objective / valid_total,
                "valid_tokens": valid_total,
                "matrix_lr": args.matrix_lr * lr_multiplier,
                "aux_lr": args.aux_lr * lr_multiplier,
                "grad_norm": grad_norm,
                "backend": segment.backend,
                "promotion_steps": segment.promotion_steps,
                "suppression_steps": segment.suppression_steps,
                "alpha": segment.alpha,
                "tokens_per_second": cumulative_tokens / max(1e-9, elapsed),
            },
        )
        if raw_diagnostic is not None:
            diagnostic_after = diagnostic_param.detach().float()
            actual_update = (diagnostic_after - diagnostic_before).cpu()
            diagnostic = matrix_optimizer.last_diagnostic
            artifact = {
                **diagnostic,
                "actual_update": actual_update,
                "raw_summary": summarize_spectrum(diagnostic["raw_gradient"]),
                "input_summary": summarize_spectrum(diagnostic["backend_input"]),
                "output_summary": summarize_spectrum(diagnostic["transformed"]),
                "actual_update_norm": float(actual_update.norm()),
            }
            artifact_path = spectra_dir / f"spectrum_update{update:06d}.pt"
            torch.save(artifact, artifact_path)
            append_event(
                events_path,
                {
                    "event": "spectral",
                    "update": update,
                    "model_step": model_step,
                    "backend": segment.backend,
                    "promotion_steps": segment.promotion_steps,
                    "suppression_steps": segment.suppression_steps,
                    "alpha": segment.alpha,
                    "input_rank_gt_0p5": artifact["input_summary"]["rank_gt_0p5"],
                    "output_rank_gt_0p5": artifact["output_summary"]["rank_gt_0p5"],
                    "input_energy_rank_95": artifact["input_summary"]["energy_rank_95"],
                    "output_energy_rank_95": artifact["output_summary"]["energy_rank_95"],
                    "input_fro": artifact["input_summary"]["fro"],
                    "output_fro": artifact["output_summary"]["fro"],
                    "actual_update_norm": artifact["actual_update_norm"],
                    "raw_gradient_hash": tensor_hash(diagnostic["raw_gradient"]),
                    "backend_input_hash": tensor_hash(diagnostic["backend_input"]),
                    "artifact": str(artifact_path),
                },
            )
        if model_step in validation_steps:
            mixture_ce, mixture_tokens = evaluate_ce(train_model, val_data, device)
            retention_ce, retention_tokens = evaluate_ce(
                train_model,
                retention_data,
                device,
                max_batches=args.monitor_retention_batches,
            )
            source_ce = {source: evaluate_ce(train_model, ds, device)[0] for source, ds in val_sources.items()}
            append_event(
                events_path,
                {
                    "event": "validation",
                    "step": model_step,
                    "mixture_ce": mixture_ce,
                    "retention_ce": retention_ce,
                    "mixture_tokens": mixture_tokens,
                    "retention_tokens": retention_tokens,
                    **{f"{source}_ce": value for source, value in source_ce.items()},
                },
            )
        if model_step in save_steps:
            save_checkpoint(
                checkpoint_dir / f"state_step{model_step:06d}.pt",
                model,
                matrix_optimizer,
                aux_optimizer,
                model_step,
                semantic_config,
            )

    final_mixture, final_mixture_tokens = evaluate_ce(train_model, val_data, device)
    final_retention, final_retention_tokens = evaluate_ce(
        train_model,
        retention_data,
        device,
        max_batches=(args.monitor_retention_batches if args.smoke else 0),
    )
    mmlu_records = json.loads((cache / "mmlu_eval.json").read_text())
    gsm_records = json.loads((cache / "gsm8k_eval.json").read_text())
    mmlu_limit = 32 if args.smoke else len(mmlu_records)
    final_mmlu_accuracy = evaluate_mmlu(
        model, tokenizer, mmlu_records[:mmlu_limit], device, manifest["context"]
    )
    gsm_limit = 32 if args.smoke else len(gsm_records)
    final_gsm_accuracy = evaluate_gsm8k(
        model,
        tokenizer,
        gsm_records[:gsm_limit],
        device,
        manifest["context"],
    )
    complete = {
        "event": "complete",
        "step": total_updates,
        "final_mixture_ce": final_mixture,
        "final_mixture_tokens": final_mixture_tokens,
        "full_retention_ce": final_retention,
        "full_retention_tokens": final_retention_tokens,
        "mmlu_accuracy": final_mmlu_accuracy,
        "mmlu_examples": mmlu_limit,
        "gsm8k_accuracy": final_gsm_accuracy,
        "gsm8k_examples": gsm_limit,
        "final_model_hash": model_hash(model),
        "matrix_optimizer_hash": optimizer_hash(matrix_optimizer),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    append_event(events_path, complete)
    json_dump(output / "complete.json", complete)


if __name__ == "__main__":
    main()
