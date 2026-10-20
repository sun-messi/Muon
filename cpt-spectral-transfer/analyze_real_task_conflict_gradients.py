#!/usr/bin/env python3
"""Paired held-out Old/New gradient cosines along one CPT trajectory."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from smoltalk_data import TokenWindowDataset, sha256_file
from train_mid import parameter_partition

STEPS = (0, 1, 5, 10, 20, 50, 100, 200, 400, 610)


def probe_groups(dataset: TokenWindowDataset, count: int, size: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(dataset), size=count * size, replace=False)
    return [selected[i * size:(i + 1) * size].tolist() for i in range(count)]


def fingerprint(dataset: TokenWindowDataset, groups: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for group in groups:
        for index in group:
            x, y, _ = dataset[index]
            digest.update(int(index).to_bytes(8, "little"))
            digest.update(x.numpy().tobytes())
            digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def gradient(model, parameters, dataset, indices, device):
    rows = [dataset[int(index)] for index in indices]
    x = torch.stack([row[0] for row in rows]).to(device)
    y = torch.stack([row[1] for row in rows]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    pieces = torch.autograd.grad(loss, parameters)
    result = [piece.detach() for piece in pieces]
    return result, float(loss.detach())


def cosine_pair(left, right, selected):
    dot = sum(float((a.float() * b.float()).sum()) for a, b, flag in zip(left, right, selected) if flag)
    an = sum(float(a.float().square().sum()) for a, flag in zip(left, selected) if flag)
    bn = sum(float(b.float().square().sum()) for b, flag in zip(right, selected) if flag)
    return dot / math.sqrt(an * bn) if an > 0 and bn > 0 else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=("adam", "muon", "pion"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", default=",".join(map(str, STEPS)))
    parser.add_argument("--fixed-checkpoint", type=Path,
                        help="Use the same checkpoint for all four New tasks.")
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--probe-size", type=int, default=4)
    args = parser.parse_args()
    root = args.run_root.resolve()
    cache = root / "cache" / args.task
    run = root / "training" / f"{args.task}_{args.method}_seed42"
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache / "manifest.json").read_text())
    old = TokenWindowDataset(cache / "retention_tokens.npy", context=1024)
    new = TokenWindowDataset(cache / "validation_tokens.npy", context=1024)
    old_groups = probe_groups(old, args.probe_count, args.probe_size, 42001)
    new_groups = probe_groups(new, args.probe_count, args.probe_size, 42002)
    probes = dict(task=args.task, old_indices=old_groups, new_indices=new_groups,
                  old_sha256=fingerprint(old, old_groups), new_sha256=fingerprint(new, new_groups),
                  old_cache_sha256=sha256_file(cache / "retention_tokens.npy"),
                  new_cache_sha256=sha256_file(cache / "validation_tokens.npy"))
    (output / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_id"], revision=manifest["model_revision"],
        torch_dtype=torch.float32, local_files_only=True).to(device)
    model.eval()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    _, _, matrix_names, _, _ = parameter_partition(model)
    matrix_names = set(matrix_names)
    matrix = [name in matrix_names for name, _ in named]
    all_mask = [True] * len(parameters)
    raw_path = output / "gradient_cosine_raw.csv"
    checkpoints = {}
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "task", "method", "step", "probe", "old_ce", "new_ce",
            "cosine_all", "cosine_matrix"])
        writer.writeheader()
        for step in (int(s) for s in args.steps.split(",") if s.strip()):
            checkpoint = (args.fixed_checkpoint.resolve() if args.fixed_checkpoint else
                          run / "checkpoints" / f"state_step{step:06d}.pt")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if int(payload["next_update"]) != step:
                raise RuntimeError(f"step mismatch: {checkpoint}")
            checkpoints[str(step)] = dict(path=str(checkpoint), sha256=sha256_file(checkpoint))
            model.load_state_dict(payload["model"])
            del payload
            for probe, (old_ids, new_ids) in enumerate(zip(old_groups, new_groups)):
                old_gradient, old_ce = gradient(model, parameters, old, old_ids, device)
                new_gradient, new_ce = gradient(model, parameters, new, new_ids, device)
                row = dict(task=args.task, method=args.method, step=step, probe=probe,
                           old_ce=old_ce, new_ce=new_ce,
                           cosine_all=cosine_pair(old_gradient, new_gradient, all_mask),
                           cosine_matrix=cosine_pair(old_gradient, new_gradient, matrix))
                if not all(math.isfinite(value) for value in
                           (old_ce, new_ce, row["cosine_all"], row["cosine_matrix"])):
                    raise RuntimeError(f"nonfinite gradient probe: {row}")
                writer.writerow(row)
                handle.flush()
                del old_gradient, new_gradient
            print(f"{args.task} {args.method} step{step} complete", flush=True)
    print(output, flush=True)
    (output / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2) + "\n")


if __name__ == "__main__":
    main()
