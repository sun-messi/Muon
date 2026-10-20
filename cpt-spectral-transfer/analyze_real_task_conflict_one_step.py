#!/usr/bin/env python3
"""Counterfactual New-only/mixed updates from copied training states."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from smoltalk_data import TokenWindowDataset, sha256_file
from spectral import MatchedMuon, parse_schedule
from train_mid import DIAGNOSTIC_PARAM, parameter_partition
from train_smoltalk import schedule_lr

METHODS = {"adam": ("pythia_adam", "none"),
           "muon": ("matched", "0:muon5ns:0:0:0"),
           "pion": ("matched", "0:pion:3:2:.45")}


def ce(model, dataset, indices, device, *, backward=False, weight=1.0):
    batches = [indices[i:i + 8] for i in range(0, len(indices), 8)]
    total = 0.0
    for ids in batches:
        rows = [dataset[int(i)] for i in ids]
        x = torch.stack([row[0] for row in rows]).to(device)
        y = torch.stack([row[1] for row in rows]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        if backward:
            (loss * weight * len(ids) / len(indices)).backward()
        total += float(loss.detach()) * len(ids) / len(indices)
    return total


def build_optimizer(model, method, payload):
    matrix, auxiliary, matrix_names, _, diagnostic = parameter_partition(model)
    family, schedule = METHODS[method]
    if family == "matched":
        main = MatchedMuon(matrix, lr=.003, momentum=.95,
                           schedule=parse_schedule(schedule), diagnostic_param=diagnostic,
                           diagnostic_name=DIAGNOSTIC_PARAM, parameter_names=matrix_names)
        aux = torch.optim.AdamW(auxiliary, lr=.001, weight_decay=.01)
    else:
        main = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=.001, betas=(.9, .95), eps=1e-8, weight_decay=.1)
        aux = None
    main.load_state_dict(payload["matrix_optimizer"])
    if aux is not None:
        aux.load_state_dict(payload["aux_optimizer"])
    return main, aux, matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", default="0,20,100,200,400,610")
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
    # Update rows and evaluation rows do not overlap. 26/32 New approximates
    # the per-update 80/20 mix; exact 80/20 holds over the 610-step training tape.
    update_new = list(range(32))
    update_old = list(range(6))
    eval_old = list(range(64, 96))
    eval_new = list(range(64, 96))
    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_id"], revision=manifest["model_revision"],
        torch_dtype=torch.float32, local_files_only=True).to(device)
    rows = []
    checkpoints = {}
    for step in (int(s) for s in args.steps.split(",") if s.strip()):
        checkpoint = run / "checkpoints" / f"state_step{step:06d}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["next_update"]) != step:
            raise RuntimeError(checkpoint)
        checkpoints[str(step)] = dict(path=str(checkpoint), sha256=sha256_file(checkpoint))
        for mixture in ("new_only", "mixed_26new_6old"):
            model.load_state_dict(payload["model"])
            model.eval()
            with torch.no_grad():
                old_before = ce(model, old, eval_old, device)
                new_before = ce(model, new, eval_new, device)
            model.train()
            main, aux, matrix = build_optimizer(model, args.method, payload)
            multiplier = schedule_lr(min(step, 609), 610)
            main.param_groups[0]["lr"] = (.001 if args.method == "adam" else .003) * multiplier
            if aux is not None:
                for group in aux.param_groups:
                    group["lr"] = .001 * multiplier
            main.zero_grad(set_to_none=True)
            if aux is not None:
                aux.zero_grad(set_to_none=True)
            if mixture == "new_only":
                update_ce = ce(model, new, update_new, device, backward=True)
            else:
                update_ce = (ce(model, new, update_new[:26], device,
                                backward=True, weight=26 / 32) * 26 / 32 +
                             ce(model, old, update_old, device,
                                backward=True, weight=6 / 32) * 6 / 32)
            trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            previous = [p.detach().clone() for p in trainable]
            main.step()
            if aux is not None:
                aux.step()
            delta = [p.detach() - before for p, before in zip(trainable, previous)]
            update_norm = math.sqrt(sum(float(part.float().square().sum()) for part in delta))
            matrix_set = {id(p) for p in matrix}
            matrix_norm = math.sqrt(sum(float(part.float().square().sum())
                                        for p, part in zip(trainable, delta) if id(p) in matrix_set))
            model.eval()
            with torch.no_grad():
                old_after = ce(model, old, eval_old, device)
                new_after = ce(model, new, eval_new, device)
            normmatched_old = None
            normmatched_new = None
            full_matrix_norm = None
            normmatch_ratio = None
            same_state_full_old = None
            same_state_full_new = None
            if args.method == "pion":
                # The reference uses the same checkpoint, momentum, minibatch,
                # clipping, and LR; only the matrix spectral transform is Muon.
                model.load_state_dict(payload["model"])
                model.train()
                full_main, full_aux, full_matrix = build_optimizer(model, "muon", payload)
                full_main.param_groups[0]["lr"] = .003 * multiplier
                if full_aux is not None:
                    for group in full_aux.param_groups:
                        group["lr"] = .001 * multiplier
                full_main.zero_grad(set_to_none=True)
                if full_aux is not None:
                    full_aux.zero_grad(set_to_none=True)
                if mixture == "new_only":
                    ce(model, new, update_new, device, backward=True)
                else:
                    ce(model, new, update_new[:26], device, backward=True, weight=26/32)
                    ce(model, old, update_old, device, backward=True, weight=6/32)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                full_before = [p.detach().clone() for p in full_matrix]
                full_main.step()
                if full_aux is not None:
                    full_aux.step()
                full_matrix_norm = math.sqrt(sum(
                    float((p.detach() - before).float().square().sum())
                    for p, before in zip(full_matrix, full_before)))
                normmatch_ratio = full_matrix_norm / matrix_norm if matrix_norm > 0 else float("nan")
                model.eval()
                with torch.no_grad():
                    same_state_full_old = ce(model, old, eval_old, device)
                    same_state_full_new = ce(model, new, eval_new, device)
                with torch.no_grad():
                    for parameter, before, piece in zip(trainable, previous, delta):
                        factor = normmatch_ratio if id(parameter) in matrix_set else 1.0
                        parameter.copy_(before + factor * piece)
                model.eval()
                with torch.no_grad():
                    normmatched_old = ce(model, old, eval_old, device)
                    normmatched_new = ce(model, new, eval_new, device)
                del full_main, full_aux, full_matrix, full_before
            row = dict(task=args.task, method=args.method, step=step, mixture=mixture,
                       update_batch_new=32 if mixture == "new_only" else 26,
                       update_batch_old=0 if mixture == "new_only" else 6,
                       update_ce=update_ce, raw_grad_norm=grad_norm,
                       update_norm=update_norm, matrix_update_norm=matrix_norm,
                       old_ce_before=old_before, old_ce_after=old_after,
                       delta_old_ce=old_after-old_before,
                       new_ce_before=new_before, new_ce_after=new_after,
                       delta_new_ce=new_after-new_before,
                       full_muon_same_state_matrix_norm=full_matrix_norm,
                       same_state_full_old_ce_after=same_state_full_old,
                       same_state_full_new_ce_after=same_state_full_new,
                       pion_normmatch_ratio=normmatch_ratio,
                       normmatched_old_ce_after=normmatched_old,
                       normmatched_new_ce_after=normmatched_new,
                       terminal_extra_step=(step == 610))
            if not all(math.isfinite(v) for v in row.values() if isinstance(v, float)):
                raise RuntimeError(f"nonfinite update result {row}")
            rows.append(row)
            print(f"{args.task} {args.method} step{step} {mixture}: "
                  f"delta Old={row['delta_old_ce']:+.5g}", flush=True)
            del main, aux, delta, previous
        del payload
    with (output / "one_step.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metadata.json").write_text(json.dumps(dict(
        checkpoints=checkpoints, update_new_indices=update_new,
        update_old_indices=update_old, eval_old_indices=eval_old,
        eval_new_indices=eval_new, mixed_fraction=26/32,
        terminal_step_note="step610 is a hypothetical post-training step at the final LR",
    ), indent=2) + "\n")


if __name__ == "__main__":
    main()
