#!/usr/bin/env python3
"""Check paired sources, initial gradients, endpoints, and checkpoints."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

TASKS = ("openwebmath", "finemath4plus", "proofpile2", "stackmathqa")
METHODS = ("adam", "muon", "pion")
STEPS = (0, 1, 5, 10, 20, 50, 100, 200, 400, 610)


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="Comma-separated task names; default preserves original study")
    parser.add_argument("--reference-run", type=Path,
                        help="Additionally verify Old rows, positions, and model against the prior OpenWebMath route.")
    args = parser.parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must specify unique task names")
    root = args.run_root.resolve()
    output = root / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    source_reference = None
    old_reference = None
    first_old_indices = None
    report = {"tasks": {}, "all_pass": True}
    for task in tasks:
        cache = root / "cache" / task
        manifest = json.loads((cache / "manifest.json").read_text())
        sources = np.load(cache / "train_seed42_rho20_source_ids.npy", mmap_mode="r")
        tape = np.load(cache / "train_seed42_rho20_tokens.npy", mmap_mode="r")
        source_hash = hashlib.sha256(sources.tobytes()).hexdigest()
        old_hash = hashlib.sha256(tape[sources == 1].tobytes()).hexdigest()
        old_indices = np.flatnonzero(sources == 1)
        if source_reference is None:
            source_reference, old_reference, first_old_indices = source_hash, old_hash, old_indices
        assert source_hash == source_reference and old_hash == old_reference
        assert np.array_equal(old_indices, first_old_indices)
        assert (int((sources == 0).sum()), int((sources == 1).sum())) == (15616, 3904)
        first_models = []
        first_gradients = []
        paired_fingerprints = []
        final = {}
        for method in METHODS:
            run = root / "training" / f"{task}_{method}_seed42"
            stream = events(run / "events.jsonl")
            assert stream[-1]["event"] == "complete", (task, method, stream[-1])
            config = stream[0]
            paired_fingerprints.append(tuple(config.get(key) for key in (
                "train_fingerprint", "monitor_fingerprint", "retention_fingerprint", "full_validation_fingerprint")))
            first_models.append(config["base_model_hash"])
            first_gradients.append(next(x["raw_gradient_hash"] for x in stream
                                        if x.get("event") == "spectral" and x.get("update") == 0))
            train = [x for x in stream if x.get("event") == "train"]
            assert len(train) == 610 and [x["update"] for x in train] == list(range(610))
            validations = [x for x in stream if x.get("event") == "validation"]
            assert validations[-1]["step"] == 610
            for row in train + validations + [stream[-1]]:
                for key, value in row.items():
                    if isinstance(value, float):
                        assert math.isfinite(value), (task, method, key, value)
            for step in STEPS:
                checkpoint = run / "checkpoints" / f"state_step{step:06d}.pt"
                assert checkpoint.is_file(), checkpoint
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                assert payload["next_update"] == step
                assert payload["model"] and payload["matrix_optimizer"]
                del payload
            diagnostic_name = f"{task}_{method}_seed42"
            gradients = csv_rows(root / "diagnostics/gradients" / diagnostic_name /
                                 "gradient_cosine_raw.csv")
            assert len(gradients) == len(STEPS) * 8
            assert {int(row["step"]) for row in gradients} == set(STEPS)
            assert all(math.isfinite(float(row["cosine_all"])) and
                       math.isfinite(float(row["cosine_matrix"])) and
                       -1.0001 <= float(row["cosine_all"]) <= 1.0001
                       for row in gradients)
            updates = csv_rows(root / "diagnostics/one_step" / diagnostic_name / "one_step.csv")
            assert len(updates) == 12
            assert all(math.isfinite(float(row[key])) for row in updates
                       for key in ("old_ce_before", "old_ce_after", "new_ce_before",
                                   "new_ce_after", "delta_old_ce", "delta_new_ce"))
            if method == "pion":
                assert all(math.isfinite(float(row[key])) for row in updates
                           for key in ("full_muon_same_state_matrix_norm",
                                       "pion_normmatch_ratio", "normmatched_old_ce_after",
                                       "normmatched_new_ce_after"))
            final[method] = dict(old_ce=validations[-1]["full_retention_ce"],
                                 new_ce=validations[-1]["full_primary_ce"],
                                 wikitext_ce=validations[-1]["full_wikitext_ce"],
                                 first_gradient_hash=first_gradients[-1],
                                 initial_model_hash=first_models[-1])
        assert len(set(first_models)) == 1, (task, first_models)
        assert len(set(first_gradients)) == 1, (task, first_gradients)
        assert len(set(paired_fingerprints)) == 1, (task, paired_fingerprints)
        report["tasks"][task] = dict(source_hash=source_hash, old_tape_sha256=old_hash,
                                     new_rows=15616, old_rows=3904, first_old_indices_sha256=
                                     hashlib.sha256(first_old_indices.tobytes()).hexdigest(),
                                     initial_hash=first_models[0],
                                     first_gradient_hash=first_gradients[0], methods=final,
                                     manifest=str(cache / "manifest.json"),
                                     pairing_manifest_note=(
                                         "pairing.pile_row_sha256 in the generated manifest "
                                         "hashes the base whole-train file; old_tape_sha256 here "
                                         "is the actual concatenated Old-row hash"))
    report["cross_task_old_rows_exact"] = True
    report["diagnostic_probe_rows_per_route"] = 80
    report["one_step_rows_per_route"] = 12
    report["same_initial_model_all_tasks"] = len(set(
        item["initial_hash"] for item in report["tasks"].values())) == 1
    assert report["same_initial_model_all_tasks"]
    if args.reference_run:
        reference = args.reference_run.resolve()
        cache = reference / "cache/openwebmath"
        reference_sources = np.load(cache / "train_seed42_rho20_source_ids.npy", mmap_mode="r")
        reference_tape = np.load(cache / "train_seed42_rho20_tokens.npy", mmap_mode="r")
        assert hashlib.sha256(reference_sources.tobytes()).hexdigest() == source_reference
        assert hashlib.sha256(reference_tape[reference_sources == 1].tobytes()).hexdigest() == old_reference
        config = json.loads((reference / "training/openwebmath_muon_seed42/config.json").read_text())
        assert all(item["initial_hash"] == config["base_model_hash"] for item in report["tasks"].values())
        report["prior_run_pairing"] = dict(run=str(reference), old_rows_exact=True,
                                           source_positions_exact=True, initial_model_exact=True)
    for step in (0, 200):
        reference_sha = None
        for task in tasks:
            fixed = root / "diagnostics/fixed" / f"fixed_step{step}_{task}"
            values = csv_rows(fixed / "gradient_cosine_raw.csv")
            assert len(values) == 8 and {int(row["step"]) for row in values} == {step}
            assert all(math.isfinite(float(row["cosine_all"])) for row in values)
            metadata = json.loads((fixed / "checkpoints.json").read_text())
            sha = metadata[str(step)]["sha256"]
            if reference_sha is None:
                reference_sha = sha
            assert sha == reference_sha
        report[f"fixed_step{step}_reference_checkpoint_sha256"] = reference_sha
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
