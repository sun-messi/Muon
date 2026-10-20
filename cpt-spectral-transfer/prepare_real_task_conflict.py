#!/usr/bin/env python3
"""Build paired 80/20 Pile-replay caches for natural New domains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from smoltalk_data import sha256_file

HERE = Path(__file__).resolve().parent
BASE = HERE / "data/openwebmath_pile_replay_rho_sweep_20m_v1"
MULTI = HERE / "data/multidataset_cpt_20m_v1"
TASKS = ("openwebmath", "finemath4plus", "proofpile2", "stackmathqa")
TRAIN_NAME = "train_seed42_rho20_tokens.npy"
SOURCE_NAME = "train_seed42_rho20_source_ids.npy"
COMPONENT_NAME = "train_seed42_rho20_pile_component_ids.npy"


def linked(source: Path, dest: Path) -> None:
    if dest.exists():
        raise FileExistsError(dest)
    os.link(source, dest)


def build(task: str, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    base = json.loads((BASE / "manifest.json").read_text())
    new_source = BASE if task == "openwebmath" else MULTI / task
    if task != "openwebmath":
        new_manifest = json.loads((new_source / "manifest.json").read_text())
        assert new_manifest["model_revision"] == base["model_revision"]
        assert new_manifest["context"] == base["context"] == 1024
        assert new_manifest["loss_mask"] == base["loss_mask"] == "none_all_tokens"
    source_ids = np.load(BASE / SOURCE_NAME, mmap_mode="r")
    base_train = np.load(BASE / TRAIN_NAME, mmap_mode="r")
    assert source_ids.shape == (19520,) and base_train.shape == (19520, 1025)
    assert (int((source_ids == 0).sum()), int((source_ids == 1).sum())) == (15616, 3904)
    linked(BASE / SOURCE_NAME, destination / SOURCE_NAME)
    linked(BASE / COMPONENT_NAME, destination / COMPONENT_NAME)
    linked(new_source / "validation_tokens.npy", destination / "validation_tokens.npy")
    linked(BASE / "retention_tokens.npy", destination / "retention_tokens.npy")
    linked(BASE / "wikitext_tokens.npy", destination / "wikitext_tokens.npy")
    if task == "openwebmath":
        linked(BASE / TRAIN_NAME, destination / TRAIN_NAME)
    else:
        new_train = np.load(new_source / "train_seed42_tokens.npy", mmap_mode="r")
        assert new_train.shape == (19520, 1025) and new_train.dtype == base_train.dtype
        combined = np.array(base_train)
        combined[source_ids == 0] = new_train[:15616]
        assert np.array_equal(combined[source_ids == 1], base_train[source_ids == 1])
        np.save(destination / TRAIN_NAME, combined)
    fields = (TRAIN_NAME, SOURCE_NAME, COMPONENT_NAME, "validation_tokens.npy",
              "retention_tokens.npy", "wikitext_tokens.npy")
    manifest = {key: base[key] for key in (
        "format", "context", "eos_token_id", "loss_mask", "model_id", "model_revision",
        "tokenizer_id", "tokenizer_revision", "vocab_size")}
    manifest.update(dataset_id=(base["dataset_id"] if task == "openwebmath" else
                               json.loads((new_source / "manifest.json").read_text())["dataset_id"] + "+EleutherAI/pile_val_test"),
                    dataset_revision=(base["dataset_revision"] if task == "openwebmath" else
                                      json.loads((new_source / "manifest.json").read_text())["dataset_revision"] + "+" +
                                      base["dataset_revision"].split("+")[-1]),
                    primary_validation_name=task, retention_validation_name="pile",
                    secondary_validations={"wikitext": "wikitext_tokens.npy"},
                    train_file_template="train_seed{seed}_{rho_label}_tokens.npy",
                    train_source_ids_template="train_seed{seed}_{rho_label}_source_ids.npy",
                    train_component_ids_template="train_seed{seed}_{rho_label}_pile_component_ids.npy",
                    training_updates=610, seeds=[42],
                    file_sha256={name: sha256_file(destination / name) for name in fields},
                    pairing=dict(base_manifest_sha256=sha256_file(BASE / "manifest.json"),
                                 new_manifest_sha256=(sha256_file(new_source / "manifest.json")
                                                      if task != "openwebmath" else sha256_file(BASE / "manifest.json")),
                                 old_rows=3904, new_rows=15616, source_ids_sha256=sha256_file(BASE / SOURCE_NAME),
                                 pile_row_sha256=sha256_file(BASE / TRAIN_NAME)))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(task=task, path=str(destination), manifest_sha256=sha256_file(destination / "manifest.json"),
                training_sha256=manifest["file_sha256"][TRAIN_NAME], old_exact=True,
                source_ids_sha256=manifest["file_sha256"][SOURCE_NAME])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    rows = [build(task, root / task) for task in TASKS]
    (root / "pairing.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
