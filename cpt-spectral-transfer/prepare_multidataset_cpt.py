#!/usr/bin/env python3
"""Prepare fixed 20M-token mathematical CPT caches for Pythia-31M."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

from multidataset_cpt_data import (
    FINE_MATH_ID,
    FINE_MATH_REVISION,
    MATHPILE_WEIGHTS,
    MATH_PILE_ID,
    MATH_PILE_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PROOF_PILE_ID,
    PROOF_PILE_REVISION,
    PROOF_WEIGHTS,
    SEEDS,
    STACK_MATH_ID,
    STACK_MATH_REVISION,
    TRAIN_ROWS,
    VALIDATION_ROWS,
    collect_windows,
    finalize_cache,
    fine_identity,
    is_validation_identity,
    iter_jsonl_files,
    largest_remainder,
    mathpile_paths,
    proof_paths,
    render_stackmath,
    stack_identity,
    text_identity,
)


DATASETS = ("finemath4plus", "proofpile2", "stackmathqa", "mathpile")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/multidataset_cpt_20m_v1"))
    parser.add_argument("--retention-source", type=Path, default=Path("data/cache_v1/retention_tokens.npy"))
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)
    return parser.parse_args()


def prepare_finemath(tokenizer, shuffle_buffer: int):
    def rows(seed: int):
        return load_dataset(
            FINE_MATH_ID,
            "finemath-4plus",
            split="train",
            revision=FINE_MATH_REVISION,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    validation, validation_stats = collect_windows(
        rows(11), tokenizer, count=VALIDATION_ROWS, identity=fine_identity, accept=is_validation_identity
    )
    training, training_stats = collect_windows(
        rows(12), tokenizer, count=TRAIN_ROWS, identity=fine_identity,
        accept=lambda value: not is_validation_identity(value),
    )
    return training, validation, training_stats.as_dict(), validation_stats.as_dict(), {
        "config": "finemath-4plus",
        "split": "train",
        "validation_rule": "sha256(url\\0text) uint32 mod 20 == 0",
        "training_rule": "complement of validation hash bucket",
        "selection_shuffle_seeds": {"validation": 11, "training": 12},
        "shuffle_buffer": shuffle_buffer,
    }


def prepare_stackmath(tokenizer, shuffle_buffer: int):
    def rows(seed: int):
        return load_dataset(
            STACK_MATH_ID,
            "stackmathqa200k",
            split="train",
            revision=STACK_MATH_REVISION,
            streaming=True,
        ).shuffle(seed=seed, buffer_size=shuffle_buffer)

    validation, validation_stats = collect_windows(
        rows(21), tokenizer, count=VALIDATION_ROWS, render=render_stackmath,
        identity=stack_identity, accept=is_validation_identity,
    )
    training, training_stats = collect_windows(
        rows(22), tokenizer, count=TRAIN_ROWS, render=render_stackmath,
        identity=stack_identity, accept=lambda value: not is_validation_identity(value),
    )
    return training, validation, training_stats.as_dict(), validation_stats.as_dict(), {
        "config": "stackmathqa200k",
        "split": "train",
        "render_template": "Question:\\n{Q}\\n\\nAnswer:\\n{A}\\n<EOS>",
        "validation_rule": "stable row identity uint32 mod 20 == 0",
        "training_rule": "complement of validation hash bucket",
        "selection_shuffle_seeds": {"validation": 21, "training": 22},
        "shuffle_buffer": shuffle_buffer,
    }


def prepare_proofpile(tokenizer):
    train_quota = largest_remainder(TRAIN_ROWS, PROOF_WEIGHTS)
    validation_quota = largest_remainder(VALIDATION_ROWS, PROOF_WEIGHTS)
    train_parts, validation_parts = [], []
    train_stats, validation_stats = {}, {}
    for index, component in enumerate(PROOF_WEIGHTS):
        train_rows = iter_jsonl_files(
            PROOF_PILE_ID, PROOF_PILE_REVISION, proof_paths("train", component), seed=100 + index
        )
        part, stats = collect_windows(
            train_rows, tokenizer, count=train_quota[component], identity=text_identity
        )
        train_parts.append(part)
        train_stats[component] = stats.as_dict()

        validation_rows = iter_jsonl_files(
            PROOF_PILE_ID, PROOF_PILE_REVISION, proof_paths("validation", component), seed=200 + index
        )
        part, stats = collect_windows(
            validation_rows, tokenizer, count=validation_quota[component], identity=text_identity
        )
        validation_parts.append(part)
        validation_stats[component] = stats.as_dict()
    training = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    validation = validation[np.random.default_rng(20260821).permutation(len(validation))]
    return training, validation, train_stats, validation_stats, {
        "components": list(PROOF_WEIGHTS),
        "published_token_ratio": PROOF_WEIGHTS,
        "train_sequence_quota": train_quota,
        "validation_sequence_quota": validation_quota,
        "train_split": "train",
        "validation_split": "validation",
    }


def prepare_mathpile(tokenizer):
    train_quota = largest_remainder(TRAIN_ROWS, MATHPILE_WEIGHTS)
    validation_quota = largest_remainder(VALIDATION_ROWS, MATHPILE_WEIGHTS)
    train_parts, validation_parts = [], []
    train_stats, validation_stats = {}, {}
    try:
        for index, component in enumerate(MATHPILE_WEIGHTS):
            train_rows = iter_jsonl_files(
                MATH_PILE_ID, MATH_PILE_REVISION, mathpile_paths("train", component), seed=300 + index
            )
            part, stats = collect_windows(
                train_rows, tokenizer, count=train_quota[component], identity=text_identity
            )
            train_parts.append(part)
            train_stats[component] = stats.as_dict()
            validation_rows = iter_jsonl_files(
                MATH_PILE_ID, MATH_PILE_REVISION, mathpile_paths("validation", component), seed=400 + index
            )
            part, stats = collect_windows(
                validation_rows, tokenizer, count=validation_quota[component], identity=text_identity
            )
            validation_parts.append(part)
            validation_stats[component] = stats.as_dict()
    except GatedRepoError as error:
        raise SystemExit(
            "MathPile is pending access (HTTP 403). Accept the dataset terms at "
            "https://huggingface.co/datasets/GAIR/MathPile and rerun --dataset mathpile."
        ) from error
    training = np.concatenate(train_parts)
    validation = np.concatenate(validation_parts)
    validation = validation[np.random.default_rng(20260821).permutation(len(validation))]
    return training, validation, train_stats, validation_stats, {
        "components": list(MATHPILE_WEIGHTS),
        "published_token_counts": MATHPILE_WEIGHTS,
        "train_sequence_quota": train_quota,
        "validation_sequence_quota": validation_quota,
        "train_split": "train",
        "validation_split": "validation",
    }


def prepare_one(name: str, output_root: Path, retention_source: Path, shuffle_buffer: int) -> dict:
    output = (output_root / name).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing cache: {output}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    builders = {
        "finemath4plus": (prepare_finemath, FINE_MATH_ID, FINE_MATH_REVISION, "finemath4plus"),
        "proofpile2": (prepare_proofpile, PROOF_PILE_ID, PROOF_PILE_REVISION, "proofpile2"),
        "stackmathqa": (prepare_stackmath, STACK_MATH_ID, STACK_MATH_REVISION, "stackmathqa"),
        "mathpile": (prepare_mathpile, MATH_PILE_ID, MATH_PILE_REVISION, "mathpile"),
    }
    builder, dataset_id, revision, primary_name = builders[name]
    payload = builder(tokenizer, shuffle_buffer) if name in {"finemath4plus", "stackmathqa"} else builder(tokenizer)
    training, validation, train_stats, validation_stats, source_config = payload
    if training.shape != (TRAIN_ROWS, 1025) or validation.shape != (VALIDATION_ROWS, 1025):
        raise RuntimeError(f"unexpected cache shapes: {training.shape}, {validation.shape}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.partial.", dir=output.parent))
    manifest = finalize_cache(
        temporary,
        tokenizer,
        dataset_id=dataset_id,
        dataset_revision=revision,
        primary_name=primary_name,
        train_pool=training,
        validation=validation,
        train_stats=train_stats,
        validation_stats=validation_stats,
        retention_source=retention_source,
        source_config=source_config,
        seeds=SEEDS,
    )
    os.rename(temporary, output)
    print(json.dumps({"cache": str(output), "manifest": manifest}, indent=2, sort_keys=True))
    return manifest


def main():
    args = parse_args()
    retention = args.retention_source.resolve()
    if not retention.is_file():
        raise FileNotFoundError(retention)
    names = ("finemath4plus", "proofpile2", "stackmathqa") if args.dataset == "all" else (args.dataset,)
    for name in names:
        prepare_one(name, args.output_root.resolve(), retention, args.shuffle_buffer)


if __name__ == "__main__":
    main()
