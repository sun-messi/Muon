#!/usr/bin/env python3
"""Prepare nested OpenWebMath + The Pile replay caches for six rho values."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from openwebmath_data import document_identity, take_stream
from openwebmath_pile_replay_data import (
    CONTEXT,
    PILE_BALANCED_ROWS_PER_SOURCE,
    PILE_WEIGHTED_ROWS,
    PILE_WEIGHTS,
    RHO_VALUES,
    SEQUENCES_PER_UPDATE,
    UPDATES,
    collect_component_windows,
    document_set_digest,
    interleave_windows,
    largest_remainder_counts,
    mixture_source_ids,
    pile_document_identity,
    rho_spec,
    select_component_prefixes,
    stream_to_windows,
    validate_source_schedule,
)
from smoltalk_data import sha256_file, write_json


MODEL_ID = "EleutherAI/pythia-31m"
MODEL_REVISION = "e556ace21b489575e94e9d50b6dad2fcc7419679"
OPENWEBMATH_ID = "open-web-math/open-web-math"
OPENWEBMATH_REVISION = "fde8ef8de2300f5e778f56261843dab89f230815"
PILE_ID = "EleutherAI/pile_val_test"
PILE_REVISION = "05b327037e6301f256d8df32193756edc4c8e3bd"
SEEDS = (42, 43, 44, 45)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/openwebmath_pile_replay_rho_sweep_20m_v1"),
    )
    parser.add_argument(
        "--openwebmath-cache",
        type=Path,
        default=Path("data/openwebmath_cpt_20m_v1"),
    )
    parser.add_argument(
        "--wikitext-retention",
        type=Path,
        default=Path("data/cache_v1/retention_tokens.npy"),
    )
    parser.add_argument("--shuffle-buffer", type=int, default=50_000)
    parser.add_argument(
        "--updates",
        type=int,
        default=UPDATES,
        help="Training horizon used to size every rho cache (default preserves Run4).",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SEEDS),
        help="Comma-separated cached data-order seeds.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain at least one unique integer")
    return seeds


def identities_digest(values) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def openwebmath_rows(seed: int, buffer_size: int):
    return load_dataset(
        OPENWEBMATH_ID,
        split="train",
        revision=OPENWEBMATH_REVISION,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=buffer_size)


def pile_rows(split: str, seed: int, buffer_size: int):
    return load_dataset(
        PILE_ID,
        split=split,
        revision=PILE_REVISION,
        streaming=True,
    ).shuffle(seed=seed, buffer_size=buffer_size)


def flatten_identities(values: dict[str, list[str]]) -> set[str]:
    return {identity for identities in values.values() for identity in identities}


def main():
    args = parse_args()
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    seeds = parse_seeds(args.seeds)
    total_rows = args.updates * SEQUENCES_PER_UPDATE
    total_targets = total_rows * CONTEXT
    output = args.output_dir.resolve()
    old_math_cache = args.openwebmath_cache.resolve()
    wiki_source = args.wikitext_retention.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"refusing to overwrite non-empty cache: {output}")
    if not (old_math_cache / "validation_tokens.npy").is_file():
        raise FileNotFoundError("historical OpenWebMath validation cache is missing")
    if not wiki_source.is_file():
        raise FileNotFoundError("historical WikiText retention cache is missing")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    generated: list[Path] = []

    # Reconstruct the historical OpenWebMath validation documents and insist on
    # byte-identical validation tokens before using them as exclusions.
    math_validation_stream, math_validation_stats, math_validation_documents = take_stream(
        iter(openwebmath_rows(42, args.shuffle_buffer)),
        tokenizer,
        stream_tokens=PILE_WEIGHTED_ROWS * CONTEXT + 1,
    )
    historical_math_validation = np.load(
        old_math_cache / "validation_tokens.npy", mmap_mode="r"
    )
    if not np.array_equal(math_validation_stream, historical_math_validation):
        raise RuntimeError("OpenWebMath validation reconstruction differs from historical cache")
    math_validation_path = output / "validation_tokens.npy"
    shutil.copyfile(old_math_cache / "validation_tokens.npy", math_validation_path)
    generated.append(math_validation_path)

    # Pile test is evaluation-only. Build one component pool large enough for
    # both weighted and 32-per-component balanced evaluation.
    weighted_counts = largest_remainder_counts(PILE_WEIGHTED_ROWS, PILE_WEIGHTS)
    test_requirements = {
        name: max(weighted_counts[name], PILE_BALANCED_ROWS_PER_SOURCE)
        for name in PILE_WEIGHTS
    }
    test_pools, test_identities = collect_component_windows(
        pile_rows("test", 6_100_042, args.shuffle_buffer),
        tokenizer,
        test_requirements,
    )
    test_identity_set = flatten_identities(test_identities)
    weighted_test, weighted_test_sources = select_component_prefixes(
        test_pools, weighted_counts, seed=71_000_042
    )
    retention_path = output / "retention_tokens.npy"
    retention_sources_path = output / "retention_component_ids.npy"
    np.save(retention_path, weighted_test)
    np.save(retention_sources_path, weighted_test_sources)
    generated.extend((retention_path, retention_sources_path))

    balanced_counts = {name: PILE_BALANCED_ROWS_PER_SOURCE for name in PILE_WEIGHTS}
    balanced_test, balanced_sources = select_component_prefixes(
        test_pools, balanced_counts, seed=72_000_042
    )
    balanced_path = output / "pile_component_balanced_tokens.npy"
    balanced_sources_path = output / "pile_component_balanced_source_ids.npy"
    np.save(balanced_path, balanced_test)
    np.save(balanced_sources_path, balanced_sources)
    generated.extend((balanced_path, balanced_sources_path))

    # A fixed, weighted held-out batch from the replay split is reserved for
    # mechanism diagnostics and excluded from every seed's replay training pool.
    diagnostic_counts = largest_remainder_counts(32, PILE_WEIGHTS)
    diagnostic_pools, diagnostic_identities = collect_component_windows(
        pile_rows("validation", 6_200_042, args.shuffle_buffer),
        tokenizer,
        diagnostic_counts,
        forbidden_identities=test_identity_set,
    )
    diagnostic_identity_set = flatten_identities(diagnostic_identities)
    diagnostic_windows, diagnostic_sources = select_component_prefixes(
        diagnostic_pools, diagnostic_counts, seed=73_000_042
    )
    diagnostic_path = output / "pile_replay_diagnostic_tokens.npy"
    diagnostic_sources_path = output / "pile_replay_diagnostic_source_ids.npy"
    np.save(diagnostic_path, diagnostic_windows)
    np.save(diagnostic_sources_path, diagnostic_sources)
    generated.extend((diagnostic_path, diagnostic_sources_path))

    wiki_path = output / "wikitext_tokens.npy"
    shutil.copyfile(wiki_source, wiki_path)
    generated.append(wiki_path)

    max_pile_counts = largest_remainder_counts(
        rho_spec(0.40, updates=args.updates).pile_rows, PILE_WEIGHTS
    )
    seed_manifest: dict[str, dict] = {}
    selected_validation_identities: set[str] = set()
    for seed in seeds:
        # One maximal Math stream is sliced into nested prefixes for all rho.
        max_math_rows = rho_spec(0.0, updates=args.updates).math_rows
        math_iterator = (
            row
            for row in openwebmath_rows(seed, args.shuffle_buffer)
            if document_identity(row) not in math_validation_documents
        )
        math_stream, math_stats, math_documents = take_stream(
            math_iterator,
            tokenizer,
            stream_tokens=max_math_rows * CONTEXT + 1,
        )
        if math_documents & math_validation_documents:
            raise RuntimeError(f"seed{seed}: OpenWebMath train/validation overlap")
        math_pool = stream_to_windows(math_stream, max_math_rows)

        # The validation split is replay-only. Exclude all exact normalized
        # documents observed while constructing the held-out test pools.
        pile_pools, pile_identities = collect_component_windows(
            pile_rows("validation", seed, args.shuffle_buffer),
            tokenizer,
            max_pile_counts,
            forbidden_identities=test_identity_set | diagnostic_identity_set,
        )
        validation_identities = flatten_identities(pile_identities)
        if validation_identities & test_identity_set:
            raise RuntimeError(f"seed{seed}: Pile validation/test leakage")
        selected_validation_identities.update(validation_identities)

        rho_manifest: dict[str, dict] = {}
        for rho in RHO_VALUES:
            spec = rho_spec(rho, updates=args.updates)
            pile_counts = largest_remainder_counts(spec.pile_rows, PILE_WEIGHTS)
            pile_windows, pile_component_ids = select_component_prefixes(
                pile_pools,
                pile_counts,
                seed=seed + 81_000_000 + int(round(rho * 10_000)),
            )
            source_ids = mixture_source_ids(seed, rho, updates=args.updates)
            source_counts = validate_source_schedule(
                source_ids, rho, updates=args.updates
            )
            mixed = interleave_windows(
                math_pool[: spec.math_rows],
                pile_windows,
                source_ids,
                updates=args.updates,
            )
            full_component_ids = np.full(total_rows, 255, dtype=np.uint8)
            full_component_ids[source_ids == 1] = pile_component_ids

            train_path = output / f"train_seed{seed}_{spec.label}_tokens.npy"
            source_path = output / f"train_seed{seed}_{spec.label}_source_ids.npy"
            component_path = output / f"train_seed{seed}_{spec.label}_pile_component_ids.npy"
            np.save(train_path, mixed)
            np.save(source_path, source_ids)
            np.save(component_path, full_component_ids)
            generated.extend((train_path, source_path, component_path))
            rho_manifest[spec.label] = {
                **spec.as_dict(),
                "target_tokens": total_targets,
                "source_counts": source_counts,
                "pile_component_counts": pile_counts,
                "train_sha256": sha256_file(train_path),
                "source_ids_sha256": sha256_file(source_path),
                "pile_component_ids_sha256": sha256_file(component_path),
            }
        seed_manifest[str(seed)] = {
            "openwebmath": math_stats.as_dict(),
            "openwebmath_document_set_sha256": identities_digest(math_documents),
            "pile_validation_document_set_sha256": document_set_digest(pile_identities),
            "rho": rho_manifest,
        }

    manifest = {
        "format": "pythia31m_openwebmath_pile_replay_rho_cache_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_id": f"{OPENWEBMATH_ID}+{PILE_ID}",
        "dataset_revision": f"{OPENWEBMATH_REVISION}+{PILE_REVISION}",
        "source_datasets": {
            "openwebmath": {"id": OPENWEBMATH_ID, "revision": OPENWEBMATH_REVISION},
            "pile_replay": {
                "id": PILE_ID,
                "revision": PILE_REVISION,
                "split": "validation",
            },
            "pile_retention": {
                "id": PILE_ID,
                "revision": PILE_REVISION,
                "split": "test",
            },
        },
        "context": CONTEXT,
        "seeds": list(seeds),
        "rho_values": list(RHO_VALUES),
        "rho_labels": {
            str(rho): rho_spec(rho, updates=args.updates).label
            for rho in RHO_VALUES
        },
        "training_updates": args.updates,
        "train_file_template": "train_seed{seed}_{rho_label}_tokens.npy",
        "train_source_ids_template": "train_seed{seed}_{rho_label}_source_ids.npy",
        "train_component_ids_template": "train_seed{seed}_{rho_label}_pile_component_ids.npy",
        "training": seed_manifest,
        "validation": {
            **math_validation_stats.as_dict(),
            "document_set_sha256": identities_digest(math_validation_documents),
        },
        "retention": {
            "name": "pile_weighted_test",
            "rows": PILE_WEIGHTED_ROWS,
            "target_tokens": PILE_WEIGHTED_ROWS * CONTEXT,
            "component_counts": weighted_counts,
            "document_set_sha256": document_set_digest(test_identities),
        },
        "component_balanced_retention": {
            "rows_per_component": PILE_BALANCED_ROWS_PER_SOURCE,
            "rows": len(PILE_WEIGHTS) * PILE_BALANCED_ROWS_PER_SOURCE,
            "target_tokens": len(PILE_WEIGHTS) * PILE_BALANCED_ROWS_PER_SOURCE * CONTEXT,
            "component_names": list(PILE_WEIGHTS),
        },
        "replay_diagnostic": {
            "rows": 32,
            "target_tokens": 32 * CONTEXT,
            "component_counts": diagnostic_counts,
            "document_set_sha256": document_set_digest(diagnostic_identities),
        },
        "pile_weights": PILE_WEIGHTS,
        "monitor_validation_rows": PILE_WEIGHTED_ROWS,
        "primary_validation_name": "openwebmath",
        "retention_validation_name": "pile",
        "secondary_validations": {"wikitext": "wikitext_tokens.npy"},
        "loss_mask": "none_all_tokens",
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pile_validation_test_exact_overlap_count": len(
            selected_validation_identities & test_identity_set
        ),
        "file_sha256": {path.name: sha256_file(path) for path in generated},
    }
    # Keep the default 610-update/four-seed manifest byte-schema compatible
    # with Run4.  Non-default caches expose their geometry explicitly so a
    # long-horizon launcher can reject an accidentally reused Run4 cache.
    if args.updates != UPDATES or seeds != SEEDS:
        manifest.update(
            sequences_per_update=SEQUENCES_PER_UPDATE,
            total_training_rows=total_rows,
            total_training_targets=total_targets,
        )
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
