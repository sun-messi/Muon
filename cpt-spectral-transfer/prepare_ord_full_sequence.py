#!/usr/bin/env python3
"""Small ORD subset and paired 1024-token full-loss CPT cache."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from transformers import AutoTokenizer

from smoltalk_data import sha256_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data/ord_small/deps"))
from ord_schema.proto import reaction_pb2  # noqa: E402


def identity(component):
    choices = [x for x in component.identifiers if x.value and x.type in (2, 6)]
    return choices[0].value if choices else next((x.value for x in component.identifiers if x.value), "")


def render(record):
    parts = {"reactants": [], "products": [], "catalysts": [], "solvents": []}
    for group in record.inputs.values():
        for comp in group.components:
            value = identity(comp)
            if not value:
                continue
            role = comp.DESCRIPTOR.fields_by_name["reaction_role"].enum_type.values_by_number[comp.reaction_role].name
            key = "reactants" if role in ("REACTANT", "REAGENT") else "catalysts" if role == "CATALYST" else "solvents" if role == "SOLVENT" else "reactants"
            parts[key].append(value)
    for outcome in record.outcomes:
        for product in outcome.products:
            value = identity(product)
            if value:
                parts["products"].append(value)
    condition = str(record.conditions).replace("\n", " ").strip()[:240]
    yields = [str(m.percentage.value) for o in record.outcomes for p in o.products for m in p.measurements if m.type == m.YIELD]
    return "\n".join(["[REACTION]", *(f"{key}: {', '.join(dict.fromkeys(values)) or 'none'}" for key, values in parts.items()), f"conditions: {condition or 'none'}", f"yield: {', '.join(yields) or 'unknown'}", "[/REACTION]"])


def main():
    out = ROOT / "data/ord_full_sequence_v3"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    source = ROOT / "data/openwebmath_pile_replay_rho_sweep_65m_seed42_v1"
    source_manifest = json.loads((source / "manifest.json").read_text())
    raw = np.load(source / "train_seed42_rho20_tokens.npy", mmap_mode="r")
    source_ids = np.load(source / "train_seed42_rho20_source_ids.npy", mmap_mode="r")
    rng = np.random.default_rng(42 + 700001)
    new = rng.choice(np.flatnonzero(source_ids == 0), 31232, replace=False)
    old = rng.choice(np.flatnonzero(source_ids == 1), 7808, replace=False)
    indices = np.concatenate((new, old))
    rng.shuffle(indices)
    np.save(out / "train_seed42_tokens.npy", np.asarray(raw[indices, :1024], dtype=np.uint16))
    np.save(out / "train_seed42_source_ids.npy", np.asarray(source_ids[indices], dtype=np.uint8))
    tok = AutoTokenizer.from_pretrained(source_manifest["tokenizer_id"], revision=source_manifest["tokenizer_revision"])
    parquet = ROOT / "data/ord_small/ord_dataset-0078ecb24abd66512594995d2e098f5e.parquet"
    records = pd.read_parquet(parquet)
    seen_ids, seen_texts, texts, ids = set(), set(), [], []
    for _, row in records.iterrows():
        rid = str(row.reaction_id)
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        reaction = reaction_pb2.Reaction(); reaction.ParseFromString(row.reaction)
        text = render(reaction)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen_texts:
            continue
        seen_texts.add(digest)
        texts.append(text); ids.append(rid)
    tokens = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    keep = [(rid, t) for rid, t in zip(ids, tokens) if len(t) >= 16]
    # Reaction IDs and full rendered texts are disjoint between train and validation.
    rng = np.random.default_rng(424242)
    order = rng.permutation(len(keep))
    split = max(1, len(keep) // 10)
    candidates_val = [keep[int(i)] for i in order[:split]]
    candidates_train = [keep[int(i)] for i in order[split:]]
    powers = np.uint64(257) ** np.arange(64, dtype=np.uint64)
    def grams(tokens):
        if len(tokens) < 64:
            return set()
        return set(map(int, sliding_window_view(np.asarray(tokens, dtype=np.uint64), 64) @ powers))
    seen_grams = set()
    def unique_rows(candidates):
        result = []
        for rid, values in candidates:
            shingles = grams(values)
            if shingles & seen_grams:
                continue
            result.append((rid, values))
            seen_grams.update(shingles)
        return result
    train = unique_rows(candidates_train)
    val = unique_rows(candidates_val)
    for prefix, rows in (("ord_train", train), ("ord_validation", val)):
        width = max(len(x[1]) for x in rows)
        values = np.zeros((len(rows), width), dtype=np.uint16)
        lengths = np.zeros(len(rows), dtype=np.int32)
        for i, (_, t) in enumerate(rows):
            values[i, :len(t)] = t; lengths[i] = len(t)
        np.save(out / f"{prefix}_tokens.npy", values)
        np.save(out / f"{prefix}_lengths.npy", lengths)
        (out / f"{prefix}_ids.json").write_text(json.dumps([x[0] for x in rows]))
    # A separate held-out OpenWebMath validation stream; same Old and Wiki validation as Core B.
    coreb = ROOT / "data/goal_coreb_interference_v1"
    for name in ("validation_tokens.npy", "retention_tokens.npy", "wikitext_tokens.npy"):
        import shutil
        shutil.copy2(coreb / name, out / name)
    # Core B held-out New bodies have 768 tokens; use independent 1024-token
    # windows from the Pile-replay reference cache for full-sequence evaluation.
    reference_new = np.load(source / "validation_tokens.npy", mmap_mode="r")
    clean = np.asarray(reference_new[:977 * 1024].reshape(977, 1024), dtype=np.uint16)
    np.save(out / "clean_new_tokens.npy", clean)
    audit = {
        "source_rho20_manifest_sha256": sha256_file(source / "manifest.json"),
        "source_ord_parquet_sha256": sha256_file(parquet),
        "train_new_rows": 31232, "train_old_rows": 7808,
        "ord_unique_reaction_ids": len(seen_ids), "ord_unique_rendered_records": len(keep),
        "ord_train_records": len(train), "ord_validation_records": len(val),
        "ord_long_ngram_length": 64,
        "ord_long_ngram_dropped": len(keep) - len(train) - len(val),
        "train_validation_ord_id_overlap": len(set(x[0] for x in train) & set(x[0] for x in val)),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    manifest = dict(source_manifest)
    manifest.update(format="pythia31m_ord_full_sequence_v1", context=1024, training_updates=1220,
                    train_file_template="train_seed{seed}_tokens.npy", seeds=[42],
                    primary_validation_name="openwebmath", loss_mask="all_1023_targets",
                    supervised_tokens_per_row=1023, new_fraction=.8, old_fraction=.2,
                    secondary_validations={"wikitext": "wikitext_tokens.npy"})
    manifest.pop("train_source_ids_template", None); manifest.pop("train_component_ids_template", None)
    manifest["file_sha256"] = {p.name: sha256_file(p) for p in out.iterdir() if p.is_file() and p.name != "manifest.json"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
