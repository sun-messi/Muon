#!/usr/bin/env python3
"""Matched CPT tapes from four components of Pythia's pretraining corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from multidataset_cpt_data import collect_windows
from openwebmath_pile_replay_data import pile_document_identity
import prepare_real_task_eight_inputs as historical
from prepare_real_task_domain_inputs import BASE, NEW_ROWS, VALIDATION_ROWS, bucket, save_task
from smoltalk_data import sha256_file

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data/real_task_domain_inputs_v3"
RAW = OUTPUT / "raw"
COMPONENTS = {"pile_cc": "Pile-CC", "openwebtext2": "OpenWebText2",
              "books3": "Books3", "pubmed_central": "PubMed Central"}
REVISION = "05b327037e6301f256d8df32193756edc4c8e3bd"


def local_rows(split, seed, buffer_size):
    path = RAW / ("pile_val.jsonl" if split == "validation" else "pile_test.jsonl")
    return load_dataset("json", data_files={split: str(path)}, split=split,
                        streaming=True).shuffle(seed=seed, buffer_size=buffer_size)


def exclusions(tokenizer):
    path = OUTPUT / "old_document_exclusions.json"
    if path.exists():
        return set(json.loads(path.read_text())["identities"])
    historical.pile_rows = local_rows
    # This function checks the reconstructed replay pool against its historical
    # document-set digest before returning any IDs.
    forbidden = historical.old_pile_identities(tokenizer)
    payload = dict(identities=sorted(forbidden),
                   reconstruction="historical old_pile_identities using identical local JSON order and shuffle",
                   old_manifest_sha256=sha256_file(BASE / "manifest.json"),
                   val_sha256=sha256_file(RAW / "pile_val.jsonl"),
                   test_sha256=sha256_file(RAW / "pile_test.jsonl"))
    path.write_text(json.dumps(payload, indent=2)+"\n")
    print(f"Reconstructed and verified {len(forbidden)} excluded Old/evaluation documents", flush=True)
    return forbidden


def prepare(task, tokenizer, forbidden):
    component = COMPONENTS[task]
    used = {False: [], True: []}
    examples = []

    def rows(validation):
        seen = set()
        for row in local_rows("validation", 8_800_042, 50_000):
            if row["meta"]["pile_set_name"] != component:
                continue
            identity = pile_document_identity(row)
            if identity in forbidden or identity in seen or (bucket(identity) == 0) != validation:
                continue
            text = str(row.get("text") or "")
            if not text.strip():
                continue
            seen.add(identity)
            used[validation].append(identity)
            if not validation and len(examples) < 2:
                examples.append(dict(id=identity, text=text[:6000]))
            yield {"id": identity, "text": text}

    validation, val_stats = collect_windows(rows(True), tokenizer, count=VALIDATION_ROWS,
                                             identity=lambda row: row["id"], encode_batch_size=16)
    training, train_stats = collect_windows(rows(False), tokenizer, count=NEW_ROWS,
                                            identity=lambda row: row["id"], encode_batch_size=16)
    used[False] = used[False][:train_stats.documents]
    used[True] = used[True][:val_stats.documents]
    assert not (set(used[True]) & set(used[False]))
    assert not ((set(used[True]) | set(used[False])) & forbidden)
    audit = dict(source_name=f"The Pile / {component}", source="https://huggingface.co/datasets/EleutherAI/pile_val_test",
                 source_revision=REVISION, component=component, source_split="validation",
                 source_role="same component source as Pythia pretraining; documents from held-out Pile validation",
                 split="normalized document hash modulo 20: zero New validation; excludes verified Old/evaluation documents",
                 forbidden_document_count=len(forbidden),
                 forbidden_identities_sha256=sha256_file(OUTPUT / "old_document_exclusions.json"),
                 raw_val_sha256=sha256_file(RAW / "pile_val.jsonl"),
                 new_documents_consumed=train_stats.documents,
                 validation_documents_consumed=val_stats.documents,
                 new_document_fingerprint=train_stats.document_fingerprint,
                 validation_document_fingerprint=val_stats.document_fingerprint,
                 render="unaltered source text; EOS-separated packing into 1025-token windows",
                 same_source_does_not_imply_individual_documents_seen_in_pretraining=True)
    destination = save_task(task, training, validation, audit, OUTPUT)
    (destination / "document_ids.json").write_text(json.dumps(
        {"train": used[False], "validation": used[True]}, indent=2)+"\n")
    (destination / "examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(dict(task=task, destination=str(destination),
                          train_documents=train_stats.documents, validation_documents=val_stats.documents)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*COMPONENTS, "exclusions", "all"), required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = json.loads((BASE / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(base["tokenizer_id"], revision=base["tokenizer_revision"], local_files_only=True)
    forbidden = exclusions(tokenizer)
    if args.task == "exclusions":
        return
    tasks = COMPONENTS if args.task == "all" else (args.task,)
    for task in tasks:
        if (OUTPUT / task).exists():
            raise FileExistsError(OUTPUT / task)
        prepare(task, tokenizer, forbidden)


if __name__ == "__main__":
    main()
