#!/usr/bin/env python3
"""Prepare just enough Chemistry StackExchange and ORD text for paired CPT.

This is data preparation only: no model or optimizer is loaded.  Both tasks use
the existing seed-42 OpenWebMath/Pile 80:20 schedule and its exact Old rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from multidataset_cpt_data import collect_windows
from smoltalk_data import sha256_file

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data/openwebmath_pile_replay_rho_sweep_20m_v1"
RAW = ROOT / "data/real_task_domain_inputs_v1/raw"
OUTPUT = ROOT / "data/real_task_domain_inputs_v1"
TRAIN_NAME = "train_seed42_rho20_tokens.npy"
SOURCE_NAME = "train_seed42_rho20_source_ids.npy"
COMPONENT_NAME = "train_seed42_rho20_pile_component_ids.npy"
NEW_ROWS = 15_616
VALIDATION_ROWS = 977


def bucket(identity: str) -> int:
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 20


def plain_html(value: str) -> str:
    return BeautifulSoup(value or "", "html.parser").get_text(" ", strip=True)


def chemistry_records(archive: Path) -> tuple[dict[int, dict], dict[int, list[dict]], dict]:
    process = subprocess.Popen(
        ["bsdtar", "-xOf", str(archive), "Posts.xml"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    questions: dict[int, dict] = {}
    answers: dict[int, list[dict]] = defaultdict(list)
    try:
        for _, elem in ET.iterparse(process.stdout, events=("end",)):
            if elem.tag != "row":
                elem.clear()
                continue
            row = elem.attrib
            kind = row.get("PostTypeId")
            if kind == "1":
                questions[int(row["Id"])] = {
                    "title": row.get("Title", ""), "body": row.get("Body", ""),
                }
            elif kind == "2" and row.get("ParentId"):
                answers[int(row["ParentId"])].append({
                    "id": int(row["Id"]), "score": int(row.get("Score", "0")),
                    "body": row.get("Body", ""),
                })
            elem.clear()
    finally:
        process.stdout.close()
    error = process.stderr.read().decode(errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"Chemistry archive extraction failed: {error[-2000:]}")
    stats = {
        "questions_in_dump": len(questions),
        "answers_in_dump": sum(map(len, answers.values())),
        "questions_with_answers": sum(bool(answers.get(qid)) for qid in questions),
    }
    return questions, answers, stats


def chemistry_row(qid: int, questions: dict, answers: dict) -> str:
    q = questions[qid]
    parts = ["Question:", q["title"], plain_html(q["body"])]
    for answer in sorted(answers[qid], key=lambda item: (-item["score"], item["id"])):
        body = plain_html(answer["body"])
        if body:
            parts.extend(["", "Answer:", body])
    return "\n".join(parts).strip()


def prepare_chemistry(tokenizer, raw: Path):
    archive = raw / "chemistry.stackexchange.com.7z"
    questions, answers, source_stats = chemistry_records(archive)
    ids = np.array(sorted(qid for qid in questions if answers.get(qid)), dtype=np.int64)
    order = np.random.default_rng(42_026).permutation(ids)
    seen_text: set[str] = set()
    counts = {"duplicate_rendered_text": 0, "empty_rendered_text": 0}

    def rows(validation: bool):
        for value in order:
            qid = int(value)
            if (bucket(str(qid)) == 0) != validation:
                continue
            text = chemistry_row(qid, questions, answers)
            if not text:
                counts["empty_rendered_text"] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_text:
                counts["duplicate_rendered_text"] += 1
                continue
            seen_text.add(digest)
            yield {"id": str(qid), "text": text}

    validation, val_stats = collect_windows(
        rows(True), tokenizer, count=VALIDATION_ROWS,
        identity=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    training, train_stats = collect_windows(
        rows(False), tokenizer, count=NEW_ROWS,
        identity=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    return training, validation, {
        **source_stats, **counts, "new_documents_consumed": train_stats.documents,
        "validation_documents_consumed": val_stats.documents,
        "new_document_fingerprint": train_stats.document_fingerprint,
        "validation_document_fingerprint": val_stats.document_fingerprint,
        "raw_sha256": sha256_file(archive),
        "render": "question title + body + all answers in descending score order; HTML stripped",
        "split": "sha256(question_id) first uint32 modulo 20; zero is validation",
        "source": "https://archive.org/details/stackexchange_20260930",
        "source_snapshot": "2026-09-30",
    }


def prepare_ord(tokenizer, raw: Path):
    source = raw / "ord_uspto_mit.parquet"
    sys.path.insert(0, str(ROOT / "data/ord_small/deps"))
    from ord_schema.proto import reaction_pb2
    from prepare_ord_full_sequence import render

    table = pq.read_table(source, columns=["reaction_id", "reaction"]).to_pydict()
    ids, blobs = table["reaction_id"], table["reaction"]
    order = np.random.default_rng(42_026).permutation(len(ids))
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    counts = {"duplicate_reaction_id": 0, "duplicate_rendered_text": 0,
              "no_reactant_or_product": 0}

    def rows(validation: bool):
        for value in order:
            index = int(value)
            rid = str(ids[index])
            if (bucket(rid) == 0) != validation:
                continue
            if rid in seen_ids:
                counts["duplicate_reaction_id"] += 1
                continue
            record = reaction_pb2.Reaction()
            record.ParseFromString(blobs[index])
            text = render(record)
            if "reactants: none" in text and "products: none" in text:
                counts["no_reactant_or_product"] += 1
                continue
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen_text:
                counts["duplicate_rendered_text"] += 1
                continue
            seen_ids.add(rid)
            seen_text.add(digest)
            yield {"id": rid, "text": text}

    validation, val_stats = collect_windows(
        rows(True), tokenizer, count=VALIDATION_ROWS,
        identity=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    training, train_stats = collect_windows(
        rows(False), tokenizer, count=NEW_ROWS,
        identity=lambda row: hashlib.sha256(row["id"].encode()).hexdigest(),
    )
    return training, validation, {
        "source_reaction_rows": len(ids), **counts,
        "new_documents_consumed": train_stats.documents,
        "validation_documents_consumed": val_stats.documents,
        "new_document_fingerprint": train_stats.document_fingerprint,
        "validation_document_fingerprint": val_stats.document_fingerprint,
        "raw_sha256": sha256_file(source),
        "render": "prepare_ord_full_sequence.render (same fields and truncation)",
        "split": "sha256(reaction_id) first uint32 modulo 20; zero is validation",
        "source": "https://huggingface.co/datasets/open-reaction-database/ord-data",
        "source_revision": "46980f8cb68aa43ed9707edec319d4eac5a2bfed",
        "source_config": "uspto-mit",
    }


def save_task(task: str, training: np.ndarray, validation: np.ndarray,
              source_stats: dict, output_root: Path) -> Path:
    destination = output_root / task
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    source_ids = np.load(BASE / SOURCE_NAME, mmap_mode="r")
    base_train = np.load(BASE / TRAIN_NAME, mmap_mode="r")
    if source_ids.shape != (19_520,) or base_train.shape != (19_520, 1_025):
        raise ValueError("historical 80/20 cache has unexpected dimensions")
    if (int((source_ids == 0).sum()), int((source_ids == 1).sum())) != (NEW_ROWS, 3_904):
        raise ValueError("historical 80/20 source counts changed")
    if training.shape != (NEW_ROWS, 1_025) or validation.shape != (VALIDATION_ROWS, 1_025):
        raise ValueError("new or validation window shape is wrong")
    temporary = Path(tempfile.mkdtemp(prefix=f".{task}.partial.", dir=output_root))
    for name in (SOURCE_NAME, COMPONENT_NAME, "retention_tokens.npy", "wikitext_tokens.npy"):
        os.link(BASE / name, temporary / name)
    mixed = np.array(base_train)
    mixed[source_ids == 0] = training
    if not np.array_equal(mixed[source_ids == 1], base_train[source_ids == 1]):
        raise AssertionError("Old replay rows changed")
    np.save(temporary / TRAIN_NAME, mixed)
    np.save(temporary / "validation_tokens.npy", validation)
    base = json.loads((BASE / "manifest.json").read_text())
    names = (TRAIN_NAME, SOURCE_NAME, COMPONENT_NAME, "validation_tokens.npy",
             "retention_tokens.npy", "wikitext_tokens.npy")
    source_name = source_stats.get("source_name", {
        "chemistry": "Chemistry StackExchange", "ord": "ORD uspto-mit",
    }.get(task, task))
    manifest = {key: base[key] for key in (
        "context", "eos_token_id", "loss_mask", "model_id", "model_revision",
        "tokenizer_id", "tokenizer_revision", "vocab_size")}
    manifest.update(
        format="pythia31m_paired_real_task_domain_v1",
        dataset_id=f"{source_name}+EleutherAI/pile_val_test",
        dataset_revision=(source_stats.get("source_snapshot") or source_stats.get("source_revision"))
                         + "+" + base["source_datasets"]["pile_replay"]["revision"],
        primary_validation_name=task, retention_validation_name="pile",
        secondary_validations={"wikitext": "wikitext_tokens.npy"},
        train_file_template="train_seed{seed}_{rho_label}_tokens.npy",
        train_source_ids_template="train_seed{seed}_{rho_label}_source_ids.npy",
        train_component_ids_template="train_seed{seed}_{rho_label}_pile_component_ids.npy",
        training_updates=610, seeds=[42],
        file_sha256={name: sha256_file(temporary / name) for name in names},
        source_audit=source_stats,
        pairing={
            "base_manifest_sha256": sha256_file(BASE / "manifest.json"),
            "new_rows": NEW_ROWS, "old_rows": 3_904,
            "new_target_tokens": NEW_ROWS * 1_024,
            "old_target_tokens": 3_904 * 1_024,
            "source_ids_sha256": sha256_file(BASE / SOURCE_NAME),
            "old_rows_sha256": hashlib.sha256(np.asarray(base_train[source_ids == 1]).tobytes()).hexdigest(),
            "old_exact": True,
        },
    )
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.rename(temporary, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chemistry", "ord", "all"), required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = json.loads((BASE / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        base["tokenizer_id"], revision=base["tokenizer_revision"])
    tasks = ("chemistry", "ord") if args.task == "all" else (args.task,)
    for task in tasks:
        builder = prepare_chemistry if task == "chemistry" else prepare_ord
        training, validation, audit = builder(tokenizer, args.raw_root)
        destination = save_task(task, training, validation, audit, args.output_root)
        print(json.dumps({"task": task, "destination": str(destination),
                          "new_documents": audit["new_documents_consumed"],
                          "validation_documents": audit["validation_documents_consumed"]},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
