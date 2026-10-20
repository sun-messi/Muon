#!/usr/bin/env python3
"""Build missing seed-42 domains for the paired eight-topic CPT study."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from multidataset_cpt_data import collect_windows
from openwebmath_pile_replay_data import (
    PILE_BALANCED_ROWS_PER_SOURCE, PILE_WEIGHTED_ROWS, PILE_WEIGHTS,
    collect_component_windows, document_set_digest, largest_remainder_counts,
    pile_document_identity, rho_spec,
)
from prepare_openwebmath_pile_replay_rho import pile_rows
from prepare_real_task_domain_inputs import BASE, OUTPUT, NEW_ROWS, VALIDATION_ROWS, bucket, save_task

WIKI = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3")
PUBMED = ("qiaojin/PubMedQA", "9001f2853fb87cab8d220904e0de81ac6973b318")
CODE = ("code_search_net", "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555")


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def wikitext_articles():
    source = load_dataset(WIKI[0], "wikitext-103-raw-v1", split="train",
                          revision=WIKI[1], streaming=True)
    title, content = None, []
    for row in source:
        line = str(row["text"])
        if re.fullmatch(r"\s*= [^=]+ =\s*", line):
            if title and content:
                yield {"id": title, "text": "\n".join(content)}
            title, content = line.strip(), [line]
        elif title:
            content.append(line)
    if title and content:
        yield {"id": title, "text": "\n".join(content)}


def pubmed_rows():
    source = load_dataset(PUBMED[0], "pqa_artificial", split="train",
                          revision=PUBMED[1], streaming=True)
    for row in source:
        context = row.get("context") or {}
        parts = context.get("contexts") or []
        text = ("Question:\n" + str(row.get("question") or "") +
                "\n\nContext:\n" + "\n".join(parts) +
                "\n\nAnswer:\n" + str(row.get("long_answer") or ""))
        yield {"id": str(row["pubid"]), "text": text}


def code_rows(split: str):
    source = load_dataset(CODE[0], "python", split=split,
                          revision=CODE[1], streaming=True)
    for row in source:
        code = str(row.get("whole_func_string") or "")
        ident = str(row.get("func_code_url") or "")
        if code.strip() and ident:
            yield {"id": ident, "text": code}


def old_pile_identities(tokenizer) -> set[str]:
    """Reconstruct the exact seed-42 replay-document pool; verify its digest."""
    test_counts = largest_remainder_counts(PILE_WEIGHTED_ROWS, PILE_WEIGHTS)
    test_requirements = {name: max(test_counts[name], PILE_BALANCED_ROWS_PER_SOURCE)
                         for name in PILE_WEIGHTS}
    _, test_ids = collect_component_windows(
        pile_rows("test", 6_100_042, 50_000), tokenizer, test_requirements)
    test_set = {identity for values in test_ids.values() for identity in values}
    diagnostic_counts = largest_remainder_counts(32, PILE_WEIGHTS)
    _, diagnostic_ids = collect_component_windows(
        pile_rows("validation", 6_200_042, 50_000), tokenizer,
        diagnostic_counts, forbidden_identities=test_set)
    diagnostic_set = {identity for values in diagnostic_ids.values() for identity in values}
    maximum = largest_remainder_counts(rho_spec(.40).pile_rows, PILE_WEIGHTS)
    _, replay_ids = collect_component_windows(
        pile_rows("validation", 42, 50_000), tokenizer,
        maximum, forbidden_identities=test_set | diagnostic_set)
    expected = json.loads((BASE / "manifest.json").read_text())["training"]["42"]["pile_validation_document_set_sha256"]
    observed = document_set_digest(replay_ids)
    if observed != expected:
        raise RuntimeError(f"replayed Old document fingerprint differs: {observed} != {expected}")
    return test_set | diagnostic_set | {x for values in replay_ids.values() for x in values}


def pile_new_rows(forbidden: set[str], validation: bool):
    # Same Pile split as replay, but exact document IDs are excluded.
    # Stable hash partition keeps New training and validation document-disjoint.
    seen = set()
    for row in pile_rows("validation", 8_800_042, 50_000):
        ident = pile_document_identity(row)
        if ident in forbidden or ident in seen or (bucket(ident) == 0) != validation:
            continue
        seen.add(ident)
        text = str(row.get("text") or "")
        if text.strip():
            yield {"id": ident, "text": text}


def build(task: str, tokenizer):
    if task == "wikitext":
        source, revision = WIKI
        def rows(validation):
            return (r for r in wikitext_articles() if (bucket(r["id"]) == 0) == validation)
        identity = lambda r: sha(r["id"])
        extra = {"source_name": "WikiText-103 raw", "source": source,
                 "source_revision": revision, "split": "article title hash 5% validation"}
    elif task == "pubmedqa":
        source, revision = PUBMED
        def rows(validation):
            return (r for r in pubmed_rows() if (bucket(r["id"]) == 0) == validation)
        identity = lambda r: sha(r["id"])
        extra = {"source_name": "PubMedQA pqa_artificial", "source": source,
                 "source_revision": revision, "split": "PMID hash 5% validation"}
    elif task == "codesearchnet":
        source, revision = CODE
        def rows(validation):
            return code_rows("validation" if validation else "train")
        identity = lambda r: sha(r["id"])
        extra = {"source_name": "CodeSearchNet Python", "source": source,
                 "source_revision": revision, "split": "official train/validation"}
    elif task == "pile_reference":
        forbidden = old_pile_identities(tokenizer)
        def rows(validation):
            return pile_new_rows(forbidden, validation)
        identity = lambda r: r["id"]
        extra = {"source_name": "Pile validation disjoint-doc reference",
                 "source": "EleutherAI/pile_val_test",
                 "source_revision": "05b327037e6301f256d8df32193756edc4c8e3bd",
                 "split": "validation; excludes reconstructed Old replay and test/diagnostic documents; 5% hash New validation",
                 "forbidden_document_count": len(forbidden)}
    else:
        raise ValueError(task)
    validation, val_stats = collect_windows(rows(True), tokenizer, count=VALIDATION_ROWS,
                                            identity=identity)
    training, train_stats = collect_windows(rows(False), tokenizer, count=NEW_ROWS,
                                            identity=identity)
    audit = {**extra, "new_documents_consumed": train_stats.documents,
             "validation_documents_consumed": val_stats.documents,
             "new_document_fingerprint": train_stats.document_fingerprint,
             "validation_document_fingerprint": val_stats.document_fingerprint}
    return training, validation, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("wikitext", "pubmedqa", "codesearchnet", "pile_reference"), required=True)
    args = parser.parse_args()
    if (OUTPUT / args.task).exists():
        raise FileExistsError(OUTPUT / args.task)
    base = json.loads((BASE / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(base["tokenizer_id"], revision=base["tokenizer_revision"])
    training, validation, audit = build(args.task, tokenizer)
    destination = save_task(args.task, training, validation, audit, OUTPUT)
    print(json.dumps({"task": args.task, "destination": str(destination),
                      "new_docs": audit["new_documents_consumed"],
                      "val_docs": audit["validation_documents_consumed"]}), flush=True)


if __name__ == "__main__":
    main()
