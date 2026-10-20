#!/usr/bin/env python3
"""Prepare four structured-text CPT datasets on the existing paired 80/20 schedule.

Source files are deliberately kept under data/real_task_domain_inputs_v2/raw.
This script never trains a model and never overwrites an existing task cache.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path

from transformers import AutoTokenizer

from multidataset_cpt_data import collect_windows
from prepare_real_task_domain_inputs import BASE, NEW_ROWS, VALIDATION_ROWS, bucket, save_task
from smoltalk_data import sha256_file

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data/real_task_domain_inputs_v2"
RAW = OUTPUT / "raw"


def _records_sdf(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = []
        for line in handle:
            if line.strip() == "$$$$":
                if lines:
                    text = "".join(lines).strip()
                    # Keep the actual molfile and properties; do not inject synthetic labels.
                    rid = hashlib.sha256(text.encode()).hexdigest()
                    yield rid, text + "\n$$$$\n"
                lines = []
            else:
                lines.append(line)


def _records_uniprot(path: Path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = []
        for line in handle:
            lines.append(line)
            if line.strip() == "//":
                text = "".join(lines)
                accession = next((part.split()[1].rstrip(";") for part in lines if part.startswith("AC   ")), None)
                if accession:
                    yield accession, text
                lines = []


def _records_cve(path: Path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json") or "/cves/" not in member.name:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                record = json.load(handle)
            except (ValueError, UnicodeError):
                continue
            rid = record.get("cveMetadata", {}).get("cveId")
            if rid and record.get("cveMetadata", {}).get("state") == "PUBLISHED":
                yield rid, json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _records_pdb(path: Path):
    for cif in sorted(path.glob("*.cif.gz")):
        try:
            with gzip.open(cif, "rt", encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except (OSError, EOFError):
            continue
        if text.strip():
            # Large ribosomal structures can be tens of MB. Use three fixed
            # excerpts (header, middle, tail) so one structure stays one
            # document without a pathological tokenizer batch.
            if len(text) > 96_000:
                middle = len(text) // 2
                text = text[:32_000] + "\n# [MIDDLE EXCERPT]\n" + text[middle:middle + 32_000] + "\n# [TAIL EXCERPT]\n" + text[-32_000:]
            yield cif.name.removesuffix(".cif.gz").upper(), text


def _source(task: str, raw: Path):
    specs = {
        "chembl": ("chembl_37.sdf.gz", _records_sdf, "ChEMBL 37 SDF", "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37.sdf.gz"),
        "uniprot": ("uniprot_sprot.dat.gz", _records_uniprot, "UniProtKB Swiss-Prot flatfile", "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz"),
        "cve": ("cvelistV5_main.tar.gz", _records_cve, "CVE published JSON records", "https://github.com/CVEProject/cvelistV5"),
        "pdb": ("pdb_mmcif", _records_pdb, "RCSB PDB mmCIF", "https://files.rcsb.org/download/"),
    }
    name, reader, source_name, source_url = specs[task]
    source = raw / name
    if not source.exists():
        raise FileNotFoundError(source)
    return source, reader(source), source_name, source_url


def prepare(task: str, raw: Path, output: Path, tokenizer):
    source, _, source_name, source_url = _source(task, raw)
    # Two explicit document streams prevent one record entering both splits.
    # The archive is reread, trading I/O for bounded memory usage.
    def rows(validation: bool):
        seen_ids = set()
        seen_text = set()
        for rid, content in _source(task, raw)[1]:
            if (bucket(rid) == 0) != validation or rid in seen_ids:
                continue
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in seen_text:
                continue
            seen_ids.add(rid)
            seen_text.add(digest)
            yield {"id": rid, "text": content}

    ident = lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    validation, val_stats = collect_windows(rows(True), tokenizer, count=VALIDATION_ROWS, identity=ident, encode_batch_size=16)
    training, train_stats = collect_windows(rows(False), tokenizer, count=NEW_ROWS, identity=ident, encode_batch_size=16)
    audit = {
        "source_name": source_name, "source": source_url,
        "source_snapshot": sha256_file(source) if source.is_file() else sha256_file(raw / "pdb_entries.idx"),
        "source_path": str(source), "source_sha256": sha256_file(source) if source.is_file() else None,
        "new_documents_consumed": train_stats.documents,
        "validation_documents_consumed": val_stats.documents,
        "new_document_fingerprint": train_stats.document_fingerprint,
        "validation_document_fingerprint": val_stats.document_fingerprint,
        "split": "sha256(record_id) first uint32 modulo 20; zero is validation",
        "render": ("PDB three 32k-character excerpts if longer than 96k; "
                   "otherwise source native text; CVE compact sorted JSON"),
    }
    destination = save_task(task, training, validation, audit, output)
    return {"task": task, "destination": str(destination), "train_documents": train_stats.documents,
            "validation_documents": val_stats.documents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("chembl", "pdb", "uniprot", "cve", "all"), required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    args = parser.parse_args()
    base = json.loads((BASE / "manifest.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(base["tokenizer_id"], revision=base["tokenizer_revision"])
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = ("chembl", "pdb", "uniprot", "cve") if args.task == "all" else (args.task,)
    for task in tasks:
        print(json.dumps(prepare(task, args.raw_root, args.output_root, tokenizer)), flush=True)


if __name__ == "__main__":
    main()
