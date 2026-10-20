#!/usr/bin/env python3
"""Download a deterministic, resumable subset of official RCSB mmCIF files."""
from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    index = args.raw_root / "pdb_entries.idx"
    target = args.raw_root / "pdb_mmcif"
    target.mkdir(parents=True, exist_ok=True)
    ids = [line.split("\t", 1)[0].lower() for line in index.read_text(errors="replace").splitlines()[2:]
           if "\t" in line]
    # Stable selection over the entire index, rather than a time-ordered prefix.
    ids = sorted(set(ids), key=lambda name: hashlib.sha256(name.encode()).digest())[:args.count]

    def fetch(pdb_id):
        destination = target / f"{pdb_id}.cif.gz"
        if destination.exists():
            try:
                with gzip.open(destination, "rb") as handle:
                    handle.read(1)
                return True
            except OSError:
                destination.unlink()
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            if not content.startswith(b"\x1f\x8b"):
                return False
            temporary = destination.with_suffix(".partial")
            temporary.write_bytes(content)
            temporary.replace(destination)
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(fetch, ids))
    print(json.dumps({"selected": len(ids), "downloaded": sum(outcomes),
                      "failed_ids": [name for name, ok in zip(ids, outcomes) if not ok],
                      "directory": str(target)}, indent=2))


if __name__ == "__main__":
    main()
