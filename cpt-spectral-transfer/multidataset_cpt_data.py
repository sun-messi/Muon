"""Deterministic 20M-token CPT caches for several mathematical corpora."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import zstandard as zstd
from huggingface_hub import HfApi, hf_hub_download

from smoltalk_data import sha256_file, write_json


MODEL_ID = "EleutherAI/pythia-31m"
MODEL_REVISION = "e556ace21b489575e94e9d50b6dad2fcc7419679"
CONTEXT = 1024
ROWS_PER_UPDATE = 32
TRAINING_UPDATES = 610
TRAIN_ROWS = ROWS_PER_UPDATE * TRAINING_UPDATES
VALIDATION_ROWS = 977
SEEDS = (42, 43, 44, 45)
FORMAT = "pythia31m_multidataset_cpt_cache_v1"

FINE_MATH_ID = "HuggingFaceTB/finemath"
FINE_MATH_REVISION = "e92b25a616738fe95dc186b64dfb19f9c8525594"
PROOF_PILE_ID = "EleutherAI/proof-pile-2"
PROOF_PILE_REVISION = "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b"
STACK_MATH_ID = "math-ai/StackMathQA"
STACK_MATH_REVISION = "5a3e18f6fa122652e1959c3cc3e714758778fdad"
MATH_PILE_ID = "GAIR/MathPile"
MATH_PILE_REVISION = "bde88d8c04147bdca0a569839a61067ac41f6bd2"

# Published corpus token ratios.  Proof-Pile-2 reports 29B/15B/11B.  MathPile
# reports the exact counts below in its dataset paper.
PROOF_WEIGHTS = {"arxiv": 29, "open-web-math": 15, "algebraic-stack": 11}
MATHPILE_WEIGHTS = {
    "textbooks": 187_194_060,
    "wikipedia": 59_990_005,
    "proofwiki": 7_608_526,
    "commoncrawl": 615_371_126,
    "stackexchange": 253_021_062,
    "arXiv": 8_324_324_917,
}


@dataclass
class WindowStats:
    documents: int
    rows: int
    target_tokens: int
    document_fingerprint: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "documents": self.documents,
            "rows": self.rows,
            "target_tokens": self.target_tokens,
            "document_fingerprint": self.document_fingerprint,
        }


def largest_remainder(total: int, weights: dict[str, int]) -> dict[str, int]:
    denominator = sum(weights.values())
    raw = {key: total * value / denominator for key, value in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: raw[key] - result[key], reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    if sum(result.values()) != total:
        raise AssertionError("largest-remainder allocation failed")
    return result


def fine_identity(row: dict) -> str:
    return hashlib.sha256((str(row.get("url", "")) + "\0" + str(row.get("text", ""))).encode()).hexdigest()


def stack_identity(row: dict) -> str:
    meta = row.get("meta") or {}
    value = "\0".join(
        [
            str(meta.get("url", "")),
            str(meta.get("answer_id", "")),
            str(row.get("Q", "")),
            str(row.get("A", "")),
        ]
    )
    return hashlib.sha256(value.encode()).hexdigest()


def text_identity(row: dict) -> str:
    meta = row.get("meta") or row.get("metadata") or {}
    value = str(meta) + "\0" + str(row.get("text") or row.get("content") or "")
    return hashlib.sha256(value.encode()).hexdigest()


def render_text(row: dict) -> str:
    return str(row.get("text") or row.get("content") or "")


def render_stackmath(row: dict) -> str:
    return f"Question:\n{row.get('Q', '')}\n\nAnswer:\n{row.get('A', '')}\n"


def is_validation_identity(identity: str) -> bool:
    """Stable five-percent document bucket used when no official split exists."""

    return int(identity[:8], 16) % 20 == 0


def collect_windows(
    rows: Iterable[dict],
    tokenizer,
    *,
    count: int,
    render: Callable[[dict], str] = render_text,
    identity: Callable[[dict], str] = text_identity,
    accept: Callable[[str], bool] | None = None,
    encode_batch_size: int = 128,
) -> tuple[np.ndarray, WindowStats]:
    """Encode whole EOS-delimited documents and return explicit causal windows."""

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Pythia tokenizer must define EOS")
    result = np.empty((count, CONTEXT + 1), dtype=np.uint16)
    pending: list[int] = [int(eos)]
    offset = 0
    written = 0
    documents = 0
    fingerprint = hashlib.sha256()
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    def consume_batch() -> bool:
        nonlocal pending, offset, written, documents
        if not batch_texts:
            return False
        encoded = tokenizer(batch_texts, add_special_tokens=False)["input_ids"]
        for doc_id, token_ids in zip(batch_ids, encoded):
            if token_ids:
                if min(token_ids) < 0 or max(token_ids) >= 2**16:
                    raise ValueError("token ID does not fit uint16")
                pending.extend(int(value) for value in token_ids)
                pending.append(int(eos))
                documents += 1
                fingerprint.update(doc_id.encode("ascii"))
            while len(pending) - offset >= CONTEXT + 1 and written < count:
                result[written] = np.asarray(pending[offset : offset + CONTEXT + 1], dtype=np.uint16)
                offset += CONTEXT
                written += 1
            if offset >= 1 << 18:
                pending = pending[offset:]
                offset = 0
            if written == count:
                return True
        batch_texts.clear()
        batch_ids.clear()
        return False

    for row in rows:
        doc_id = identity(row)
        if accept is not None and not accept(doc_id):
            continue
        text = render(row)
        if not text.strip():
            continue
        batch_texts.append(text)
        batch_ids.append(doc_id)
        if len(batch_texts) >= encode_batch_size and consume_batch():
            break
    if written < count:
        consume_batch()
    if written != count:
        raise RuntimeError(f"dataset exhausted at {written:,}/{count:,} windows")
    return result, WindowStats(documents, count, count * CONTEXT, fingerprint.hexdigest())


def iter_jsonl_files(repo: str, revision: str, paths: list[str], *, seed: int) -> Iterator[dict]:
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    for relative in paths:
        local = Path(hf_hub_download(repo, relative, repo_type="dataset", revision=revision))
        # A few empty Git-LFS placeholders exist in Proof-Pile-2.
        if local.stat().st_size < 100:
            continue
        if relative.endswith(".zst"):
            raw = local.open("rb")
            binary = zstd.ZstdDecompressor().stream_reader(raw)
            handle = io.TextIOWrapper(binary, encoding="utf-8")
        elif relative.endswith(".gz"):
            handle = gzip.open(local, "rt", encoding="utf-8")
        else:
            handle = local.open(encoding="utf-8")
        try:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        finally:
            handle.close()


def proof_paths(split: str, component: str) -> list[str]:
    prefix = f"{component}/{split}/"
    files = HfApi().list_repo_files(PROOF_PILE_ID, repo_type="dataset", revision=PROOF_PILE_REVISION)
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(".jsonl.zst"))


def mathpile_paths(split: str, component: str) -> list[str]:
    prefix = f"{split}/{component}/"
    files = HfApi().list_repo_files(MATH_PILE_ID, repo_type="dataset", revision=MATH_PILE_REVISION)
    return sorted(path for path in files if path.startswith(prefix) and path.endswith(".jsonl.gz"))


def permute_and_save(pool: np.ndarray, output: Path, seeds: tuple[int, ...]) -> tuple[list[Path], dict[str, dict]]:
    paths: list[Path] = []
    stats: dict[str, dict] = {}
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(pool))
        path = output / f"train_seed{seed}_tokens.npy"
        np.save(path, pool[order])
        paths.append(path)
        stats[str(seed)] = {
            "rows": len(pool),
            "target_tokens": len(pool) * CONTEXT,
            "order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        }
    return paths, stats


def finalize_cache(
    output: Path,
    tokenizer,
    *,
    dataset_id: str,
    dataset_revision: str,
    primary_name: str,
    train_pool: np.ndarray,
    validation: np.ndarray,
    train_stats: dict,
    validation_stats: dict,
    retention_source: Path,
    source_config: dict,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    train_paths, training = permute_and_save(train_pool, output, seeds)
    validation_path = output / "validation_tokens.npy"
    retention_path = output / "retention_tokens.npy"
    np.save(validation_path, validation)
    shutil.copyfile(retention_source, retention_path)
    retention = np.load(retention_path, mmap_mode="r")
    retention_targets = ((len(retention) - 1) // CONTEXT) * CONTEXT if retention.ndim == 1 else len(retention) * CONTEXT
    files = train_paths + [validation_path, retention_path]
    manifest = {
        "format": FORMAT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "context": CONTEXT,
        "seeds": list(seeds),
        "training_updates": TRAINING_UPDATES,
        "train_file_template": "train_seed{seed}_tokens.npy",
        "training": training,
        "training_content": train_stats,
        "validation": validation_stats,
        "monitor_validation_rows": VALIDATION_ROWS,
        "primary_validation_name": primary_name,
        "retention_source": str(retention_source),
        "retention_source_sha256": sha256_file(retention_source),
        "retention_target_tokens": retention_targets,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "loss_mask": "none_all_tokens",
        "document_overlap_count": 0,
        "source_config": source_config,
        "file_sha256": {path.name: sha256_file(path) for path in files},
    }
    write_json(output / "manifest.json", manifest)
    return manifest
