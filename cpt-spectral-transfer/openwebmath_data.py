"""Deterministic OpenWebMath stream construction for Pythia-31M CPT."""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass
from typing import Iterable

import numpy as np


CONTEXT = 1024
GLOBAL_SEQUENCE_BATCH = 32
TRAINING_UPDATES = 610
TRAIN_TARGETS = TRAINING_UPDATES * GLOBAL_SEQUENCE_BATCH * CONTEXT
VALIDATION_ROWS = 977
VALIDATION_TARGETS = VALIDATION_ROWS * CONTEXT


@dataclass
class StreamStats:
    documents: int
    stream_tokens: int
    target_tokens: int
    document_fingerprint: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "documents": self.documents,
            "stream_tokens": self.stream_tokens,
            "target_tokens": self.target_tokens,
            "document_fingerprint": self.document_fingerprint,
        }


def document_identity(row: dict) -> str:
    """Return a stable identity without relying on streaming row numbers."""

    url = str(row.get("url") or "")
    text = str(row.get("text") or "")
    return hashlib.sha256((url + "\0" + text).encode("utf-8")).hexdigest()


def encode_document(tokenizer, row: dict) -> list[int]:
    text = str(row.get("text") or "")
    if not text.strip():
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Pythia tokenizer must define EOS")
    token_ids.append(int(tokenizer.eos_token_id))
    if token_ids and (min(token_ids) < 0 or max(token_ids) >= 2**16):
        raise ValueError("token id does not fit uint16")
    return token_ids


def take_stream(
    rows: Iterable[dict],
    tokenizer,
    *,
    stream_tokens: int,
    forbidden_documents: set[str] | None = None,
) -> tuple[np.ndarray, StreamStats, set[str]]:
    """Consume whole documents until enough tokens exist, then trim the tail."""

    if stream_tokens < 2:
        raise ValueError("stream_tokens must be at least 2")
    values = array("H")
    identities: set[str] = set()
    fingerprint = hashlib.sha256()
    documents = 0
    for row in rows:
        identity = document_identity(row)
        if forbidden_documents is not None and identity in forbidden_documents:
            raise RuntimeError("training and validation document sets overlap")
        token_ids = encode_document(tokenizer, row)
        if not token_ids:
            continue
        identities.add(identity)
        fingerprint.update(identity.encode("ascii"))
        values.extend(token_ids)
        documents += 1
        if len(values) >= stream_tokens:
            break
    if len(values) < stream_tokens:
        raise RuntimeError(f"dataset exhausted at {len(values):,}/{stream_tokens:,} tokens")
    result = np.frombuffer(values, dtype=np.uint16, count=stream_tokens).copy()
    stats = StreamStats(
        documents=documents,
        stream_tokens=len(result),
        target_tokens=len(result) - 1,
        document_fingerprint=fingerprint.hexdigest(),
    )
    return result, stats, identities

