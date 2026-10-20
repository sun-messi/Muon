"""Deterministic OpenWebMath + The Pile replay schedules for a rho sweep."""

from __future__ import annotations

import hashlib
import math
import re
from array import array
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTEXT = 1024
UPDATES = 610
SEQUENCES_PER_UPDATE = 32
TOTAL_ROWS = UPDATES * SEQUENCES_PER_UPDATE
TOTAL_TARGETS = TOTAL_ROWS * CONTEXT
RHO_VALUES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)
RHO_LABELS = {rho: f"rho{int(round(rho * 100)):02d}" for rho in RHO_VALUES}
PILE_WEIGHTED_ROWS = 977
PILE_BALANCED_ROWS_PER_SOURCE = 32

# Official effective mixture weights from The Pile release.  Names match
# ``meta.pile_set_name`` in EleutherAI/pile_val_test.
PILE_WEIGHTS: dict[str, float] = {
    "Pile-CC": 18.11,
    "PubMed Central": 14.40,
    "Books3": 12.07,
    "OpenWebText2": 10.01,
    "ArXiv": 8.96,
    "Github": 7.59,
    "FreeLaw": 6.12,
    "StackExchange": 5.13,
    "USPTO Backgrounds": 3.65,
    "PubMed Abstracts": 3.07,
    "Gutenberg (PG-19)": 2.17,
    "OpenSubtitles": 1.55,
    "Wikipedia (en)": 1.53,
    "DM Mathematics": 1.24,
    "Ubuntu IRC": 0.88,
    "BookCorpus2": 0.75,
    "EuroParl": 0.73,
    "HackerNews": 0.62,
    "YoutubeSubtitles": 0.60,
    "PhilPapers": 0.38,
    "NIH ExPorter": 0.30,
    "Enron Emails": 0.14,
}

PILE_GROUPS: dict[str, tuple[str, ...]] = {
    "general_web": ("Pile-CC", "OpenWebText2", "Wikipedia (en)", "HackerNews"),
    "prose_dialogue": (
        "Books3",
        "Gutenberg (PG-19)",
        "BookCorpus2",
        "OpenSubtitles",
        "YoutubeSubtitles",
        "Ubuntu IRC",
    ),
    "stem_technical": (
        "PubMed Central",
        "ArXiv",
        "Github",
        "StackExchange",
        "PubMed Abstracts",
        "DM Mathematics",
        "PhilPapers",
        "NIH ExPorter",
    ),
    "legal_institutional": ("FreeLaw", "USPTO Backgrounds", "EuroParl", "Enron Emails"),
}


@dataclass(frozen=True)
class RhoSpec:
    rho: float
    label: str
    math_rows: int
    pile_rows: int
    low_pile_per_batch: int
    high_pile_per_batch: int
    low_batches: int
    high_batches: int

    def as_dict(self) -> dict[str, int | float | str]:
        return self.__dict__.copy()


def rho_spec(
    rho: float,
    *,
    updates: int = UPDATES,
    sequences_per_update: int = SEQUENCES_PER_UPDATE,
) -> RhoSpec:
    """Return an exact replay allocation for an arbitrary training horizon.

    Defaults deliberately preserve the original 610-update Run4 schedule.
    Longer experiments must pass their horizon explicitly so they cannot
    silently reuse the 19,520-row cache geometry.
    """

    if rho not in RHO_VALUES:
        raise ValueError(f"unsupported rho: {rho}")
    if updates <= 0 or sequences_per_update <= 0:
        raise ValueError("updates and sequences_per_update must be positive")
    total_rows = updates * sequences_per_update
    pile_rows = int(round(total_rows * rho))
    if pile_rows != total_rows * rho:
        raise AssertionError("rho must produce an exact integral row count")
    low, remainder = divmod(pile_rows, updates)
    high = low + (1 if remainder else 0)
    return RhoSpec(
        rho=rho,
        label=RHO_LABELS[rho],
        math_rows=total_rows - pile_rows,
        pile_rows=pile_rows,
        low_pile_per_batch=low,
        high_pile_per_batch=high,
        low_batches=updates - remainder,
        high_batches=remainder,
    )


def largest_remainder_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate an exact integer total using Hamilton's largest-remainder rule."""

    if total < 0 or not weights or any(value < 0 for value in weights.values()):
        raise ValueError("invalid total or weights")
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")
    quotas = {name: total * float(value) / weight_sum for name, value in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in quotas.items()}
    missing = total - sum(counts.values())
    order = sorted(weights, key=lambda name: (-(quotas[name] - counts[name]), name))
    for name in order[:missing]:
        counts[name] += 1
    if sum(counts.values()) != total:
        raise AssertionError("largest-remainder allocation failed")
    return counts


def mixture_source_ids(
    seed: int,
    rho: float,
    *,
    updates: int = UPDATES,
    sequences_per_update: int = SEQUENCES_PER_UPDATE,
) -> np.ndarray:
    """Return exact per-row source IDs: 0=OpenWebMath, 1=The Pile."""

    spec = rho_spec(
        rho, updates=updates, sequences_per_update=sequences_per_update
    )
    rng = np.random.default_rng(seed + 91_000_003 + int(round(rho * 10_000)))
    pile_per_batch = np.asarray(
        [spec.high_pile_per_batch] * spec.high_batches
        + [spec.low_pile_per_batch] * spec.low_batches,
        dtype=np.uint8,
    )
    rng.shuffle(pile_per_batch)
    source = np.zeros((updates, sequences_per_update), dtype=np.uint8)
    for update, pile_count in enumerate(pile_per_batch):
        if pile_count:
            positions = rng.choice(sequences_per_update, int(pile_count), replace=False)
            source[update, positions] = 1
    flat = source.reshape(-1)
    if int(flat.sum()) != spec.pile_rows:
        raise AssertionError("internal replay schedule error")
    return flat


def validate_source_schedule(
    source_ids: np.ndarray,
    rho: float,
    *,
    updates: int = UPDATES,
    sequences_per_update: int = SEQUENCES_PER_UPDATE,
) -> dict[str, int]:
    spec = rho_spec(
        rho, updates=updates, sequences_per_update=sequences_per_update
    )
    source = np.asarray(source_ids, dtype=np.uint8).reshape(
        updates, sequences_per_update
    )
    if not np.all(np.isin(source, [0, 1])):
        raise ValueError("source schedule contains values other than 0/1")
    pile_counts = source.sum(axis=1)
    allowed = {spec.low_pile_per_batch, spec.high_pile_per_batch}
    if not np.all(np.isin(pile_counts, list(allowed))):
        raise ValueError("per-batch replay counts are invalid")
    result = {
        "math_rows": int(np.count_nonzero(source == 0)),
        "pile_rows": int(np.count_nonzero(source == 1)),
        f"batches_with_{spec.low_pile_per_batch}_pile": int(
            np.count_nonzero(pile_counts == spec.low_pile_per_batch)
        ),
    }
    if spec.high_pile_per_batch != spec.low_pile_per_batch:
        result[f"batches_with_{spec.high_pile_per_batch}_pile"] = int(
            np.count_nonzero(pile_counts == spec.high_pile_per_batch)
        )
    return result


def interleave_windows(
    math_windows: np.ndarray,
    pile_windows: np.ndarray,
    source_ids: Sequence[int] | np.ndarray,
    *,
    updates: int = UPDATES,
    sequences_per_update: int = SEQUENCES_PER_UPDATE,
) -> np.ndarray:
    source = np.asarray(source_ids, dtype=np.uint8)
    total_rows = updates * sequences_per_update
    if source.shape != (total_rows,):
        raise ValueError(f"expected {total_rows} source IDs, found {source.shape}")
    expected_math = int(np.count_nonzero(source == 0))
    expected_pile = int(np.count_nonzero(source == 1))
    if math_windows.shape != (expected_math, CONTEXT + 1):
        raise ValueError("wrong OpenWebMath window shape")
    if pile_windows.shape != (expected_pile, CONTEXT + 1):
        raise ValueError("wrong Pile window shape")
    output = np.empty((total_rows, CONTEXT + 1), dtype=np.uint16)
    output[source == 0] = math_windows
    output[source == 1] = pile_windows
    return output


def stream_to_windows(stream: np.ndarray, rows: int) -> np.ndarray:
    expected = rows * CONTEXT + 1
    if stream.ndim != 1 or len(stream) != expected:
        raise ValueError(f"expected stream of {expected} tokens, found {stream.shape}")
    output = np.empty((rows, CONTEXT + 1), dtype=np.uint16)
    for index in range(rows):
        start = index * CONTEXT
        output[index] = stream[start : start + CONTEXT + 1]
    return output


def pile_document_identity(row: dict) -> str:
    text = str(row.get("text") or "")
    normalized = re.sub(r"\s+", " ", text).strip()
    # Deliberately omit the component name: the same normalized document in
    # two components or splits must still be treated as overlap.
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def encode_text(tokenizer, text: str) -> list[int]:
    if not text or not text.strip():
        return []
    values = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Pythia tokenizer must define EOS")
    values.append(int(tokenizer.eos_token_id))
    if values and (min(values) < 0 or max(values) >= 2**16):
        raise ValueError("token ID does not fit uint16")
    return values


def collect_component_windows(
    rows: Iterable[dict],
    tokenizer,
    required_rows: Mapping[str, int],
    *,
    forbidden_identities: set[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Collect exact, component-local windows without crossing components."""

    unknown = set(required_rows) - set(PILE_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown Pile components: {sorted(unknown)}")
    buffers = {name: array("H") for name, count in required_rows.items() if count > 0}
    identities = {name: [] for name in buffers}
    targets = {name: count * CONTEXT + 1 for name, count in required_rows.items() if count > 0}
    for row in rows:
        meta = row.get("meta") or {}
        source = meta.get("pile_set_name") if isinstance(meta, dict) else None
        if source not in buffers or len(buffers[source]) >= targets[source]:
            continue
        identity = pile_document_identity(row)
        if forbidden_identities and identity in forbidden_identities:
            continue
        token_ids = encode_text(tokenizer, str(row.get("text") or ""))
        if not token_ids:
            continue
        buffers[source].extend(token_ids)
        identities[source].append(identity)
        if all(len(buffers[name]) >= targets[name] for name in buffers):
            break
    missing = {name: targets[name] - len(values) for name, values in buffers.items() if len(values) < targets[name]}
    if missing:
        raise RuntimeError(f"Pile split exhausted before component quotas were met: {missing}")
    windows: dict[str, np.ndarray] = {}
    for name, count in required_rows.items():
        if count == 0:
            windows[name] = np.empty((0, CONTEXT + 1), dtype=np.uint16)
            continue
        stream = np.frombuffer(buffers[name], dtype=np.uint16, count=count * CONTEXT + 1).copy()
        windows[name] = stream_to_windows(stream, count)
    return windows, identities


def select_component_prefixes(
    pools: Mapping[str, np.ndarray], counts: Mapping[str, int], *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take nested per-component prefixes, then deterministically shuffle them."""

    source_names = list(PILE_WEIGHTS)
    chunks: list[np.ndarray] = []
    source_ids: list[np.ndarray] = []
    for source_id, name in enumerate(source_names):
        count = int(counts.get(name, 0))
        pool = pools[name]
        if count > len(pool):
            raise ValueError(f"component pool {name} has {len(pool)} rows, need {count}")
        if count:
            chunks.append(np.asarray(pool[:count]))
            source_ids.append(np.full(count, source_id, dtype=np.uint8))
    if not chunks:
        return np.empty((0, CONTEXT + 1), dtype=np.uint16), np.empty(0, dtype=np.uint8)
    windows = np.concatenate(chunks)
    ids = np.concatenate(source_ids)
    order = np.random.default_rng(seed).permutation(len(windows))
    return windows[order], ids[order]


def document_set_digest(identities: Mapping[str, Sequence[str]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(identities):
        for identity in sorted(identities[name]):
            digest.update(name.encode("utf-8"))
            digest.update(identity.encode("ascii"))
    return digest.hexdigest()


assert abs(sum(PILE_WEIGHTS.values()) - 100.0) < 1e-9
assert set().union(*map(set, PILE_GROUPS.values())) == set(PILE_WEIGHTS)
