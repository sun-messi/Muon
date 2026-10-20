"""Deterministic nanochat-style small midtraining data preparation/loading."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_COUNTS = {"smoltalk": 16_200, "mmlu": 3_520, "gsm8k": 280}
VAL_COUNTS = {"smoltalk": 1_620, "mmlu": 352, "gsm8k": 28}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_messages(messages: list[dict], eos: str) -> str:
    parts = [eos]
    for message in messages:
        role = str(message["role"]).strip().capitalize()
        content = message["content"]
        if isinstance(content, list):
            rendered = []
            for item in content:
                if isinstance(item, dict):
                    kind = item.get("type", "text")
                    text = item.get("text", "")
                    rendered.append(f"<{kind}>{text}</{kind}>")
                else:
                    rendered.append(str(item))
            content = "".join(rendered)
        parts.append(f"{role}: {content}\n")
    parts.append(eos)
    return "".join(parts)


def mmlu_messages(row: dict) -> list[dict]:
    if "train" in row and isinstance(row["train"], dict):
        row = row["train"]
    letters = "ABCD"
    choices = row["choices"]
    prompt = row["question"] + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    ) + "\nAnswer with A, B, C, or D."
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": letters[int(row["answer"])]},
    ]


def gsm8k_messages(row: dict) -> list[dict]:
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


def select_tokenized(
    rows: Iterable[dict],
    *,
    source: str,
    count: int,
    tokenizer,
    max_stream_tokens: int,
) -> tuple[list[list[int]], dict]:
    selected: list[list[int]] = []
    rejected_long = 0
    for row in rows:
        if source == "smoltalk":
            messages = row["messages"]
        elif source == "mmlu":
            messages = mmlu_messages(row)
        elif source == "gsm8k":
            messages = gsm8k_messages(row)
        else:
            raise ValueError(source)
        text = render_messages(messages, tokenizer.eos_token)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(token_ids) > max_stream_tokens:
            rejected_long += 1
            continue
        if len(token_ids) < 2:
            continue
        selected.append(token_ids)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"{source}: needed {count} examples but found {len(selected)} fitting the context"
        )
    return selected, {"selected": count, "rejected_long": rejected_long}


def best_fit_pack(
    conversations: list[list[int]],
    *,
    row_capacity: int,
    pad_id: int,
    buffer_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack complete conversations without cropping using nanochat-style best fit."""

    pending = list(conversations)
    cursor = 0
    buffer: list[list[int]] = []
    rows: list[list[int]] = []
    lengths: list[int] = []
    while cursor < len(pending) or buffer:
        while cursor < len(pending) and len(buffer) < buffer_size:
            buffer.append(pending[cursor])
            cursor += 1
        row: list[int] = []
        while buffer:
            remaining = row_capacity - len(row)
            candidates = [
                (len(item), index) for index, item in enumerate(buffer) if len(item) <= remaining
            ]
            if not candidates:
                break
            _, best_index = max(candidates)
            row.extend(buffer.pop(best_index))
            while cursor < len(pending) and len(buffer) < buffer_size:
                buffer.append(pending[cursor])
                cursor += 1
        if not row:
            raise RuntimeError("a conversation did not fit despite length filtering")
        lengths.append(len(row))
        rows.append(row + [pad_id] * (row_capacity - len(row)))
    return np.asarray(rows, dtype=np.uint16), np.asarray(lengths, dtype=np.uint16)


def contiguous_text_stream(tokenizer, texts: Iterable[str], target_tokens: int) -> np.ndarray:
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for text in texts:
        if not text or not text.strip():
            continue
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids.append(eos)
        if len(ids) >= target_tokens + 1:
            break
    if len(ids) < target_tokens + 1:
        raise RuntimeError(f"retention corpus has only {len(ids)} tokens")
    return np.asarray(ids[: target_tokens + 1], dtype=np.uint16)


class PackedDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.streams = np.load(root / f"{split}_streams.npy", mmap_mode="r")
        self.lengths = np.load(root / f"{split}_lengths.npy", mmap_mode="r")
        if len(self.streams) != len(self.lengths):
            raise ValueError(f"{split}: streams/lengths mismatch")

    def __len__(self) -> int:
        return len(self.streams)

    def __getitem__(self, index: int):
        stream = np.asarray(self.streams[index], dtype=np.int64)
        content_length = int(self.lengths[index])
        x = torch.from_numpy(stream[:-1].copy())
        y = torch.from_numpy(stream[1:].copy())
        if content_length < len(stream):
            y[content_length - 1 :] = -100
        return x, y, index


class ContiguousDataset(Dataset):
    def __init__(self, path: Path, context: int):
        self.tokens = np.load(path, mmap_mode="r")
        self.context = context

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.context

    def __getitem__(self, index: int):
        start = index * self.context
        stream = np.asarray(self.tokens[start : start + self.context + 1], dtype=np.int64)
        return torch.from_numpy(stream[:-1].copy()), torch.from_numpy(stream[1:].copy()), index


def dataset_fingerprint(dataset: Dataset, count: int = 8) -> str:
    digest = hashlib.sha256()
    for index in range(min(count, len(dataset))):
        x, y, sample_index = dataset[index]
        digest.update(int(sample_index).to_bytes(8, "little"))
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
