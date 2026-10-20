"""Deterministic all-token SmolTalk streams for Pythia midtraining."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


SMOLTALK_ID = "HuggingFaceTB/smol-smoltalk"
PYTHIA_ID = "EleutherAI/pythia-31m"
CONTEXT = 1024
ROWS_PER_UPDATE = 32
TOTAL_UPDATES = 1000
TRAIN_ROWS = ROWS_PER_UPDATE * TOTAL_UPDATES
TRAIN_TARGET_TOKENS = TRAIN_ROWS * CONTEXT
TRAIN_STREAM_TOKENS = TRAIN_TARGET_TOKENS + 1
MONITOR_ROWS = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_messages(messages: list[dict]) -> str:
    """Render every message; no role is masked from causal LM training."""

    parts: list[str] = []
    for message in messages:
        role = str(message["role"]).strip().capitalize()
        content = message["content"]
        if isinstance(content, list):
            rendered: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    kind = str(item.get("type", "text"))
                    text = str(item.get("text", ""))
                    rendered.append(f"<{kind}>{text}</{kind}>")
                else:
                    rendered.append(str(item))
            content = "".join(rendered)
        parts.append(f"{role}: {content}\n")
    return "".join(parts)


def build_fixed_stream(
    rows,
    tokenizer,
    *,
    seed: int,
    target_tokens: int = TRAIN_STREAM_TOKENS,
    encode_batch_size: int = 256,
) -> tuple[np.ndarray, dict]:
    """Shuffle conversations and fill an exact-size EOS-delimited token stream."""

    if target_tokens < 2:
        raise ValueError("target_tokens must be at least two")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    order = np.random.default_rng(seed).permutation(len(rows))
    stream = np.empty(target_tokens, dtype=np.uint16)
    stream[0] = eos
    cursor = 1
    conversations_started = 0
    conversations_completed = 0
    for batch_start in range(0, len(order), encode_batch_size):
        batch_indices = order[batch_start : batch_start + encode_batch_size]
        texts = [render_messages(rows[int(index)]["messages"]) for index in batch_indices]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for token_ids in encoded:
            if cursor >= target_tokens:
                break
            conversations_started += 1
            available = target_tokens - cursor
            take = min(len(token_ids), available)
            if take:
                stream[cursor : cursor + take] = np.asarray(token_ids[:take], dtype=np.uint16)
                cursor += take
            if take == len(token_ids) and cursor < target_tokens:
                stream[cursor] = eos
                cursor += 1
                conversations_completed += 1
            elif take == len(token_ids):
                conversations_completed += 1
            if cursor >= target_tokens:
                break
        if cursor >= target_tokens:
            break
    if cursor != target_tokens:
        raise RuntimeError(f"SmolTalk has only {cursor} tokens; needed {target_tokens}")
    return stream, {
        "seed": seed,
        "stream_tokens": target_tokens,
        "target_tokens": target_tokens - 1,
        "rows": (target_tokens - 1) // CONTEXT,
        "conversations_started": conversations_started,
        "conversations_completed": conversations_completed,
    }


def build_complete_stream(rows, tokenizer, *, encode_batch_size: int = 256) -> tuple[np.ndarray, dict]:
    """Encode an entire split in source order with one EOS between conversations."""

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    chunks: list[np.ndarray] = [np.asarray([eos], dtype=np.uint16)]
    for batch_start in range(0, len(rows), encode_batch_size):
        batch = [rows[index] for index in range(batch_start, min(len(rows), batch_start + encode_batch_size))]
        texts = [render_messages(row["messages"]) for row in batch]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for token_ids in encoded:
            values = np.empty(len(token_ids) + 1, dtype=np.uint16)
            values[:-1] = np.asarray(token_ids, dtype=np.uint16)
            values[-1] = eos
            chunks.append(values)
    stream = np.concatenate(chunks)
    return stream, {
        "conversations": len(rows),
        "stream_tokens": int(len(stream)),
        "target_tokens": int(len(stream) - 1),
        "rows": int((len(stream) - 1) // CONTEXT),
        "unused_tail_tokens": int((len(stream) - 1) % CONTEXT),
    }


class TokenWindowDataset(Dataset):
    """Read either a continuous 1-D stream or explicit ``[N, T+1]`` windows."""

    def __init__(self, path: Path, *, context: int = CONTEXT, max_rows: int = 0):
        self.path = Path(path)
        self.context = int(context)
        self.tokens = np.load(self.path, mmap_mode="r")
        if self.tokens.ndim == 1:
            available = (len(self.tokens) - 1) // self.context
        elif self.tokens.ndim == 2 and self.tokens.shape[1] == self.context + 1:
            available = self.tokens.shape[0]
        else:
            raise ValueError(
                f"expected a 1-D stream or [N,{self.context + 1}] windows, "
                f"found shape {self.tokens.shape}: {self.path}"
            )
        self.rows = min(available, max_rows) if max_rows else available
        if self.rows <= 0:
            raise ValueError(f"token stream is too short: {self.path}")

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int):
        if index < 0 or index >= self.rows:
            raise IndexError(index)
        if self.tokens.ndim == 1:
            start = index * self.context
            source = self.tokens[start : start + self.context + 1]
        else:
            source = self.tokens[index]
        values = np.asarray(source, dtype=np.int64)
        return torch.from_numpy(values[:-1].copy()), torch.from_numpy(values[1:].copy()), index


def dataset_fingerprint(dataset: TokenWindowDataset) -> str:
    digest = hashlib.sha256()
    candidates = sorted({0, len(dataset) // 2, len(dataset) - 1})
    for index in candidates:
        x, y, _ = dataset[index]
        digest.update(str(index).encode())
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
