"""Deterministic full-sequence ORD replacement for the 15-route CPT experiment."""
from pathlib import Path
import numpy as np
import torch


class ORDFullSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: Path, prefix: str, *, k: int, seed: int = 42,
                 validation: bool = False, continuation_only: bool = False,
                 max_rows: int = 0):
        if k not in (0, 32, 64, 128, 256):
            raise ValueError(k)
        self.tokens = np.load(cache_dir / f"{prefix}_tokens.npy", mmap_mode="r")
        if self.tokens.ndim != 2 or self.tokens.shape[1] != 1024:
            raise ValueError("ORD cache must have 1024-token rows")
        source_path = cache_dir / f"{prefix}_source_ids.npy"
        self.source = np.load(source_path, mmap_mode="r") if source_path.exists() else None
        ord_prefix = "ord_validation" if validation else "ord_train"
        self.ord_tokens = np.load(cache_dir / f"{ord_prefix}_tokens.npy", mmap_mode="r")
        self.ord_lengths = np.load(cache_dir / f"{ord_prefix}_lengths.npy", mmap_mode="r")
        self.k = k
        self.seed = seed
        self.continuation_only = continuation_only
        self.rows = min(len(self.tokens), max_rows) if max_rows else len(self.tokens)

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        if index < 0 or index >= self.rows:
            raise IndexError(index)
        seq = np.asarray(self.tokens[index], dtype=np.int64).copy()
        is_new = self.source is None or int(self.source[index]) == 0
        if self.k and is_new:
            rng = np.random.default_rng(self.seed * 1_000_003 + index)
            starts = []
            while len(starts) < 16:
                start = int(rng.integers(0, 1009))
                if all(abs(start - previous) >= 16 for previous in starts):
                    starts.append(start)
            for start in starts[:self.k // 16]:
                record_index = int(rng.integers(len(self.ord_tokens)))
                length = int(self.ord_lengths[record_index])
                offset = int(rng.integers(length - 15))
                seq[start:start + 16] = self.ord_tokens[record_index, offset:offset + 16]
        x = torch.from_numpy(seq[:-1].copy())
        y = torch.from_numpy(seq[1:].copy())
        if self.continuation_only:
            y[:511] = -100
        return x, y, index
