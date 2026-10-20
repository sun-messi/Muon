import random

import numpy as np
import torch

from smoltalk_data import TokenWindowDataset, build_complete_stream, build_fixed_stream, render_messages
from spectral import MatchedMuon, coefficient_sequence, parse_schedule, segment_at
from train_smoltalk import load_state, save_state
from train_mid import config_hash


class FakeRows:
    def __init__(self, count=20):
        self.values = [
            {
                "messages": [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ]
            }
            for index in range(count)
        ]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.values[index]
        return self.values[index]


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        encoded = [[1 + (ord(character) % 200) for character in text] for text in texts]
        return {"input_ids": encoded[0] if single else encoded}


def test_render_includes_all_roles():
    text = render_messages(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    assert text == "System: rules\nUser: question\nAssistant: answer\n"


def test_fixed_stream_exact_and_reproducible():
    rows = FakeRows(40)
    tokenizer = FakeTokenizer()
    left, left_stats = build_fixed_stream(rows, tokenizer, seed=42, target_tokens=1025)
    right, right_stats = build_fixed_stream(rows, tokenizer, seed=42, target_tokens=1025)
    other, _ = build_fixed_stream(rows, tokenizer, seed=43, target_tokens=1025)
    assert len(left) == 1025
    assert left[0] == tokenizer.eos_token_id
    assert np.array_equal(left, right)
    assert left_stats == right_stats
    assert not np.array_equal(left, other)
    assert left_stats["target_tokens"] == 1024


def test_complete_stream_and_windows_have_no_mask(tmp_path):
    values, stats = build_complete_stream(FakeRows(8), FakeTokenizer())
    assert stats["conversations"] == 8
    np.save(tmp_path / "tokens.npy", values)
    context = 16
    dataset = TokenWindowDataset(tmp_path / "tokens.npy", context=context)
    x, y, _ = dataset[0]
    assert len(x) == context and len(y) == context
    assert torch.equal(x[1:], y[:-1])
    assert bool((y >= 0).all())


def test_pion_setting_is_six_p_one_s_alpha_one():
    sequence = coefficient_sequence("pion", 6, 1, 1.0)
    assert len(sequence) == 7
    assert sequence[:6] == ((1.875, -1.25, .375),) * 6
    assert sequence[-1] == (1.0, .5, -.5)


def test_spectral_fork_restores_full_state(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([3], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    source_schedule = parse_schedule("0:muon5ns:0:0:0")
    model = torch.nn.Linear(6, 4)
    matrix = MatchedMuon([model.weight], lr=.01, schedule=source_schedule)
    auxiliary = torch.optim.AdamW([model.bias], lr=.001)
    model.weight.grad = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 100
    model.bias.grad = torch.arange(4, dtype=torch.float32) / 100
    matrix.step()
    auxiliary.step()
    invariants = {"format": "test"}
    invariants["invariant_hash"] = config_hash(invariants)
    config = {"invariants": invariants, "spectral_schedule": "0:muon5ns:0:0:0"}
    path = tmp_path / "prefix.pt"
    save_state(path, model, matrix, auxiliary, 1, config)

    resumed = torch.nn.Linear(6, 4)
    pion_schedule = parse_schedule("0:muon5ns:0:0:0,1:pion:6:1:1")
    resumed_matrix = MatchedMuon([resumed.weight], lr=.01, schedule=pion_schedule)
    resumed_aux = torch.optim.AdamW([resumed.bias], lr=.001)
    next_update, source = load_state(
        path,
        resumed,
        resumed_matrix,
        resumed_aux,
        invariants=invariants,
        requested_schedule=pion_schedule,
        allow_spectral_fork=True,
        expected_fork_step=1,
    )
    assert next_update == 1
    assert source["spectral_schedule"] == "0:muon5ns:0:0:0"
    assert segment_at(resumed_matrix.schedule, 1).backend == "pion"
    for left, right in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(left, right)
    assert resumed_matrix.update_step == 1
