import json

import pytest
import torch

from adaptive_pion import (
    AdaptivePionController,
    DEFAULT_CANDIDATES,
    candidate_metrics,
    load_controller_config,
    mode_snr,
    scalar_polynomial_response,
    soft_reliability,
    write_controller_config,
)
from spectral import MatchedMuon, parse_schedule, spectral_transform


NAMES = [
    "gpt_neox.layers.0.attention.query_key_value.weight",
    "gpt_neox.layers.0.attention.dense.weight",
    "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
    "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
]


def test_scalar_candidate_response_matches_matrix_polynomial_svd():
    torch.manual_seed(123)
    matrix = torch.randn(9, 15)
    singular = torch.linalg.svdvals(matrix)
    for candidate in DEFAULT_CANDIDATES:
        expected = torch.linalg.svdvals(
            spectral_transform(
                matrix,
                candidate.backend,
                candidate.promotion_steps,
                candidate.suppression_steps,
                candidate.alpha,
            )
        )
        actual = scalar_polynomial_response(singular, candidate)
        torch.testing.assert_close(
            torch.sort(actual, descending=True).values,
            expected,
            rtol=2e-4,
            atol=2e-5,
        )


def test_mode_snr_and_soft_target_are_hand_checkable():
    projections = torch.tensor([[1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [1.0, -1.0]])
    snr = mode_snr(projections)
    assert snr[0] > 1e10
    assert snr[1] == 0
    target = soft_reliability(snr, 1.0)
    torch.testing.assert_close(target, torch.tensor([1.0, 0.0]))


def test_candidate_metrics_are_finite_and_muon_norm_is_one():
    singular = torch.tensor([4.0, 2.0, 0.5, 0.1])
    target = torch.tensor([1.0, 1.0, 0.0, 0.0])
    rows = candidate_metrics(singular, target)
    assert len(rows) == 5
    assert rows[0]["candidate"] == "muon5ns"
    assert rows[0]["norm_ratio_muon"] == pytest.approx(1.0)
    assert all(torch.isfinite(torch.tensor(row["mismatch"])) for row in rows)


def test_controller_config_round_trip_and_validation(tmp_path):
    path = tmp_path / "controller.json"
    original = write_controller_config(
        path, signal="mode_snr", granularity="operator", tau=.75
    )
    loaded, digest = load_controller_config(path)
    assert loaded == original
    assert len(digest) == 64
    broken = dict(original)
    broken["candidates"] = broken["candidates"][1:]
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="level 0"):
        load_controller_config(path)


def test_controller_decision_dynamic_override_and_resume():
    torch.manual_seed(9)
    parameters = [torch.nn.Parameter(torch.randn(6, 10)) for _ in NAMES]
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=.95,
        schedule=parse_schedule("0:muon5ns:0:0:0"),
        parameter_names=NAMES,
    )
    for parameter in parameters:
        parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    config = {
        "format": "adaptive_pion_controller_v1",
        "signal": "mode_snr",
        "granularity": "operator",
        "warmup_updates": 2,
        "decision_interval": 2,
        "observation_updates": 2,
        "tau": .75,
        "norm_guard": [.20, 1.15],
        "hysteresis_relative_improvement": .10,
        "max_level_change": 1,
        "candidates": [candidate.__dict__ for candidate in DEFAULT_CANDIDATES],
    }
    controller = AdaptivePionController(config, NAMES)
    microbatches = []
    for group in range(8):
        values = {}
        for index, (name, parameter) in enumerate(zip(NAMES, parameters)):
            # A stable leading direction plus changing tail entries.
            gradient = torch.randn_like(parameter) * .1
            gradient[0, 0] = 2.0 + index
            gradient[-1, -1] = (-1.0 if group % 2 else 1.0) * (index + 1)
            values[name] = gradient
        microbatches.append(values)
        controller.observe_microbatch(values)
    summed = {
        name: sum(group[name] for group in microbatches) for name in NAMES
    }
    reconstructed = {
        name: torch.stack([group[name] for group in microbatches]).sum(0) for name in NAMES
    }
    for name in NAMES:
        torch.testing.assert_close(summed[name], reconstructed[name])
    event = controller.decide(2, optimizer)
    assert event["step"] == 2
    assert len(event["decisions"]) == 4
    assert set(optimizer.dynamic_segments) == set(NAMES)
    state = controller.state_dict()

    cloned_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    cloned_optimizer = MatchedMuon(
        cloned_parameters,
        lr=.01,
        momentum=.95,
        schedule=parse_schedule("0:muon5ns:0:0:0"),
        parameter_names=NAMES,
    )
    cloned_optimizer.load_state_dict(optimizer.state_dict())
    resumed = AdaptivePionController(config, NAMES)
    resumed.load_state_dict(state, cloned_optimizer)
    assert resumed.current_levels == controller.current_levels
    assert cloned_optimizer.dynamic_segments == optimizer.dynamic_segments


def test_matched_muon_without_dynamic_overrides_remains_identical():
    torch.manual_seed(77)
    left = torch.nn.Parameter(torch.randn(7, 11))
    right = torch.nn.Parameter(left.detach().clone())
    schedule = parse_schedule("0:muon5ns:0:0:0")
    a = MatchedMuon([left], lr=.01, schedule=schedule, parameter_names=[NAMES[0]])
    b = MatchedMuon([right], lr=.01, schedule=schedule, parameter_names=[NAMES[0]])
    gradient = torch.randn_like(left)
    left.grad = gradient.clone()
    right.grad = gradient.clone()
    a.step()
    b.step()
    assert torch.equal(left, right)
