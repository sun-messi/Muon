import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from middata import PackedDataset, best_fit_pack, dataset_fingerprint
from spectral import (
    MUON,
    MatchedMuon,
    apply_polynomial,
    coefficient_sequence,
    parse_schedule,
    segment_at,
    spectral_transform,
)
from train_mid import DIAGNOSTIC_PARAM, load_checkpoint, parameter_partition, save_checkpoint


def reference_muon(matrix):
    result = matrix / (matrix.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.mT
    for _ in range(5):
        gram = result @ result.mT
        result = 3.4445 * result - 4.7750 * (gram @ result) + 2.0315 * (gram @ gram @ result)
    return result.mT if transposed else result


@pytest.mark.parametrize("shape", [(7, 13), (13, 7), (3, 7, 13)])
def test_muon_matches_reference(shape):
    torch.manual_seed(4)
    matrix = torch.randn(shape)
    actual = apply_polynomial(matrix, coefficient_sequence("muon5ns"))
    torch.testing.assert_close(actual, reference_muon(matrix), rtol=1e-5, atol=1e-6)


def test_zero_and_pion_sequence():
    zero = torch.zeros(4, 8)
    assert torch.equal(apply_polynomial(zero, (MUON,) * 5), zero)
    sequence = coefficient_sequence("pion", 10, 1, 0.0)
    assert len(sequence) == 11
    assert sequence[-1] == (0.0, 2.5, -1.5)


def test_ordered_pion_coefficients_preserve_declared_stage_counts():
    canonical = coefficient_sequence("pion", 2, 4, 1.0)
    assert coefficient_sequence("pion_order_ppssss", 2, 4, 1.0) == canonical
    interleaved = coefficient_sequence("pion_order_spspss", 2, 4, 1.0)
    assert interleaved != canonical
    assert len(interleaved) == 6


def test_ordered_pion_rejects_mismatched_counts():
    with pytest.raises(ValueError, match="disagrees"):
        coefficient_sequence("pion_order_psp", 3, 0, 1.0)


@pytest.mark.parametrize("shape", [(7, 13), (13, 7), (3, 7, 13)])
def test_norm_controlled_psspp_blend_preserves_psspp_frobenius_norm(shape):
    torch.manual_seed(404)
    matrix = torch.randn(shape)
    psspp = spectral_transform(matrix, "pion_order_psspp", 3, 2, .45)
    blend0 = spectral_transform(matrix, "pion_blend_psspp_norm_b000", 3, 2, .45)
    blend40 = spectral_transform(matrix, "pion_blend_psspp_norm_b040", 3, 2, .45)
    blend100 = spectral_transform(matrix, "pion_blend_psspp_norm_b100", 3, 2, .45)
    torch.testing.assert_close(blend0, psspp)
    target = psspp.norm(dim=(-2, -1))
    torch.testing.assert_close(blend40.norm(dim=(-2, -1)), target)
    torch.testing.assert_close(blend100.norm(dim=(-2, -1)), target)
    muon = spectral_transform(matrix, "muon5ns")
    cosine = torch.nn.functional.cosine_similarity(
        blend100.flatten(-2), muon.flatten(-2), dim=-1
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-5, atol=1e-6)


def test_norm_controlled_psspp_blend_validates_encoding_and_counts():
    assert len(coefficient_sequence("pion_blend_psspp_norm_b025", 3, 2, .45)) == 5
    with pytest.raises(ValueError, match="requires P=3, S=2"):
        coefficient_sequence("pion_blend_psspp_norm_b025", 2, 2, .45)
    with pytest.raises(ValueError, match="invalid norm-controlled"):
        coefficient_sequence("pion_blend_psspp_norm_bXYZ", 3, 2, .45)
    with pytest.raises(ValueError, match="must be in"):
        coefficient_sequence("pion_blend_psspp_norm_b101", 3, 2, .45)


@pytest.mark.parametrize("shape", [(7, 13), (13, 7), (3, 7, 13)])
def test_norm_controlled_signal_residual_preserves_norm_and_raw_endpoint(shape):
    torch.manual_seed(405)
    matrix = torch.randn(shape)
    psspp = spectral_transform(matrix, "pion_order_psspp", 3, 2, .45)
    signal0 = spectral_transform(matrix, "pion_signal_psspp_norm_b000", 3, 2, .45)
    signal40 = spectral_transform(matrix, "pion_signal_psspp_norm_b040", 3, 2, .45)
    signal100 = spectral_transform(matrix, "pion_signal_psspp_norm_b100", 3, 2, .45)
    torch.testing.assert_close(signal0, psspp)
    target = psspp.norm(dim=(-2, -1))
    torch.testing.assert_close(signal40.norm(dim=(-2, -1)), target)
    torch.testing.assert_close(signal100.norm(dim=(-2, -1)), target)
    cosine = torch.nn.functional.cosine_similarity(
        signal100.flatten(-2), matrix.float().flatten(-2), dim=-1
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine), rtol=1e-5, atol=1e-6)


def test_norm_controlled_signal_residual_validates_encoding_and_counts():
    assert len(coefficient_sequence("pion_signal_psspp_norm_b025", 3, 2, .45)) == 5
    with pytest.raises(ValueError, match="requires P=3, S=2"):
        coefficient_sequence("pion_signal_psspp_norm_b025", 2, 2, .45)
    with pytest.raises(ValueError, match="invalid norm-controlled"):
        coefficient_sequence("pion_signal_psspp_norm_bXYZ", 3, 2, .45)
    with pytest.raises(ValueError, match="must be in"):
        coefficient_sequence("pion_signal_psspp_norm_b101", 3, 2, .45)


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("alt20", [20, 0, 20, 0, 20, 0]),
        ("block20p10", [20, 20, 20, 20, 20, 0, 0, 0, 0, 0]),
        ("pulse40p4", [40, 0, 0, 0, 40, 0, 0, 0]),
        ("pulse50p5", [50, 0, 0, 0, 0, 50, 0, 0, 0, 0]),
        ("pulse100p10", [100, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("burst100p100", [100] * 10 + [0] * 10),
    ],
)
def test_psspp_pulse_policy_selects_expected_residual(pattern, expected):
    parameter = torch.nn.Parameter(torch.randn(4, 8))
    schedule = parse_schedule(f"0:pion_pulse_psspp_{pattern}:3:2:.45")
    optimizer = MatchedMuon([parameter], schedule=schedule)
    selected = []
    for step in range(len(expected)):
        optimizer.update_step = step
        active = optimizer._segment_for_parameter(schedule[0], parameter)
        selected.append(int(active.backend.rsplit("b", 1)[1]))
    assert selected == expected


def test_psspp_pulse_policy_rejects_unknown_pattern_and_wrong_counts():
    with pytest.raises(ValueError, match="unknown PSSPP pulse"):
        coefficient_sequence("pion_pulse_psspp_unknown", 3, 2, .45)
    with pytest.raises(ValueError, match="requires P=3, S=2"):
        coefficient_sequence("pion_pulse_psspp_alt20", 2, 2, .45)


def test_schedule_boundaries():
    schedule = parse_schedule("0:muon5ns:0:0:0,3:pion:10:1:0")
    assert segment_at(schedule, 2).backend == "muon5ns"
    assert segment_at(schedule, 3).backend == "pion"
    with pytest.raises(ValueError):
        parse_schedule("1:muon5ns:0:0:0")


def test_best_fit_pack_and_dataset(tmp_path):
    conversations = [[0, 1, 2], [0, 3], [0, 4, 5, 6], [0, 7]]
    streams, lengths = best_fit_pack(conversations, row_capacity=8, pad_id=0, buffer_size=4)
    assert int(lengths.sum()) == sum(map(len, conversations))
    assert streams.shape[1] == 8
    np.save(tmp_path / "train_streams.npy", streams)
    np.save(tmp_path / "train_lengths.npy", lengths)
    dataset = PackedDataset(tmp_path, "train")
    x, y, index = dataset[0]
    assert index == 0 and len(x) == 7 and len(y) == 7
    valid = int((y != -100).sum())
    assert torch.equal(x[1 : valid + 1], y[:valid])
    assert dataset_fingerprint(dataset) == dataset_fingerprint(dataset)


def tiny_model():
    config = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        use_parallel_residual=True,
    )
    return GPTNeoXForCausalLM(config)


def test_parameter_partition_exact():
    model = tiny_model()
    matrix, auxiliary, matrix_names, auxiliary_names, diagnostic = parameter_partition(model)
    assert DIAGNOSTIC_PARAM in matrix_names
    assert diagnostic.ndim == 2
    assert len({id(p) for p in matrix + auxiliary}) == len(list(model.parameters()))
    assert "gpt_neox.embed_in.weight" in auxiliary_names
    assert "embed_out.weight" in auxiliary_names


def test_matched_muon_state_and_identical_prefix():
    torch.manual_seed(1)
    left = torch.nn.Parameter(torch.randn(8, 16))
    right = torch.nn.Parameter(left.detach().clone())
    muon = parse_schedule("0:muon5ns:0:0:0")
    warm = parse_schedule("0:muon5ns:0:0:0,2:pion:10:1:0")
    opt_left = MatchedMuon([left], lr=0.01, momentum=0.95, schedule=muon)
    opt_right = MatchedMuon([right], lr=0.01, momentum=0.95, schedule=warm)
    for step in range(2):
        gradient = torch.arange(left.numel(), dtype=torch.float32).reshape_as(left) / 100
        left.grad = gradient.clone()
        right.grad = gradient.clone()
        opt_left.step()
        opt_right.step()
        assert torch.equal(left, right)
    state = opt_right.state_dict()
    clone = MatchedMuon([torch.nn.Parameter(right.detach().clone())], schedule=warm)
    clone.load_state_dict(state)
    assert clone.update_step == 2
    assert next(iter(clone.state.values()))["momentum_buffer"].shape == right.shape


def test_qkv_mlp_expand_policy_only_transforms_selected_operators():
    torch.manual_seed(101)
    names = [
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.dense.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    ]
    parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in names]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=0.0,
        schedule=parse_schedule("0:pion_policy_qkv_mlp_expand:3:2:.45"),
        parameter_names=names,
    )
    optimizer.step()
    for index, (parameter, initial, gradient) in enumerate(zip(parameters, before, gradients)):
        selected = index in (0, 2)
        transformed = spectral_transform(
            gradient,
            "pion" if selected else "muon5ns",
            3 if selected else 0,
            2 if selected else 0,
            .45 if selected else 0.0,
        )
        torch.testing.assert_close(parameter, initial - .01 * transformed)


def test_parameter_policy_changes_only_selected_matrix_transform():
    torch.manual_seed(17)
    mlp = torch.nn.Parameter(torch.randn(8, 16))
    attention = torch.nn.Parameter(torch.randn(8, 16))
    mlp_before = mlp.detach().clone()
    attention_before = attention.detach().clone()
    mlp_grad = torch.randn_like(mlp)
    attention_grad = torch.randn_like(attention)
    mlp.grad = mlp_grad.clone()
    attention.grad = attention_grad.clone()
    schedule = parse_schedule("0:pion_policy_mlp:2:4:1")
    optimizer = MatchedMuon(
        [mlp, attention],
        lr=.01,
        momentum=.95,
        schedule=schedule,
        parameter_names=[
            "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
            "gpt_neox.layers.0.attention.dense.weight",
        ],
    )
    optimizer.step()
    scale = (8 / 16) ** .5 if 8 > 16 else 1.0
    expected_mlp = mlp_before - .01 * scale * spectral_transform(mlp_grad, "pion", 2, 4, 1)
    expected_attention = attention_before - .01 * scale * spectral_transform(
        attention_grad, "muon5ns"
    )
    torch.testing.assert_close(mlp, expected_mlp)
    torch.testing.assert_close(attention, expected_attention)


def test_parameter_policy_can_combine_with_ordered_polynomial():
    torch.manual_seed(18)
    qkv = torch.nn.Parameter(torch.randn(8, 16))
    output = torch.nn.Parameter(torch.randn(8, 16))
    qkv_before, output_before = qkv.detach().clone(), output.detach().clone()
    qkv.grad, output.grad = torch.randn_like(qkv), torch.randn_like(output)
    qkv_grad, output_grad = qkv.grad.clone(), output.grad.clone()
    schedule = parse_schedule("0:pion_policy_qkv_order_pspp:3:1:.5")
    optimizer = MatchedMuon(
        [qkv, output],
        lr=.01,
        momentum=.95,
        schedule=schedule,
        parameter_names=[
            "gpt_neox.layers.0.attention.query_key_value.weight",
            "gpt_neox.layers.0.attention.dense.weight",
        ],
    )
    optimizer.step()
    expected_qkv = qkv_before - .01 * spectral_transform(
        qkv_grad, "pion_order_pspp", 3, 1, .5
    )
    expected_output = output_before - .01 * spectral_transform(output_grad, "muon5ns")
    torch.testing.assert_close(qkv, expected_qkv)
    torch.testing.assert_close(output, expected_output)


@pytest.mark.parametrize("excluded_layer", range(6))
def test_qkv_layer_exclusion_policy_skips_exactly_one_layer(excluded_layer):
    torch.manual_seed(100 + excluded_layer)
    parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in range(6)]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    schedule = parse_schedule(
        f"0:pion_policy_qkv_exclude_layer{excluded_layer}_order_ppsp:3:1:.5"
    )
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=.95,
        schedule=schedule,
        parameter_names=[
            f"gpt_neox.layers.{layer}.attention.query_key_value.weight"
            for layer in range(6)
        ],
    )
    optimizer.step()
    for layer, (parameter, initial, gradient) in enumerate(zip(parameters, before, gradients)):
        backend = "muon5ns" if layer == excluded_layer else "pion_order_ppsp"
        if backend == "muon5ns":
            transformed = spectral_transform(gradient, backend)
        else:
            transformed = spectral_transform(gradient, backend, 3, 1, .5)
        expected = initial - .01 * transformed
        torch.testing.assert_close(parameter, expected)


def test_all_matrix_policy_can_restore_one_qkv_layer_to_muon():
    torch.manual_seed(89)
    parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in range(3)]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=0.0,
        schedule=parse_schedule(
            "0:pion_policy_all_except_qkv_layer1_order_psspp:3:2:.45"
        ),
        parameter_names=[
            "gpt_neox.layers.0.attention.query_key_value.weight",
            "gpt_neox.layers.1.attention.query_key_value.weight",
            "gpt_neox.layers.1.mlp.dense_h_to_4h.weight",
        ],
    )
    optimizer.step()
    expected_backends = ("pion_order_psspp", "muon5ns", "pion_order_psspp")
    for parameter, initial, gradient, backend in zip(
        parameters, before, gradients, expected_backends
    ):
        transformed = (
            spectral_transform(gradient, backend)
            if backend == "muon5ns"
            else spectral_transform(gradient, backend, 3, 2, .45)
        )
        torch.testing.assert_close(parameter, initial - .01 * transformed)


def test_all_matrix_profile_can_soften_qkv_layer4_order_only():
    torch.manual_seed(91)
    qkv4 = torch.nn.Parameter(torch.randn(8, 16))
    qkv5 = torch.nn.Parameter(torch.randn(8, 16))
    mlp4 = torch.nn.Parameter(torch.randn(8, 16))
    parameters = [qkv4, qkv5, mlp4]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=0.0,
        schedule=parse_schedule(
            "0:pion_profile_all_qkv_l4_pppss_order_psspp:3:2:.45"
        ),
        parameter_names=[
            "gpt_neox.layers.4.attention.query_key_value.weight",
            "gpt_neox.layers.5.attention.query_key_value.weight",
            "gpt_neox.layers.4.mlp.dense_h_to_4h.weight",
        ],
    )
    optimizer.step()
    orders = ("pppss", "psspp", "psspp")
    for parameter, initial, gradient, order in zip(
        parameters, before, gradients, orders
    ):
        transformed = spectral_transform(
            gradient, f"pion_order_{order}", 3, 2, .45
        )
        torch.testing.assert_close(parameter, initial - .01 * transformed)


@pytest.mark.parametrize(
    ("profile", "selected_index", "selected_fraction"),
    [
        ("residual_qkv_b000", 0, "000"),
        ("residual_qkv_b020", 0, "020"),
        ("residual_attention_output_b000", 1, "000"),
        ("residual_attention_output_b020", 1, "020"),
        ("residual_mlp_expand_b000", 2, "000"),
        ("residual_mlp_contract_b000", 3, "000"),
    ],
)
def test_residual_profile_changes_only_selected_operator(
    profile, selected_index, selected_fraction
):
    torch.manual_seed(505)
    names = [
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.dense.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    ]
    parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in names]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=0.0,
        schedule=parse_schedule(
            f"0:pion_profile_{profile}_order_psspp:3:2:.45"
        ),
        parameter_names=names,
    )
    optimizer.step()
    for index, (parameter, initial, gradient) in enumerate(
        zip(parameters, before, gradients)
    ):
        fraction = selected_fraction if index == selected_index else "010"
        transformed = spectral_transform(
            gradient, f"pion_blend_psspp_norm_b{fraction}", 3, 2, .45
        )
        torch.testing.assert_close(parameter, initial - .01 * transformed)


def test_qkv_layer_profile_can_use_different_alpha_by_layer():
    torch.manual_seed(222)
    layer4 = torch.nn.Parameter(torch.randn(8, 16))
    layer5 = torch.nn.Parameter(torch.randn(8, 16))
    before4, before5 = layer4.detach().clone(), layer5.detach().clone()
    layer4.grad, layer5.grad = torch.randn_like(layer4), torch.randn_like(layer5)
    grad4, grad5 = layer4.grad.clone(), layer5.grad.clone()
    optimizer = MatchedMuon(
        [layer4, layer5],
        lr=.01,
        schedule=parse_schedule("0:pion_profile_qkv_l4_a075_order_ppsp:3:1:.5"),
        parameter_names=[
            "gpt_neox.layers.4.attention.query_key_value.weight",
            "gpt_neox.layers.5.attention.query_key_value.weight",
        ],
    )
    optimizer.step()
    torch.testing.assert_close(
        layer4, before4 - .01 * spectral_transform(grad4, "pion_order_ppsp", 3, 1, .75)
    )
    torch.testing.assert_close(
        layer5, before5 - .01 * spectral_transform(grad5, "pion_order_ppsp", 3, 1, .5)
    )


def test_qkv_layer_profile_can_use_different_order_by_layer():
    torch.manual_seed(223)
    early = torch.nn.Parameter(torch.randn(8, 16))
    late = torch.nn.Parameter(torch.randn(8, 16))
    early_before, late_before = early.detach().clone(), late.detach().clone()
    early.grad, late.grad = torch.randn_like(early), torch.randn_like(late)
    early_grad, late_grad = early.grad.clone(), late.grad.clone()
    optimizer = MatchedMuon(
        [early, late],
        lr=.01,
        schedule=parse_schedule("0:pion_profile_qkv_splitorder_order_ppsp:3:1:.5"),
        parameter_names=[
            "gpt_neox.layers.1.attention.query_key_value.weight",
            "gpt_neox.layers.5.attention.query_key_value.weight",
        ],
    )
    optimizer.step()
    torch.testing.assert_close(
        early, early_before - .01 * spectral_transform(early_grad, "pion_order_ppps", 3, 1, .5)
    )
    torch.testing.assert_close(
        late, late_before - .01 * spectral_transform(late_grad, "pion_order_ppsp", 3, 1, .5)
    )


def test_heterogeneous_profile_changes_qkv_and_contract_alpha_only():
    torch.manual_seed(224)
    names = [
        "gpt_neox.layers.0.attention.query_key_value.weight",
        "gpt_neox.layers.0.attention.dense.weight",
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight",
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight",
    ]
    parameters = [torch.nn.Parameter(torch.randn(8, 16)) for _ in names]
    before = [parameter.detach().clone() for parameter in parameters]
    gradients = [torch.randn_like(parameter) for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    optimizer = MatchedMuon(
        parameters,
        lr=.01,
        momentum=0.0,
        schedule=parse_schedule("0:pion_profile_qkv_a075_contract_a025:3:2:.45"),
        parameter_names=names,
    )
    optimizer.step()
    expected_alphas = (.75, .45, .45, .25)
    for parameter, initial, gradient, alpha in zip(
        parameters, before, gradients, expected_alphas
    ):
        expected = initial - .01 * spectral_transform(gradient, "pion", 3, 2, alpha)
        torch.testing.assert_close(parameter, expected)


def test_full_state_resume_restores_trajectory_and_rng(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.tensor([7], dtype=torch.uint8))
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state: None)
    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    schedule = parse_schedule("0:muon5ns:0:0:0")
    model = torch.nn.Linear(6, 4)
    matrix = MatchedMuon([model.weight], lr=.01, schedule=schedule)
    auxiliary = torch.optim.AdamW([model.bias], lr=.001)

    model.weight.grad = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 100
    model.bias.grad = torch.arange(4, dtype=torch.float32) / 100
    matrix.step()
    auxiliary.step()
    checkpoint = tmp_path / "state.pt"
    config = {"semantic_config_hash": "resume-test"}
    save_checkpoint(checkpoint, model, matrix, auxiliary, 1, config)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))

    resumed = torch.nn.Linear(6, 4)
    resumed_matrix = MatchedMuon([resumed.weight], lr=.01, schedule=schedule)
    resumed_auxiliary = torch.optim.AdamW([resumed.bias], lr=.001)
    next_update = load_checkpoint(
        checkpoint, resumed, resumed_matrix, resumed_auxiliary, "resume-test"
    )
    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(())))
    assert next_update == 1
    assert actual_rng == expected_rng

    second_weight_grad = torch.flip(
        torch.arange(24, dtype=torch.float32).reshape(4, 6), dims=(1,)
    ) / 77
    second_bias_grad = torch.arange(4, dtype=torch.float32) / 77
    for candidate, matrix_opt, aux_opt in (
        (model, matrix, auxiliary),
        (resumed, resumed_matrix, resumed_auxiliary),
    ):
        candidate.weight.grad = second_weight_grad.clone()
        candidate.bias.grad = second_bias_grad.clone()
        matrix_opt.step()
        aux_opt.step()
    for left, right in zip(model.parameters(), resumed.parameters()):
        assert torch.equal(left, right)
