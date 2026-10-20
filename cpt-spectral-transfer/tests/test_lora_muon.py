import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from lora_muon import LoRALinear, expected_trainable_parameters, frozen_state_hash, inject_hidden_lora
from spectral import MatchedMuon, parse_schedule, summarize_spectrum


def test_lora_forward_and_first_gradient():
    torch.manual_seed(7)
    base = torch.nn.Linear(5, 4)
    wrapped = LoRALinear(base, rank=2, alpha=4)
    inputs = torch.randn(3, 5)
    expected = F.linear(inputs, base.weight, base.bias) + 2 * F.linear(
        F.linear(inputs, wrapped.lora_A), wrapped.lora_B
    )
    assert torch.equal(wrapped(inputs), expected)
    assert torch.equal(wrapped(inputs), F.linear(inputs, base.weight, base.bias))
    wrapped(inputs).sum().backward()
    assert torch.count_nonzero(wrapped.lora_A.grad) == 0
    assert torch.count_nonzero(wrapped.lora_B.grad) > 0
    assert base.weight.grad is None
    assert base.bias.grad is None


def test_factor_wise_muon_updates_only_adapter_and_preserves_base():
    torch.manual_seed(9)
    wrapped = LoRALinear(torch.nn.Linear(5, 4), rank=2, alpha=4)
    frozen_before = frozen_state_hash(wrapped)
    before_A = wrapped.lora_A.detach().clone()
    wrapped(torch.randn(3, 5)).square().sum().backward()
    optimizer = MatchedMuon(
        [wrapped.lora_A, wrapped.lora_B],
        lr=0.01,
        momentum=0.95,
        schedule=parse_schedule("0:muon5ns:0:0:0"),
        parameter_names=["lora_A", "lora_B"],
    )
    optimizer.step()
    assert torch.equal(wrapped.lora_A, before_A)
    assert torch.count_nonzero(wrapped.lora_B) > 0
    assert torch.linalg.matrix_rank(wrapped.effective_delta()) <= 2
    assert frozen_state_hash(wrapped) == frozen_before


def test_pythia_hidden_lora_partition_and_exact_counts():
    config = GPTNeoXConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
    )
    for rank in (16, 64):
        torch.manual_seed(42)
        model = GPTNeoXForCausalLM(config)
        injected = inject_hidden_lora(model, rank=rank, alpha=2 * rank)
        assert len(injected.modules) == 24
        assert len(injected.parameters) == 48
        assert sum(parameter.numel() for parameter in injected.parameters) == expected_trainable_parameters(rank)
        assert all(parameter.requires_grad for parameter in injected.parameters)
        assert all(
            parameter.requires_grad == ("lora_A" in name or "lora_B" in name)
            for name, parameter in model.named_parameters()
        )


def test_zero_spectrum_has_zero_ranks():
    summary = summarize_spectrum(torch.zeros(4, 7))
    assert summary["stable_rank"] == 0
    assert summary["rank_gt_0p5"] == 0
    assert summary["energy_rank_90"] == 0
    assert summary["energy_rank_95"] == 0
    assert summary["energy_rank_99"] == 0
