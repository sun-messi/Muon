from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import Batch, DataModel
from .model import SparseAttentionRegressor, l2_regression_loss
from .optim import AdaptiveMuon


EVAL_KINDS = ("source", "only_m1", "only_m2", "mixed")
PARAMETER_NAMES = ("WQ", "WK", "WV", "Wout")


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def gradients(
    model: SparseAttentionRegressor,
    batch: Batch,
) -> dict[str, torch.Tensor]:
    prediction, _, _ = model(batch.X)
    loss = l2_regression_loss(prediction, batch.y)
    parameters = model.named_matrix_parameters()
    values = torch.autograd.grad(loss, [parameter for _, parameter in parameters])
    return {name: value.detach() for (name, _), value in zip(parameters, values)}


@torch.no_grad()
def evaluate(model: SparseAttentionRegressor, batch: Batch) -> float:
    prediction, _, _ = model(batch.X)
    return float(l2_regression_loss(prediction, batch.y).item())


def make_eval_batches(
    data: DataModel,
    config: ExperimentConfig,
) -> dict[str, Batch]:
    generator = torch.Generator(device=data.device).manual_seed(config.seed + 30_000)
    return {
        "source": data.sample_source(config.eval_size, generator),
        "only_m1": data.sample_fixed_eval(config.eval_size, "only_m1", generator),
        "only_m2": data.sample_fixed_eval(config.eval_size, "only_m2", generator),
        "mixed": data.sample_fixed_eval(config.eval_size, "mixed", generator),
    }


def train(config: ExperimentConfig, output_dir: Path) -> Path:
    config.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.npz"

    device = resolve_device(config.device)
    dtype = torch.float64
    initialization_generator = torch.Generator(device=device).manual_seed(config.seed)
    data = DataModel.create(config, initialization_generator, device, dtype)
    model = SparseAttentionRegressor(
        config, initialization_generator, device=device, dtype=dtype
    )
    optimizer = AdaptiveMuon(model.named_matrix_parameters(), config)
    eval_batches = make_eval_batches(data, config)
    train_generator = torch.Generator(device=device).manual_seed(config.seed + 1)

    eval_steps = sorted(
        set(range(0, config.steps + 1, config.eval_every)) | {config.steps}
    )
    losses = np.zeros((len(eval_steps), len(EVAL_KINDS)), dtype=np.float64)
    ranks = np.zeros((config.steps + 1, len(PARAMETER_NAMES)), dtype=np.int64)
    raw_ranks = np.zeros_like(ranks)
    ranks[0] = [min(parameter.shape) for _, parameter in model.named_matrix_parameters()]
    raw_ranks[0] = ranks[0]

    eval_row = 0
    for step in range(config.steps + 1):
        if step in eval_steps:
            losses[eval_row] = [
                evaluate(model, eval_batches[kind]) for kind in EVAL_KINDS
            ]
            eval_row += 1
        if step == config.steps:
            break

        batch = data.sample_source(config.batch_size, train_generator)
        diagnostics = optimizer.step(gradients(model, batch), step + 1)
        ranks[step + 1] = [
            diagnostics.selected_ranks[name] for name in PARAMETER_NAMES
        ]
        raw_ranks[step + 1] = [
            diagnostics.raw_ranks[name] for name in PARAMETER_NAMES
        ]

    zero_losses = np.asarray(
        [
            float(
                l2_regression_loss(
                    torch.zeros_like(eval_batches[kind].y),
                    eval_batches[kind].y,
                ).item()
            )
            for kind in EVAL_KINDS
        ],
        dtype=np.float64,
    )
    token_counts = np.asarray(
        [
            [4.0, 1.0, config.noise_tokens],
            [5.0, 0.0, config.noise_tokens],
            [0.0, 5.0, config.noise_tokens],
            [4.0, 1.0, config.noise_tokens],
        ]
    )
    final_weights = {
        f"final_{name}": parameter.detach().cpu().numpy()
        for name, parameter in model.named_matrix_parameters()
    }
    np.savez_compressed(
        metrics_path,
        eval_steps=np.asarray(eval_steps, dtype=np.int64),
        eval_names=np.asarray(EVAL_KINDS),
        eval_loss_history=losses,
        zero_predictor_losses=zero_losses,
        k_history=ranks,
        raw_rank_history=raw_ranks,
        parameter_names=np.asarray(PARAMETER_NAMES),
        expected_token_counts=token_counts,
        dictionary=data.dictionary.detach().cpu().numpy(),
        dictionary_perp=data.dictionary_perp.detach().cpu().numpy(),
        W_true=data.teacher.detach().cpu().numpy(),
        config_json=np.asarray(json.dumps(config.as_dict(), sort_keys=True)),
        **final_weights,
    )
    print(f"saved {metrics_path}")
    return metrics_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=23, dest="sequence_length")
    parser.add_argument(
        "--method", choices=("muon", "adaptive_wv_wout"), default="adaptive_wv_wout"
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        sequence_length=args.sequence_length,
        method=args.method,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_size=args.eval_size,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
    )
    train(config, args.output)


if __name__ == "__main__":
    main()

