from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from .sweep import K_VALUES, METHODS


METHOD_LABELS = {
    "muon": "Muon",
    "adaptive_wv_wout": "Adaptive WV+Wout",
}
METHOD_COLORS = {
    "muon": "#d62728",
    "adaptive_wv_wout": "#1f77b4",
}
METHOD_STYLES = {
    "muon": "-",
    "adaptive_wv_wout": "--",
}
EVAL_KINDS = ("source", "only_m1", "only_m2", "mixed")
EVAL_LABELS = ("Source", "Only M1", "Only M2", "Mixed 4 M1 + 1 M2")


def load_runs(root: Path) -> dict[tuple[int, str], dict]:
    runs = {}
    for K in K_VALUES:
        for method in METHODS:
            path = root / f"K{K}" / method / "metrics.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            archive = np.load(path, allow_pickle=False)
            runs[(K, method)] = {key: archive[key] for key in archive.files}
    return runs


def late_mask(steps: np.ndarray) -> np.ndarray:
    mask = (steps >= 800) & (steps <= 1000)
    if mask.any():
        return mask
    cutoff = float(steps.max()) * 0.8
    return steps >= cutoff


def make_rows(runs: dict[tuple[int, str], dict]) -> list[dict]:
    rows = []
    for K in K_VALUES:
        muon = runs[(K, "muon")]
        steps = np.asarray(muon["eval_steps"])
        mask = late_mask(steps)
        muon_late = np.asarray(muon["eval_loss_history"])[mask].mean(axis=0)
        for method in METHODS:
            run = runs[(K, method)]
            losses = np.asarray(run["eval_loss_history"])
            late = losses[mask].mean(axis=0)
            ranks = np.asarray(run["k_history"])[-1]
            for index, kind in enumerate(EVAL_KINDS):
                rows.append(
                    {
                        "K": K,
                        "n_Mperp": K - 5,
                        "eval_kind": kind,
                        "method": METHOD_LABELS[method],
                        "late_mean_loss_800_1000": late[index],
                        "late_rel_delta_pct_vs_muon": (
                            100.0 * (late[index] - muon_late[index]) / muon_late[index]
                        ),
                        "normalized_late_loss": late[index]
                        / float(run["zero_predictor_losses"][index]),
                        "final_ranks": " ".join(str(int(value)) for value in ranks),
                    }
                )
    return rows


def save_rows(root: Path, rows: list[dict]) -> None:
    with (root / "000_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_STYLES[method],
            lw=2.2,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def robust_limits(values: list[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([value[np.isfinite(value)] for value in values])
    low, high = np.percentile(combined, [1, 99])
    margin = max((high - low) * 0.12, 1e-8)
    return max(0.0, low - margin), high + margin


def plot_eval_curves(
    root: Path,
    runs: dict[tuple[int, str], dict],
    after600: bool,
) -> None:
    fig, axes = plt.subplots(
        len(K_VALUES),
        len(EVAL_KINDS) + 1,
        figsize=(18, 3.8 * len(K_VALUES)),
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]},
        constrained_layout=True,
    )
    for row, K in enumerate(K_VALUES):
        for column, (kind, label) in enumerate(zip(EVAL_KINDS, EVAL_LABELS)):
            ax = axes[row, column]
            curves = []
            for method in METHODS:
                run = runs[(K, method)]
                steps = np.asarray(run["eval_steps"])
                curve = np.asarray(run["eval_loss_history"])[:, column]
                if after600:
                    mask = steps >= 600
                    if not mask.any():
                        mask = steps >= float(steps.max()) * 0.8
                else:
                    mask = np.ones_like(steps, dtype=bool)
                ax.plot(
                    steps[mask],
                    curve[mask],
                    color=METHOD_COLORS[method],
                    linestyle=METHOD_STYLES[method],
                    lw=1.8,
                )
                curves.append(curve[mask])
            if after600:
                ax.set_ylim(*robust_limits(curves))
            ax.set_title(f"K={K}, Mperp={K-5}\n{label}", pad=10)
            ax.set_xlabel("step")
            ax.set_ylabel("L2 loss")
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(True, alpha=0.22)
        note = axes[row, -1]
        note.axis("off")
        if row == 0:
            note.legend(
                handles=legend_handles(),
                loc="upper left",
                title="METHOD",
                frameon=True,
            )
        note.text(
            0,
            0.38,
            "5 distinct signal tokens\n"
            "train groups random, E[M1:M2]=4:1\n"
            f"{K-5} unit-norm Mperp tokens\n"
            "matched train/eval K",
            va="top",
            fontsize=10,
        )
    suffix = "after600_zoomed" if after600 else "full"
    fig.suptitle(f"Experiment 202 evaluation loss ({suffix})", fontsize=16)
    fig.savefig(root / f"000_all_methods_eval_loss_{suffix}.png", dpi=180)
    plt.close(fig)


def plot_rank_history(
    root: Path,
    runs: dict[tuple[int, str], dict],
) -> None:
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15, 7),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.8]},
        constrained_layout=True,
    )
    parameter_names = ("WQ", "WK", "WV", "Wout")
    for row, K in enumerate((23, 30)):
        run = runs[(K, "adaptive_wv_wout")]
        ranks = np.asarray(run["k_history"])
        for column, parameter_index in enumerate((2, 3)):
            ax = axes[row, column]
            ax.plot(
                np.arange(len(ranks)),
                ranks[:, parameter_index],
                color=METHOD_COLORS["adaptive_wv_wout"],
                lw=2,
            )
            ax.set_title(f"K={K}: {parameter_names[parameter_index]} rank")
            ax.set_xlabel("step")
            ax.set_ylabel("selected rank")
            ax.grid(True, alpha=0.22)
        axes[row, 2].axis("off")
        axes[row, 2].text(
            0,
            0.85,
            "WQ and WK remain full rank.\n"
            "WV and Wout use hard top-k.\n\n"
            "Rank is estimated from the\n"
            "momentum-buffer spectrum.",
            va="top",
        )
        axes[row, 3].axis("off")
        if row == 0:
            axes[row, 3].legend(
                handles=legend_handles()[1:],
                loc="upper left",
                title="METHOD",
                frameon=True,
            )
    fig.suptitle("Adaptive rank history at representative K values", fontsize=15)
    fig.savefig(root / "000_all_methods_rank_history.png", dpi=180)
    plt.close(fig)


def plot_heatmap(root: Path, rows: list[dict]) -> None:
    values = np.asarray(
        [
            [
                float(
                    next(
                        row
                        for row in rows
                        if row["K"] == K
                        and row["eval_kind"] == kind
                        and row["method"] == METHOD_LABELS["adaptive_wv_wout"]
                    )["late_rel_delta_pct_vs_muon"]
                )
                for kind in EVAL_KINDS
            ]
            for K in K_VALUES
        ]
    )
    limit = max(1.0, float(np.max(np.abs(values))))
    fig, (ax, cax, note) = plt.subplots(
        1,
        3,
        figsize=(15, 8),
        gridspec_kw={"width_ratios": [4.8, 0.18, 1.7]},
        constrained_layout=True,
    )
    image = ax.imshow(
        values,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(range(len(EVAL_LABELS)), EVAL_LABELS)
    ax.set_yticks(
        range(len(K_VALUES)),
        [f"K={K}\nMperp={K-5}" for K in K_VALUES],
    )
    ax.set_xlabel("evaluation distribution")
    ax.set_ylabel("matched training/evaluation K")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:+.1f}%", ha="center", va="center")
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("late loss delta vs Muon (%)")
    note.axis("off")
    note.text(
        0,
        0.9,
        "negative = adaptive lower loss\n"
        "positive = Muon lower loss\n\n"
        "late window = steps 800-1000\n"
        "adaptive parameters = WV + Wout\n"
        "hard top-k update\n\n"
        "seed = 0",
        va="top",
    )
    fig.suptitle("Adaptive WV+Wout late loss delta vs Muon", fontsize=16)
    fig.savefig(root / "000_heatmap_adaptiveWVWout_all_eval_delta_vs_muon.png", dpi=190)
    plt.close(fig)


def write_readme(root: Path, rows: list[dict]) -> None:
    lines = [
        "# Experiment 202 result summary",
        "",
        "Negative deltas mean Adaptive WV+Wout has lower step-800-1000 loss than Muon.",
        "",
        "| K | Mperp | Source | Only M1 | Only M2 | Mixed |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for K in K_VALUES:
        deltas = []
        for kind in EVAL_KINDS:
            row = next(
                row
                for row in rows
                if row["K"] == K
                and row["eval_kind"] == kind
                and row["method"] == METHOD_LABELS["adaptive_wv_wout"]
            )
            deltas.append(float(row["late_rel_delta_pct_vs_muon"]))
        lines.append(
            f"| {K} | {K-5} | "
            + " | ".join(f"{value:+.2f}%" for value in deltas)
            + " |"
        )
    lines.extend(
        [
            "",
            "The seed-0 result shows an intermediate K/noise window: adaptive is strongest at "
            "K=20-23, weakens at K=25-28, and reverses at K=30. This is evidence for a "
            "regime change, not yet a statistically established threshold; additional seeds "
            "and pooling/rank controls are required.",
        ]
    )
    (root / "000_README_conclusion.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    args.input.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.input)
    rows = make_rows(runs)
    save_rows(args.input, rows)
    plot_eval_curves(args.input, runs, after600=False)
    plot_eval_curves(args.input, runs, after600=True)
    plot_rank_history(args.input, runs)
    plot_heatmap(args.input, rows)
    write_readme(args.input, rows)
    print(f"plots written to {args.input}")


if __name__ == "__main__":
    main()
