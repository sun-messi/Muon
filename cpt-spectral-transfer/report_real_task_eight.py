#!/usr/bin/env python3
"""Compact eight-theme seed-42 CPT comparison, with traceable raw tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

TASKS = ("pile_reference", "wikitext", "stackmathqa", "openwebmath",
         "pubmedqa", "codesearchnet", "chemistry", "ord")
LABELS = ("Pile reference", "WikiText-103", "StackMathQA", "OpenWebMath",
          "PubMedQA", "CodeSearchNet Python", "Chemistry SE", "ORD reactions")
METHODS = ("adam", "muon", "pion")
METHOD_LABELS = {"adam": "AdamW", "muon": "Full Muon", "pion": "Pion 3P2S"}
COLORS = {"adam": "#7b3294", "muon": "#202020", "pion": "#e66101"}


def rows(path: Path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save_csv(path: Path, data: list[dict]):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "comparison"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    streams, gradients, one_steps = {}, {}, {}
    endpoints, gradient_summary, one_step_all = [], [], []
    for task in TASKS:
        for method in METHODS:
            tag = f"{task}_{method}_seed42"
            stream = [json.loads(line) for line in (root / "training" / tag / "events.jsonl").read_text().splitlines()]
            if stream[-1]["event"] != "complete":
                raise RuntimeError(f"incomplete {tag}")
            streams[task, method] = stream
            raw = rows(root / "diagnostics/gradients" / tag / "gradient_cosine_raw.csv")
            gradients[task, method] = raw
            one = rows(root / "diagnostics/one_step" / tag / "one_step.csv")
            one_steps[task, method] = one
            one_step_all.extend(one)
            initial = next(row for row in stream if row.get("event") == "validation" and row.get("step") == 0)
            early = next(row for row in stream if row.get("event") == "validation" and row.get("step") == 50)
            hundred = next(row for row in stream if row.get("event") == "validation" and row.get("step") == 100)
            final = stream[-1]
            endpoints.append(dict(task=task, method=method, seed=42,
                                  new_ce_step0=initial.get("full_primary_ce", initial.get("primary_ce")),
                                  old_ce_step0=initial.get("full_retention_ce", initial.get("retention_monitor_ce")),
                                  new_ce_step50=early["primary_ce"],
                                  new_ce_step100=hundred["primary_ce"],
                                  new_ce_step610=final["full_primary_ce"],
                                  old_ce_step610=final["full_retention_ce"]))
            for step in sorted({int(row["step"]) for row in raw}):
                values = np.array([float(row["cosine_all"]) for row in raw if int(row["step"]) == step])
                gradient_summary.append(dict(task=task, method=method, step=step,
                                             probe_count=len(values), cosine_mean=float(values.mean()),
                                             cosine_sd=float(values.std(ddof=1))))
    save_csv(output / "endpoints.csv", endpoints)
    save_csv(output / "gradient_summary.csv", gradient_summary)
    save_csv(output / "one_step_all.csv", one_step_all)
    index = {(r["task"], r["method"]): r for r in endpoints}
    relative = []
    for task in TASKS:
        pion = float(index[task, "pion"]["new_ce_step610"])
        pion_old = float(index[task, "pion"]["old_ce_step610"])
        for baseline in ("muon", "adam"):
            base = index[task, baseline]
            relative.append(dict(task=task, baseline=baseline,
                                 pion_minus_baseline_new_ce=pion-float(base["new_ce_step610"]),
                                 pion_minus_baseline_old_ce=pion_old-float(base["old_ce_step610"])))
    save_csv(output / "pion_differences.csv", relative)

    pdf_path = output / "eight_theme_cpt_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Eight-topic Pythia-31M CPT | seed 42", fontsize=20, y=.93)
        lines = (
            "Model: pretrained EleutherAI/Pythia-31M; sequence length 1024; full-token causal CE.\n"
            "Training: 610 updates, effective batch 32; 15,616 New + 3,904 identical Pile Old windows.\n"
            "Methods: AdamW, Full Muon, Pion 3P2S alpha=0.45; no per-task retuning.\n"
            "New topics: Pile reference, WikiText, StackMathQA, OpenWebMath, PubMedQA,\n"
            "CodeSearchNet Python, Chemistry StackExchange, ORD reactions.\n\n"
            "New validation: 977 independent windows per topic; Old: same 977 Pile test windows.\n"
            "Each method within a topic shares starting model, training tape, and sample order.\n"
            "New CE across topics has different inherent difficulty: compare methods within each topic.\n"
            "Gradient shading is SD across 8 probe minibatches, not seed uncertainty.\n"
            "One-step Old CE delta measures immediate effect, not the entire forgetting trajectory.\n"
            "One seed only: descriptive mechanism study, not a statistical stability claim."
        )
        fig.text(.08, .78, lines, va="top", fontsize=12, linespacing=1.7)
        fig.text(.08, .08, f"Source: {root}", fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        for block in (range(0, 4), range(4, 8)):
            fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.2), sharex=True)
            for i, task_index in enumerate(block):
                task, label = TASKS[task_index], LABELS[task_index]
                for method in METHODS:
                    valid = [r for r in streams[task, method] if r.get("event") == "validation"]
                    x = [r["step"] for r in valid]
                    axes[i, 0].plot(x, [r["primary_ce"] for r in valid], color=COLORS[method],
                                    label=METHOD_LABELS[method], lw=1.7)
                    axes[i, 1].plot(x, [r["retention_monitor_ce"] for r in valid],
                                    color=COLORS[method], lw=1.7)
                axes[i, 0].set_ylabel(label + "\nNew CE")
                axes[i, 1].set_ylabel("Pile Old CE")
                for ax in axes[i]: ax.grid(alpha=.2)
            axes[0, 0].set_title("Held-out New CE (32-row monitor)")
            axes[0, 1].set_title("Pile Old CE (32-row monitor)")
            axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
            for ax in axes[-1]: ax.set_xlabel("Update")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for ax, task in zip(axes.flat, ("pile_reference", "openwebmath", "chemistry", "ord")):
            for method in METHODS:
                data = [r for r in gradient_summary if r["task"] == task and r["method"] == method]
                x = np.array([r["step"] for r in data]); y = np.array([r["cosine_mean"] for r in data])
                sd = np.array([r["cosine_sd"] for r in data])
                ax.plot(x, y, color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
                ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.10)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_title(LABELS[TASKS.index(task)]); ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(0, 610); ax.set_xticks([0, 1, 10, 100, 610])
            ax.set_xticklabels(["0", "1", "10", "100", "610"])
            ax.grid(alpha=.2); ax.set_xlabel("Update"); ax.set_ylabel("Old/New gradient cosine")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle("Raw gradient cosine | representative domains", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, .96]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for ax, task in zip(axes.flat, ("wikitext", "stackmathqa", "pubmedqa", "codesearchnet")):
            for method in METHODS:
                data = [r for r in gradient_summary if r["task"] == task and r["method"] == method]
                x = np.array([r["step"] for r in data]); y = np.array([r["cosine_mean"] for r in data])
                sd = np.array([r["cosine_sd"] for r in data])
                ax.plot(x, y, color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
                ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.10)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_title(LABELS[TASKS.index(task)]); ax.set_xscale("symlog", linthresh=10)
            ax.set_xlim(0, 610); ax.set_xticks([0, 1, 10, 100, 610])
            ax.set_xticklabels(["0", "1", "10", "100", "610"])
            ax.grid(alpha=.2); ax.set_xlabel("Update"); ax.set_ylabel("Old/New gradient cosine")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle("Raw gradient cosine | remaining domains", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, .96]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for ax, task in zip(axes.flat, ("pile_reference", "openwebmath", "chemistry", "ord")):
            for method in METHODS:
                data = [r for r in one_steps[task, method] if r["mixture"] == "new_only"]
                ax.plot([int(r["step"]) for r in data], [float(r["delta_old_ce"]) for r in data],
                        color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
            ax.axhline(0, color="gray", lw=.8); ax.set_title(LABELS[TASKS.index(task)])
            ax.set_xlabel("Checkpoint update"); ax.set_ylabel("One New-only step: Delta Old CE")
            ax.grid(alpha=.2)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for ax, task in zip(axes.flat, ("wikitext", "stackmathqa", "pubmedqa", "codesearchnet")):
            for method in METHODS:
                data = [r for r in one_steps[task, method] if r["mixture"] == "new_only"]
                ax.plot([int(r["step"]) for r in data], [float(r["delta_old_ce"]) for r in data],
                        color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
            ax.axhline(0, color="gray", lw=.8); ax.set_title(LABELS[TASKS.index(task)])
            ax.set_xlabel("Checkpoint update"); ax.set_ylabel("One New-only step: Delta Old CE")
            ax.grid(alpha=.2)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3), sharey=True)
        positions = np.arange(len(TASKS))
        for i, method in enumerate(("muon", "pion")):
            for j, metric in enumerate(("new_ce_step610", "old_ce_step610")):
                ax = axes[i, j]
                values = [float(index[task, method][metric])-float(index[task, "adam"][metric])
                          for task in TASKS]
                ax.barh(positions, values, color=COLORS[method], height=.66)
                ax.axvline(0, color="gray", lw=.8)
                ax.set_yticks(positions, LABELS)
                ax.invert_yaxis()
                ax.set_xlabel("Delta CE vs AdamW (lower is better)")
                ax.set_title(f"{METHOD_LABELS[method]} | {'New' if j == 0 else 'Pile Old'}")
                ax.grid(axis="x", alpha=.2)
        fig.suptitle("Within-topic Old/New trade-off at update 610", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, .95]); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.7, 6.2))
        x = np.arange(len(TASKS))
        width = .34
        for offset, baseline, color in ((-.5, "muon", COLORS["muon"]), (.5, "adam", COLORS["adam"])):
            vals = [float(next(r["pion_minus_baseline_new_ce"] for r in relative
                               if r["task"] == task and r["baseline"] == baseline)) for task in TASKS]
            ax.bar(x+offset*width, vals, width, color=color, label=f"Pion - {METHOD_LABELS[baseline]}")
        ax.axhline(0, color="gray", lw=.8)
        ax.set_xticks(x, LABELS, rotation=25, ha="right")
        ax.set_ylabel("Final New CE difference (lower is better for Pion)")
        ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(pdf_path)


if __name__ == "__main__":
    main()
