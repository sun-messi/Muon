#!/usr/bin/env python3
"""Paper-facing figures and source tables for the one-seed conflict study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

TASKS = ("openwebmath", "finemath4plus", "proofpile2", "stackmathqa")
LABELS = {"openwebmath": "OpenWebMath", "finemath4plus": "FineMath-4plus",
          "proofpile2": "ProofPile-2", "stackmathqa": "StackMathQA"}
METHODS = ("adam", "muon", "pion")
NAMES = {"adam": "AdamW", "muon": "Full Muon", "pion": "Pion 3P2S"}
COLORS = {"adam": "#7b3294", "muon": "#202020", "pion": "#e66101"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})


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
    source_files = ["train_openwebmath.py", "train_smoltalk.py", "spectral.py",
                    "prepare_real_task_conflict.py", "run_real_task_conflict.py",
                    "analyze_real_task_conflict_gradients.py",
                    "analyze_real_task_conflict_one_step.py",
                    "audit_real_task_conflict.py", "report_real_task_conflict.py"]
    code_sha = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in source_files}
    (output / "code_sha256.json").write_text(json.dumps(code_sha, indent=2) + "\n")
    style()
    events = {}
    raw = {}
    one = {}
    endpoints = []
    gradient_rows = []
    one_rows = []
    for task in TASKS:
        for method in METHODS:
            name = f"{task}_{method}_seed42"
            stream = [json.loads(s) for s in (root / "training" / name / "events.jsonl").read_text().splitlines()]
            events[task, method] = [s for s in stream if s.get("event") == "validation"]
            final = stream[-1]
            if final["event"] != "complete":
                raise RuntimeError(f"incomplete {name}")
            endpoints.append(dict(task=task, method=method, seed=42,
                                  old_pile_ce=final["full_retention_ce"],
                                  new_ce=final["full_primary_ce"],
                                  wikitext_ce=final["full_wikitext_ce"]))
            raw[task, method] = read_csv(root / "diagnostics/gradients" / name / "gradient_cosine_raw.csv")
            one[task, method] = read_csv(root / "diagnostics/one_step" / name / "one_step.csv")
            if method != "pion":
                for row in one[task, method]:
                    for key in ("full_muon_same_state_matrix_norm",
                                "same_state_full_old_ce_after", "same_state_full_new_ce_after",
                                "pion_normmatch_ratio", "normmatched_old_ce_after",
                                "normmatched_new_ce_after"):
                        if row.get(key, "").lower() == "nan":
                            row[key] = ""
            one_rows.extend(one[task, method])
            for step in sorted({int(r["step"]) for r in raw[task, method]}):
                subset = [r for r in raw[task, method] if int(r["step"]) == step]
                for measure in ("cosine_all", "cosine_matrix"):
                    vals = np.array([float(r[measure]) for r in subset])
                    gradient_rows.append(dict(task=task, method=method, step=step,
                                              measure=measure, probe_count=len(vals),
                                              mean=float(vals.mean()),
                                              sd=float(vals.std(ddof=1)),
                                              new_ce_mean=float(np.mean([float(r["new_ce"]) for r in subset])),
                                              old_ce_mean=float(np.mean([float(r["old_ce"]) for r in subset]))))
    write_csv(output / "endpoints.csv", endpoints)
    write_csv(output / "gradient_summary.csv", gradient_rows)
    write_csv(output / "one_step_all.csv", one_rows)

    with PdfPages(output / "real_task_conflict_summary.pdf") as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("Real-data Old/New gradient conflict | seed 42", fontsize=18, y=.93)
        text = (
            "Pythia-31M pretrained checkpoint; 1024-token full causal CE; 610 updates; effective batch 32.\n"
            "New: OpenWebMath / FineMath-4plus / ProofPile-2 / StackMathQA.\n"
            "Old: identical Pile replay rows at identical update positions for all four tasks.\n"
            "Training mix: 80% New / 20% Old. AdamW, Full Muon, Pion 3P2S alpha=0.45.\n\n"
            "Measured: eight fixed held-out Old/New minibatch pairs at each saved checkpoint.\n"
            "Cosine is measured on raw gradients, not transformed updates. Shading = batch SD, not seed uncertainty.\n"
            "One-step tests copy both model and optimizer state; evaluation rows are disjoint from update rows.\n"
            "Mixed one-step batch: 26 New + 6 Old (81.25/18.75, nearest batch-32 approximation).\n"
            "Step 610 one-step result is a hypothetical post-training update at final LR.\n\n"
            "CE traces use the same fixed 32-row held-out monitor; endpoint table uses full held-out sets.\n"
            "A negative gradient cosine alone is not proof of forgetting. Single training seed: exploratory only."
        )
        fig.text(.08, .79, text, fontsize=11, va="top", linespacing=1.7)
        fig.text(.08, .15, f"Source root: {root}", fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.5), sharex=True)
        for i, task in enumerate(TASKS):
            for method in METHODS:
                rows = events[task, method]
                x = [r["step"] for r in rows]
                axes[i, 0].plot(x, [r["primary_ce"] for r in rows],
                                color=COLORS[method], label=NAMES[method], lw=1.8)
                axes[i, 1].plot(x, [r["retention_monitor_ce"] for r in rows],
                                color=COLORS[method], label=NAMES[method], lw=1.8)
            axes[i, 0].set_ylabel(LABELS[task] + "\nNew CE")
            axes[i, 1].set_ylabel("Pile Old CE")
            for ax in axes[i]:
                ax.grid(alpha=.2)
        axes[0, 0].legend(ncol=3, loc="upper right", frameon=False)
        axes[-1, 0].set_xlabel("Update")
        axes[-1, 1].set_xlabel("Update")
        fig.suptitle("Held-out New and Pile Old CE during CPT (32-row fixed monitor)", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .96])
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(4, 1, figsize=(11.7, 11.5), sharex=True)
        for i, task in enumerate(TASKS):
            ax = axes[i]
            for method in METHODS:
                series = [r for r in gradient_rows if r["task"] == task and
                          r["method"] == method and r["measure"] == "cosine_all"]
                x = np.array([r["step"] for r in series]); y = np.array([r["mean"] for r in series])
                sd = np.array([r["sd"] for r in series])
                ax.plot(x, y, color=COLORS[method], marker="o", ms=3.5,
                        label=NAMES[method], lw=1.8)
                ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.11)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_ylabel(LABELS[task] + "\nGradient cosine")
            ax.set_xscale("symlog", linthresh=10)
            ax.grid(alpha=.2)
        axes[0].legend(ncol=3, frameon=False)
        axes[-1].set_xticks([0, 1, 5, 10, 20, 50, 100, 200, 400, 610])
        axes[-1].set_xticklabels(["0", "1", "5", "10", "20", "50", "100", "200", "400", "610"])
        axes[-1].set_xlabel("Update (symmetric-log scale to resolve early steps)")
        fig.suptitle("Old/New raw-gradient cosine | mean ± SD across 8 probe batches", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .96]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.5), sharex=True)
        for i, task in enumerate(TASKS):
            for j, mixture in enumerate(("new_only", "mixed_26new_6old")):
                ax = axes[i, j]
                for method in METHODS:
                    series = [r for r in one[task, method] if r["mixture"] == mixture]
                    x = [int(r["step"]) for r in series]
                    y = [float(r["delta_old_ce"]) for r in series]
                    ax.plot(x, y, color=COLORS[method], marker="o", ms=3.5,
                            label=NAMES[method], lw=1.8)
                ax.axhline(0, color="gray", lw=.8)
                ax.set_ylabel(LABELS[task] + "\nDelta Pile Old CE")
                ax.grid(alpha=.2)
        axes[0, 0].set_title("New-only hypothetical step")
        axes[0, 1].set_title("Mixed hypothetical step (26 New + 6 Old)")
        axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
        for ax in axes[-1]: ax.set_xlabel("Checkpoint update")
        fig.suptitle("Actual one-step impact on disjoint held-out Pile Old CE | own trajectories", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .96]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.5), sharex=True)
        for i, task in enumerate(TASKS):
            for j, mixture in enumerate(("new_only", "mixed_26new_6old")):
                ax = axes[i, j]
                series = [r for r in one[task, "pion"] if r["mixture"] == mixture]
                x = [int(r["step"]) for r in series]
                before = [float(r["old_ce_before"]) for r in series]
                ax.plot(x, [float(r["delta_old_ce"]) for r in series],
                        color=COLORS["pion"], marker="o", ms=3.5, label="Pion original")
                ax.plot(x, [float(r["same_state_full_old_ce_after"])-b
                            for r, b in zip(series, before)],
                        color=COLORS["muon"], marker="o", ms=3.5,
                        label="Full Muon, same Pion checkpoint")
                ax.plot(x, [float(r["normmatched_old_ce_after"])-b
                            for r, b in zip(series, before)],
                        color=COLORS["pion"], ls="--", marker="o", ms=3.5,
                        label="Pion matched to Full matrix norm")
                ax.axhline(0, color="gray", lw=.8)
                ax.set_ylabel(LABELS[task] + "\nDelta Pile Old CE")
                ax.grid(alpha=.2)
        axes[0, 0].set_title("New-only hypothetical step")
        axes[0, 1].set_title("Mixed hypothetical step")
        axes[0, 0].legend(ncol=1, fontsize=8, frameon=False)
        for ax in axes[-1]: ax.set_xlabel("Checkpoint update")
        fig.suptitle("Same-state Full Muon vs Pion, including equal-matrix-norm control", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .96]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        for ax, task in zip(axes.flat, TASKS):
            for method in METHODS:
                series = [r for r in gradient_rows if r["task"] == task and
                          r["method"] == method and r["measure"] == "cosine_all"]
                ax.scatter([r["new_ce_mean"] for r in series], [r["mean"] for r in series],
                           color=COLORS[method], label=NAMES[method], s=26, alpha=.9)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_title(LABELS[task]); ax.set_xlabel("Held-out New CE on the same probe batches")
            ax.set_ylabel("Old/New gradient cosine"); ax.grid(alpha=.2)
        axes[0, 0].legend(frameon=False)
        fig.suptitle("Compare methods at similar New CE, not only equal update", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .95]); pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.7), sharey=True)
        fixed_rows = []
        for i, step in enumerate((0, 200)):
            means, sds = [], []
            for task in TASKS:
                path = root / "diagnostics/fixed" / f"fixed_step{step}_{task}" / "gradient_cosine_raw.csv"
                selected = read_csv(path)
                values = np.array([float(r["cosine_all"]) for r in selected])
                means.append(values.mean()); sds.append(values.std(ddof=1))
                fixed_rows.append(dict(step=step, task=task, cosine_mean=means[-1],
                                       cosine_sd=sds[-1], probe_count=len(values)))
            ax = axes[i]
            xx = np.arange(4)
            ax.bar(xx, means, yerr=sds, color="#4c72b0", alpha=.8, capsize=3)
            ax.axhline(0, color="gray", lw=.8)
            ax.set_xticks(xx, [LABELS[t] for t in TASKS], rotation=20, ha="right")
            ax.set_title(f"Same OWM Full Muon checkpoint, step {step}")
            ax.set_ylabel("Old/New raw-gradient cosine")
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("Four New tasks at identical model weights", fontsize=14)
        fig.tight_layout(rect=[0, .02, 1, .92]); pdf.savefig(fig); plt.close(fig)
        write_csv(output / "fixed_checkpoint_task_comparison.csv", fixed_rows)

        fig, ax = plt.subplots(figsize=(11.7, 8.3)); ax.axis("off")
        columns = ["New task", "Method", "Pile Old CE", "New CE", "WikiText CE"]
        cells = [[LABELS[r["task"]], NAMES[r["method"]],
                  f'{r["old_pile_ce"]:.3f}', f'{r["new_ce"]:.3f}', f'{r["wikitext_ce"]:.3f}']
                 for r in endpoints]
        table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
        ax.set_title("Full held-out CE at update 610 | seed 42", fontsize=15, pad=20)
        fig.text(.07, .08, "One seed and eight probe batches do not establish cross-seed stability. "
                 "Interpret negative cosine jointly with one-step Delta Old CE and full Old CE.", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        old_wins = sum(next(r for r in endpoints if r["task"] == task and r["method"] == "pion")
                       ["old_pile_ce"] < min(r["old_pile_ce"] for r in endpoints
                                              if r["task"] == task and r["method"] != "pion")
                       for task in TASKS)
        new_wins = sum(next(r for r in endpoints if r["task"] == task and r["method"] == "pion")
                       ["new_ce"] < min(r["new_ce"] for r in endpoints
                                        if r["task"] == task and r["method"] != "pion")
                       for task in TASKS)
        negative_means = [r for r in gradient_rows if r["measure"] == "cosine_all" and r["mean"] < 0]
        matched = [r for task in TASKS for r in one[task, "pion"]
                   if r["mixture"] == "mixed_26new_6old" and int(r["step"]) in (20, 100, 200, 400)]
        matched_worse = sum(float(r["normmatched_old_ce_after"]) >
                            float(r["same_state_full_old_ce_after"]) for r in matched)
        norm_ratio = float(np.mean([float(r["pion_normmatch_ratio"]) for r in matched]))
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("What this one-seed experiment supports", fontsize=17, y=.92)
        summary = (
            f"At update 610, Pion has the lowest full Pile Old CE in {old_wins}/4 tasks "
            f"and the lowest full New CE in {new_wins}/4 tasks.\n\n"
            f"Only {len(negative_means)}/120 trajectory-level mean raw-gradient cosines are negative; "
            "the small negative values occur near zero, not as a common negative valley.\n\n"
            "Fixed-checkpoint task comparisons likewise separate task differences from optimizer trajectories.\n"
            f"In {matched_worse}/{len(matched)} intermediate mixed one-step probes, norm-matched Pion "
            f"harms Old CE more than Full Muon on the same checkpoint; matching needs "
            f"about {norm_ratio:.1f}x Pion's original matrix-update norm on average.\n\n"
            "Thus, better Old retention here is an observed optimizer outcome, but this data does not "
            "establish a universal negative-gradient conflict mechanism. One training seed cannot "
            "establish repeatability."
        )
        fig.text(.08, .78, summary, fontsize=12, va="top", linespacing=1.65,
                 wrap=True)
        pdf.savefig(fig); plt.close(fig)
    print(output / "real_task_conflict_summary.pdf")


if __name__ == "__main__":
    main()
