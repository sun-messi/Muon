#!/usr/bin/env python3
"""Traceable four-domain extension of the existing eight-topic CPT report."""
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

from report_real_task_eight import COLORS, METHODS, METHOD_LABELS, rows, save_csv

TASKS = ("chembl", "pdb", "uniprot", "cve")
LABELS = {"chembl": "ChEMBL 37 SDF", "pdb": "PDB mmCIF",
          "uniprot": "UniProt Swiss-Prot", "cve": "CVE JSON"}


def main():
    global TASKS, LABELS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", choices=("structured", "near_pile"), default="structured")
    parser.add_argument("--extra-reference-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.profile == "near_pile":
        TASKS = ("pile_cc", "openwebtext2", "books3", "pubmed_central")
        LABELS = {"pile_cc": "Pile-CC", "openwebtext2": "OpenWebText2",
                  "books3": "Books3", "pubmed_central": "PubMed Central"}
    root, reference = args.run_root.resolve(), args.reference_root.resolve()
    output = args.output_dir.resolve() if args.output_dir else root / "comparison"
    output.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    streams, gradients, one_steps = {}, {}, {}
    endpoints, gradient_summary, one_step_all, source_rows = [], [], [], []
    for task in TASKS:
        manifest = json.loads((root / "cache" / task / "manifest.json").read_text())
        audit = manifest["source_audit"]
        source_rows.append(dict(task=task, source=audit["source"],
                                train_documents=audit["new_documents_consumed"],
                                validation_documents=audit["validation_documents_consumed"],
                                new_train_windows=15616, old_train_windows=3904,
                                new_validation_windows=977,
                                manifest=str((root / "cache" / task / "manifest.json").resolve())))
        for method in METHODS:
            tag = f"{task}_{method}_seed42"
            stream = [json.loads(line) for line in (root / "training" / tag / "events.jsonl").read_text().splitlines()]
            assert stream[-1]["event"] == "complete", tag
            streams[task, method] = stream
            raw = rows(root / "diagnostics/gradients" / tag / "gradient_cosine_raw.csv")
            gradients[task, method] = raw
            one = rows(root / "diagnostics/one_step" / tag / "one_step.csv")
            one_steps[task, method] = one
            one_step_all.extend(one)
            valid = [row for row in stream if row.get("event") == "validation"]
            initial = valid[0]
            endpoint = dict(task=task, method=method, seed=42,
                            new_ce_step0=initial["full_primary_ce"],
                            old_ce_step0=initial["full_retention_ce"],
                            new_ce_step50=next(row["primary_ce"] for row in valid if row["step"] == 50),
                            new_ce_step100=next(row["primary_ce"] for row in valid if row["step"] == 100),
                            new_ce_step610=stream[-1]["full_primary_ce"],
                            old_ce_step610=stream[-1]["full_retention_ce"])
            endpoint["new_ce_change"] = endpoint["new_ce_step610"] - endpoint["new_ce_step0"]
            endpoint["old_ce_change"] = endpoint["old_ce_step610"] - endpoint["old_ce_step0"]
            endpoints.append(endpoint)
            for step in sorted({int(row["step"]) for row in raw}):
                item = dict(task=task, method=method, step=step)
                for field in ("cosine_all", "cosine_matrix"):
                    values = np.array([float(row[field]) for row in raw if int(row["step"]) == step])
                    item.update({field + "_mean": float(values.mean()),
                                 field + "_sd": float(values.std(ddof=1))})
                item["probe_count"] = len(values)
                gradient_summary.append(item)
    index = {(row["task"], row["method"]): row for row in endpoints}
    differences = []
    for task in TASKS:
        for baseline in ("muon", "adam"):
            differences.append(dict(task=task, baseline=baseline,
                                    pion_minus_baseline_new_ce=index[task, "pion"]["new_ce_step610"]-index[task, baseline]["new_ce_step610"],
                                    pion_minus_baseline_old_ce=index[task, "pion"]["old_ce_step610"]-index[task, baseline]["old_ce_step610"]))
    for name, data in (("endpoints", endpoints), ("gradient_summary", gradient_summary),
                       ("one_step_all", one_step_all), ("pion_differences", differences),
                       ("data_sources", source_rows)):
        save_csv(output / f"{name}.csv", data)

    combined = []
    for previous_root in (reference, *(path.resolve() for path in args.extra_reference_root)):
        completion = previous_root / "complete.json"
        if completion.exists():
            endpoint_path = Path(json.loads(completion.read_text())["report"]).parent / "endpoints.csv"
        else:
            endpoint_path = previous_root / "comparison_v3/endpoints.csv"
        previous = rows(endpoint_path)
        for task in dict.fromkeys(row["task"] for row in previous):
            methods = {row["method"]: row for row in previous if row["task"] == task}
            raw = rows(previous_root / "diagnostics/gradients" / f"{task}_muon_seed42/gradient_cosine_raw.csv")
            start = [float(row["cosine_all"]) for row in raw if int(row["step"]) == 0]
            combined.append(dict(task=task, initial_cosine_mean=float(np.mean(start)),
                                 initial_cosine_sd=float(np.std(start, ddof=1)),
                                 pion_minus_full_new_ce=float(methods["pion"]["new_ce_step610"])-float(methods["muon"]["new_ce_step610"]),
                                 pion_minus_full_old_ce=float(methods["pion"]["old_ce_step610"])-float(methods["muon"]["old_ce_step610"]),
                                 source_run=str(previous_root)))
    for task in TASKS:
        start = next(row for row in gradient_summary if row["task"] == task and row["method"] == "muon" and row["step"] == 0)
        diff = next(row for row in differences if row["task"] == task and row["baseline"] == "muon")
        combined.append(dict(task=task, initial_cosine_mean=start["cosine_all_mean"],
                             initial_cosine_sd=start["cosine_all_sd"],
                             pion_minus_full_new_ce=diff["pion_minus_baseline_new_ce"],
                             pion_minus_full_old_ce=diff["pion_minus_baseline_old_ce"], source_run=str(root)))
    combined_stem = "all_twelve" if args.profile == "structured" and len(combined) == 12 else f"all_{len(combined)}"
    save_csv(output / f"{combined_stem}_topics_cosine_and_differences.csv", combined)
    fixed_summary = []
    for step in (0, 200):
        for task in TASKS:
            raw = rows(root / "diagnostics/fixed" / f"fixed_step{step}_{task}" / "gradient_cosine_raw.csv")
            values = np.array([float(row["cosine_all"]) for row in raw])
            fixed_summary.append(dict(task=task, reference_step=step, cosine_mean=float(values.mean()),
                                      cosine_sd=float(values.std(ddof=1)), probe_count=len(values)))
    save_csv(output / "fixed_checkpoint_gradient_summary.csv", fixed_summary)

    pdf_path = output / ("near_pile_four_cpt_summary.pdf" if args.profile == "near_pile" else "structured_four_cpt_summary.pdf")
    with PdfPages(pdf_path) as pdf:
        def save(fig, stem):
            pdf.savefig(fig)
            fig.savefig(output / f"{stem}.png", dpi=170)
            plt.close(fig)

        fig = plt.figure(figsize=(11.7, 8.3))
        title = "Near-pretraining-source CPT | four Pile components" if args.profile == "near_pile" else "Structured-data CPT | four new domains"
        fig.text(.06, .94, title, fontsize=19, va="top")
        fig.text(.06, .87,
                 "Pythia-31M | context 1024 | full-token causal CE | seed 42\n"
                 "610 updates, effective batch 32; 15,616 New + 3,904 shared Pile Old windows.\n"
                 "AdamW; Full Muon; Pion 3P2S (alpha=0.45). Original eight-topic hyperparameters.\n"
                 "Each topic: 977 held-out New windows. All topics share Old replay and Pile validation.\n"
                 "Colors: AdamW purple, Full Muon black, Pion orange. Lower CE is better.",
                 fontsize=11, va="top", linespacing=1.6)
        columns = ["New topic", "Initial cosine", "AdamW New", "Full New", "Pion New", "Pion - Full"]
        cells = []
        for task in TASKS:
            c = next(row for row in combined if row["task"] == task)
            cells.append([LABELS[task], f"{c['initial_cosine_mean']:.3f}",
                          *[f"{index[task, method]['new_ce_step610']:.4f}" for method in METHODS],
                          f"{c['pion_minus_full_new_ce']:+.4f}"])
        ax = fig.add_axes([.055, .37, .90, .22]); ax.axis("off")
        table = ax.table(cellText=cells, colLabels=columns, cellLoc="center", loc="center",
                         colWidths=[.24, .16, .15, .15, .15, .15])
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.8)
        wins = {}
        for metric in ("new", "old"):
            field = metric+"_ce_step610"
            wins[metric] = {method: sum(index[task, method][field] == min(index[task, m][field] for m in METHODS)
                                        for task in TASKS) for method in METHODS}
        result_text = "\n".join("Lowest final "+metric.capitalize()+" CE: "+"; ".join(
            f"{METHOD_LABELS[method]} {wins[metric][method]}/4" for method in METHODS) for metric in ("new", "old"))
        improved = {method: sum(index[task, method]["new_ce_change"] < 0 for task in TASKS) for method in METHODS}
        result_text += "\nNew CE below pretrained start: "+"; ".join(
            f"{METHOD_LABELS[method]} {improved[method]}/4" for method in METHODS)
        fig.text(.06, .345, result_text, fontsize=9.5, weight="bold", linespacing=1.5, va="top")
        source_text = (
            "Pile-CC, OpenWebText2, Books3, PubMed Central: actual sources in The Pile pretraining mixture.\n"
            "New texts come from the Pile validation split, excluding verified Old/evaluation document IDs.\n"
            "Same source does not establish prior exposure of individual documents, or low interference.\n"
            if args.profile == "near_pile" else
            "ChEMBL: molecular SDF records (not activity labels). UniProt: annotated Swiss-Prot flatfiles.\n"
            "PDB: native mmCIF; files over 96k characters use head/middle/tail 32k excerpts.\n"
            "CVE: published vulnerability records as compact sorted JSON. Splits use hashed record IDs.\n")
        fig.text(.06, .245, source_text +
                 "Single training seed. Probe SD is minibatch variation, not training-seed uncertainty.\n"
                 "Compare optimizers within a topic; raw CE differs across formats and token distributions.",
                 fontsize=10, va="top", linespacing=1.5)
        fig.text(.06, .07, f"Run: {root}\nSources and exact values: data_sources.csv, endpoints.csv; audit: ../audit.json", fontsize=8)
        save(fig, "page1_setting_and_endpoints")

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.2), sharex=True)
        for i, task in enumerate(TASKS):
            for method in METHODS:
                valid = [row for row in streams[task, method] if row.get("event") == "validation"]
                for j, field in enumerate(("primary_ce", "retention_monitor_ce")):
                    axes[i, j].plot([row["step"] for row in valid], [row[field] for row in valid],
                                    color=COLORS[method], lw=1.6, label=METHOD_LABELS[method])
                    full_field = "full_primary_ce" if j == 0 else "full_retention_ce"
                    full = [row for row in valid if row.get(full_field) is not None]
                    axes[i, j].scatter([row["step"] for row in full], [row[full_field] for row in full],
                                       edgecolor=COLORS[method], facecolor="white", s=30, zorder=5)
            axes[i, 0].set_ylabel(LABELS[task]+"\nNew CE")
            axes[i, 1].set_ylabel("Pile Old CE")
            for ax in axes[i]: ax.grid(alpha=.2)
        axes[0, 0].set_title("New validation"); axes[0, 1].set_title("Shared Old validation")
        axes[0, 0].legend(ncol=3, fontsize=8, frameon=False)
        for ax in axes[-1]: ax.set_xlabel("Update")
        fig.suptitle("Lines: fixed 32-window monitor | open circles: full 977-window evaluation", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, .97]); save(fig, "new_old_ce_dynamics")

        for field, title in (("cosine_all", "All-parameter Old/New gradient cosine"),
                             ("cosine_matrix", "Matrix-parameter Old/New gradient cosine")):
            fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
            for ax, task in zip(axes.flat, TASKS):
                for method in METHODS:
                    data = [row for row in gradient_summary if row["task"] == task and row["method"] == method]
                    x = np.array([row["step"] for row in data])
                    y = np.array([row[field+"_mean"] for row in data])
                    sd = np.array([row[field+"_sd"] for row in data])
                    ax.plot(x, y, color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
                    ax.fill_between(x, y-sd, y+sd, color=COLORS[method], alpha=.10)
                ax.axhline(0, color="gray", lw=.8); ax.set_title(LABELS[task])
                ax.set_xscale("symlog", linthresh=10); ax.set_xlim(0, 610)
                ax.set_xticks([0, 1, 10, 100, 610], ["0", "1", "10", "100", "610"])
                ax.set_xlabel("Update"); ax.set_ylabel("Cosine (mean +/- probe SD)"); ax.grid(alpha=.2)
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle(title+" | eight fixed probe batches", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, .96]); save(fig, field+"_dynamics")

        fig, ax = plt.subplots(figsize=(10.5, 6.3))
        positions = np.arange(len(TASKS))
        for offset, step, color in ((-.18, 0, "#999999"), (.18, 200, "#202020")):
            data = [next(row for row in fixed_summary if row["task"] == task and row["reference_step"] == step) for task in TASKS]
            ax.bar(positions+offset, [row["cosine_mean"] for row in data], width=.34,
                   yerr=[row["cosine_sd"] for row in data], capsize=4, color=color,
                   label=f"Shared OpenWebMath Full Muon checkpoint: update {step}")
        ax.axhline(0, color="gray", lw=.8); ax.grid(axis="y", alpha=.2)
        ax.set_xticks(positions, [LABELS[task] for task in TASKS])
        ax.set_ylabel("Old/New gradient cosine (mean +/- probe SD)")
        ax.set_title("Four New domains evaluated at the same model states")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout(); save(fig, "shared_checkpoint_cosine")

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 11.2), sharex=True)
        for i, task in enumerate(TASKS):
            for j, mixture in enumerate(("new_only", "mixed_26new_6old")):
                for method in METHODS:
                    data = [row for row in one_steps[task, method] if row["mixture"] == mixture]
                    axes[i, j].plot([int(row["step"]) for row in data],
                                    [float(row["delta_old_ce"]) for row in data],
                                    color=COLORS[method], marker="o", ms=3, label=METHOD_LABELS[method])
                axes[i, j].axhline(0, color="gray", lw=.8); axes[i, j].grid(alpha=.2)
                axes[i, j].set_ylabel(LABELS[task]+"\nDelta Old CE")
        axes[0, 0].set_title("One New-only update"); axes[0, 1].set_title("One 26-New / 6-Old update")
        axes[0, 0].legend(frameon=False, fontsize=8)
        for ax in axes[-1]: ax.set_xlabel("Checkpoint update")
        fig.suptitle("Independent one-step probes from copied states | positive = immediate Old damage", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, .97]); save(fig, "one_step_old_ce")

        fig, axes = plt.subplots(1, 3, figsize=(11.7, max(7.5, len(combined)*.48+1.5)), sharey=True)
        positions = np.arange(len(combined))
        colors = ["#e66101" if row["task"] in TASKS else "#777777" for row in combined]
        for ax, field, title in zip(axes,
                                   ("initial_cosine_mean", "pion_minus_full_new_ce", "pion_minus_full_old_ce"),
                                   ("Initial Old/New cosine", "Pion - Full final New CE", "Pion - Full final Old CE")):
            values = [row[field] for row in combined]
            ax.barh(positions, values, color=colors, height=.60)
            ax.axvline(0, color="gray", lw=.8); ax.grid(axis="x", alpha=.2)
            ax.set_title(title, fontsize=10)
            ax.set_yticks(positions, [row["task"] for row in combined])
            ax.invert_yaxis()
            extent = max(abs(value) for value in values) or 1
            if field == "initial_cosine_mean" and min(values) >= 0:
                ax.set_xlim(0, max(values)+.25*extent)
            else:
                ax.set_xlim(min(0, min(values))-.38*extent, max(0, max(values))+.35*extent)
            for y, value in zip(positions, values):
                ax.annotate(f"{value:+.3f}" if field != "initial_cosine_mean" else f"{value:.3f}",
                            (value, y), xytext=(3 if value >= 0 else -3, 0),
                            textcoords="offset points", va="center", ha="left" if value >= 0 else "right", fontsize=8)
        axes[0].set_xlabel("Cosine"); axes[1].set_xlabel("Lower favors Pion"); axes[2].set_xlabel("Lower favors Pion")
        fig.suptitle(f"{len(combined)} topics | orange: four new "+("Pile-source controls" if args.profile == "near_pile" else "structured domains"), fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, .95]); save(fig, combined_stem+"_cosine_vs_gap")
    print(pdf_path)


if __name__ == "__main__":
    main()
