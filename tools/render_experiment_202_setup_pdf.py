#!/usr/bin/env python3
"""Render a concise PDF companion for docs/EXPERIMENT_202_SETUP.md."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "EXPERIMENT_202_SETUP.pdf"


def page(title: str):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.955, title, fontsize=20, fontweight="bold", va="top")
    return fig


def paragraph(fig, y: float, text: str, *, size: float = 10.5, width: int = 96):
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    fig.text(0.08, y, wrapped, fontsize=size, va="top", linespacing=1.35)
    lines = wrapped.count("\n") + 1
    return y - lines * 0.027


def heading(fig, y: float, text: str):
    fig.text(0.08, y, text, fontsize=13.5, fontweight="bold", va="top")
    return y - 0.046


def equation(fig, y: float, text: str, *, size: float = 13):
    fig.text(0.5, y, text, fontsize=size, ha="center", va="top")
    return y - 0.055


def bullet(fig, y: float, text: str, *, width: int = 88):
    wrapped = textwrap.wrap(text, width=width)
    fig.text(0.095, y, r"$\bullet$ " + wrapped[0], fontsize=10.2, va="top")
    for index, line in enumerate(wrapped[1:], start=1):
        fig.text(0.125, y - index * 0.025, line, fontsize=10.2, va="top")
    return y - max(1, len(wrapped)) * 0.027


def render() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        fig = page("Experiment 202: Data Model")
        y = 0.9
        y = paragraph(
            fig,
            y,
            "Focused comparison of full Muon and hard-top-k Adaptive Muon on "
            "WV+Wout. K varies while every sample retains exactly five distinct "
            "signal features.",
        )
        y = heading(fig, y - 0.01, "1. Signal and nuisance dictionaries")
        y = equation(
            fig,
            y,
            r"$d=K_1+K_2,\qquad \mathcal{O}=[\,M\;\;M_\perp\,]$",
        )
        y = equation(
            fig,
            y,
            r"$M^\top M=I,\quad M_\perp^\top M_\perp=I,\quad M^\top M_\perp=0$",
        )
        y = paragraph(
            fig,
            y,
            "A Gaussian d1-by-d1 matrix is orthogonalized by QR. The first "
            "K1 columns of M are M1 features, the next K2 are M2 features, and "
            "Mperp spans the orthogonal nuisance subspace.",
        )
        y = heading(fig, y - 0.01, "2. Fixed five-signal construction")
        y = equation(
            fig,
            y,
            r"$X=[x_1,\ldots,x_K]^\top\in\mathbb{R}^{K\times d_1}$",
        )
        y = bullet(fig, y, "Every sample contains exactly 5 signal tokens.")
        y = bullet(fig, y, "Each signal slot is M1 with probability 0.8 and M2 with probability 0.2.")
        y = equation(
            fig,
            y,
            r"$n_{\rm M1}\sim{\rm Binomial}(5,0.8),\quad n_{\rm M2}=5-n_{\rm M1}$",
        )
        y = equation(
            fig,
            y,
            r"$\mathbb{E}[n_{\rm M1}]=4,\qquad\mathbb{E}[n_{\rm M2}]=1$",
        )
        y = bullet(
            fig,
            y,
            "Features are sampled without replacement inside each group, so all five signal features are distinct.",
        )
        y = bullet(
            fig,
            y,
            "The remaining K-5 tokens are independently sampled unit-norm columns of Mperp.",
        )
        y = bullet(
            fig,
            y,
            "Signal and Mperp amplitudes are both 1; there is no additive Gaussian noise.",
        )
        y = heading(fig, y - 0.01, "3. K sweep")
        y = paragraph(
            fig,
            y,
            "K = 18, 20, 23, 25, 28, 30. The corresponding Mperp counts are "
            "13, 15, 18, 20, 23, 25. K changes only nuisance-token count; signal "
            "count remains five.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("Experiment 202: Label and Network")
        y = 0.9
        y = heading(fig, y, "1. Presence label and binary teacher")
        y = equation(
            fig,
            y,
            r"$z\in\{0,1\}^{d},\qquad \|z\|_0=5$",
        )
        y = paragraph(
            fig,
            y,
            "z marks which signal features occur in the sequence. Mperp tokens "
            "activate no label coordinate.",
        )
        y = equation(
            fig,
            y,
            r"$W_{\rm true}\in\{0,1\}^{C\times d},\qquad y=W_{\rm true}z$",
        )
        y = paragraph(
            fig,
            y,
            "Each teacher column has exactly round(0.1 C) ones. The target is "
            "the sum of the five active binary code columns. There is no M1/M2 "
            "teacher reweighting.",
        )
        y = heading(fig, y - 0.01, "2. Softmax self-attention")
        y = equation(
            fig,
            y,
            r"$Q=XW_Q,\qquad K_{\rm att}=XW_K$",
        )
        y = equation(
            fig,
            y,
            r"$A={\rm softmax}\!\left(QK_{\rm att}^{\top}/\sqrt{d_{qk}}\right)$",
        )
        y = equation(
            fig,
            y,
            r"$\widetilde X=AX$",
        )
        y = heading(fig, y - 0.01, "3. Value map and readout")
        y = equation(
            fig,
            y,
            r"$V={\rm ReLU}(\widetilde XW_V)$",
        )
        y = equation(
            fig,
            y,
            r"$h=\sum_{k=1}^{K}V_k,\qquad \widehat y=W_{\rm out}h$",
        )
        y = paragraph(
            fig,
            y,
            "There is no additional MLP. Pooling is a sum, not a mean, so the "
            "activation/readout scale may depend on K.",
        )
        y = heading(fig, y - 0.01, "4. Exact shapes")
        y = equation(
            fig,
            y,
            r"$W_Q,W_K,W_V\in\mathbb{R}^{100\times100}$",
        )
        y = equation(
            fig,
            y,
            r"$W_{\rm out}\in\mathbb{R}^{50\times100}$",
        )
        y = equation(
            fig,
            y,
            r"$d_1=100,\quad d=20,\quad K_1=K_2=10,\quad C=50$",
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = page("Experiment 202: Loss, Optimizer, and Evaluation")
        y = 0.9
        y = heading(fig, y, "1. L2 regression loss")
        y = equation(
            fig,
            y,
            r"$\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\frac{1}{2}\|\widehat y^{(i)}-y^{(i)}\|_2^2$",
        )
        y = paragraph(
            fig,
            y,
            "Training uses N=32, 1000 steps, learning rate 0.03, and seed 0. "
            "Normalized loss is post-processing only.",
        )
        y = heading(fig, y - 0.01, "2. Full Muon")
        y = equation(
            fig,
            y,
            r"$B_t=0.95B_{t-1}+G_t,\qquad B_t=U\,{\rm diag}(\sigma)V^\top$",
        )
        y = equation(
            fig,
            y,
            r"$\Delta W_t=UV^\top$",
        )
        y = heading(fig, y - 0.01, "3. Hard-top-k Adaptive Muon")
        y = paragraph(
            fig,
            y,
            "WQ and WK remain full Muon. WV and Wout estimate rank from the "
            "momentum-buffer spectrum. The threshold is tau=5 times the median "
            "of the bottom half of the singular values.",
        )
        y = equation(
            fig,
            y,
            r"$k_{\rm raw}=\#\{i:\sigma_i>5\sigma_{\rm noise}\}$",
        )
        y = equation(
            fig,
            y,
            r"$\bar k_t=0.005k_{{\rm raw},t}+0.995\bar k_{t-1}$",
        )
        y = equation(
            fig,
            y,
            r"$\Delta W_t=U_{:,1:k_t}V_{:,1:k_t}^{\top}$",
        )
        y = paragraph(
            fig,
            y,
            "There is no warmup. All singular directions after k receive zero update.",
        )
        y = heading(fig, y - 0.01, "4. Matched-K evaluation")
        y = bullet(fig, y, "Source: five random M1/M2 features, expected ratio 4:1.")
        y = bullet(fig, y, "Only M1: five distinct M1 features.")
        y = bullet(fig, y, "Only M2: five distinct M2 features.")
        y = bullet(fig, y, "Mixed: exactly four M1 features and one M2 feature.")
        y = paragraph(
            fig,
            y,
            "Every evaluation set contains the same K-5 Mperp count as training "
            "and has 5000 fixed held-out samples. The reported late statistic is "
            "mean loss over steps 800-1000.",
        )
        pdf.savefig(fig)
        plt.close(fig)

    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    render()
