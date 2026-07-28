# Adaptive Muon for Sparse Attention Regression

This repository contains a compact, reproducible version of the sparse
attention experiments developed in the larger `muon_project` workspace. It
focuses on one question:

> When useful signal directions are sparse, can hard-top-k Adaptive Muon on
> the value/readout matrices suppress nuisance directions more effectively
> than full Muon?

## Focused result

Experiment 202 fixes five distinct signal tokens per sample and increases
sequence length `K`, so only the number of orthogonal Mperp tokens changes.
The comparison is:

- **Muon**: full-rank Muon on `WQ`, `WK`, `WV`, and `Wout`.
- **Adaptive WV+Wout**: full-rank Muon on `WQ/WK`; hard-top-k Adaptive Muon on
  `WV/Wout`.

The seed-0 sweep found the strongest adaptive gains at intermediate K:

| K | Mperp | Source | Only M1 | Only M2 | Mixed |
|---:|---:|---:|---:|---:|---:|
| 18 | 13 | -0.62% | -0.59% | -0.82% | +0.33% |
| 20 | 15 | -10.06% | -1.82% | -26.82% | -9.43% |
| 23 | 18 | -16.22% | -10.41% | -29.56% | -16.25% |
| 25 | 20 | -5.96% | -5.68% | -8.43% | -5.10% |
| 28 | 23 | -5.19% | -8.32% | +0.00% | -4.32% |
| 30 | 25 | +37.39% | +47.76% | +15.21% | +35.72% |

Negative means Adaptive WV+Wout has lower step-800-1000 mean loss. This is a
seed-0 mechanism result, not a statistically established transition point.

![Adaptive loss delta across K](results/experiment_202/000_heatmap_adaptiveWVWout_all_eval_delta_vs_muon.png)

## Data model

### Signal and nuisance dictionaries

Tokens live in an ambient space of dimension `d1=100`. A random Gaussian
matrix is orthogonalized by QR and split into

$$
\mathcal{O}=[\,M\;\;M_\perp\,],
$$

where

$$
M\in\mathbb{R}^{100\times20},
\qquad
M_\perp\in\mathbb{R}^{100\times80}.
$$

The signal dictionary is divided into two groups:

$$
M=[\,M_1\;\;M_2\,],
\qquad
M_1,M_2\in\mathbb{R}^{100\times10}.
$$

The dictionaries satisfy

$$
M^\top M=I,\qquad
M_\perp^\top M_\perp=I,\qquad
M^\top M_\perp=0.
$$

Thus, M1 and M2 are signal features, while Mperp is an orthogonal nuisance
subspace. Every signal and nuisance token has unit norm.

### Fixed five-signal sampling

Each sample is a sequence

$$
X=[x_1,\ldots,x_K]^\top\in\mathbb{R}^{K\times100}.
$$

Every sample contains exactly five signal tokens and `K-5` Mperp tokens. This
is **not** an iid token-type model: increasing K does not increase the number
of signal tokens.

For each of the five signal slots, independently sample its group:

$$
\Pr(g_s=\mathrm{M1})=0.8,
\qquad
\Pr(g_s=\mathrm{M2})=0.2.
$$

Therefore

$$
n_{\mathrm{M1}}\sim\operatorname{Binomial}(5,0.8),
\qquad
n_{\mathrm{M2}}=5-n_{\mathrm{M1}},
$$

with expected counts

$$
\mathbb{E}[n_{\mathrm{M1}}]=4,
\qquad
\mathbb{E}[n_{\mathrm{M2}}]=1.
$$

The realized counts vary by sample. Conditional on the group counts, features
are sampled **without replacement** inside each group, so the five signal
features are always distinct:

$$
x_s=M_{:,j_s}.
$$

Each nuisance token independently samples one Mperp column:

$$
r\sim\operatorname{Uniform}\{1,\ldots,80\},
\qquad
x_{\mathrm{noise}}=M_{\perp,:,r}.
$$

Signal and nuisance tokens are concatenated and randomly permuted. There is no
additive Gaussian noise and no M1/M2 amplitude difference in this experiment.

The sweep changes only the number of Mperp tokens:

| K | signal tokens | Mperp tokens | expected M1 | expected M2 |
|---:|---:|---:|---:|---:|
| 18 | 5 | 13 | 4 | 1 |
| 20 | 5 | 15 | 4 | 1 |
| 23 | 5 | 18 | 4 | 1 |
| 25 | 5 | 20 | 4 | 1 |
| 28 | 5 | 23 | 4 | 1 |
| 30 | 5 | 25 | 4 | 1 |

### Binary teacher and target

The latent vector

$$
z\in\{0,1\}^{20}
$$

records feature presence. Because all five signal features are distinct,

$$
\lVert z\rVert_0=5.
$$

Mperp tokens do not activate any coordinate of z. The fixed teacher is

$$
W_{\mathrm{true}}\in\{0,1\}^{50\times20}.
$$

Every teacher column contains exactly five ones (`density=0.1`). The target is

$$
y=W_{\mathrm{true}}z\in\mathbb{R}^{50}.
$$

Thus, y is the sum of the five binary code columns corresponding to the active
signal features. The teacher does not reweight M1 and M2.

## Model

The network is a single-head softmax attention layer followed by a ReLU value
map, sum pooling, and a linear readout:

$$
Q=XW_Q,\qquad K_{\mathrm{att}}=XW_K,
$$

$$
A=\operatorname{softmax}
\left(
\frac{QK_{\mathrm{att}}^\top}{\sqrt{d_{qk}}}
\right),
\qquad
\widetilde X=AX,
$$

$$
V=\operatorname{ReLU}(\widetilde XW_V),
\qquad
h=\sum_{k=1}^{K}V_k,
$$

$$
\widehat y=hW_{\mathrm{out}}^\top.
$$

There is no additional MLP. Pooling is a **sum**, not a mean, so readout scale
can depend on K.

Matrix dimensions are

$$
W_Q,W_K,W_V\in\mathbb{R}^{100\times100},
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{50\times100}.
$$

For minibatch size `N=32`, the training objective is

$$
\mathcal{L}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{2}
\left\lVert
\widehat y^{(i)}-y^{(i)}
\right\rVert_2^2.
$$

See [docs/EXPERIMENT_202.md](docs/EXPERIMENT_202.md) for the complete data
model, optimizer, dimensions, and evaluation protocol. See
[docs/CODE_MAP.md](docs/CODE_MAP.md) for how the original numbered scripts were
reorganized into this standalone package.

For the formal mathematical specification, use:

- [Experiment 202 data and network](docs/EXPERIMENT_202_SETUP.md)
- [Rendered PDF](docs/EXPERIMENT_202_SETUP.pdf)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Reproduce experiment 202

Check GPU availability first:

```bash
nvidia-smi
```

Preview the 12 tasks:

```bash
muon-sweep --plan
```

Run the complete sweep and generate plots:

```bash
muon-sweep \
  --output outputs/experiment_202 \
  --gpus 0,1,2,3,4,5,6,7 \
  --jobs-per-gpu 2 \
  --eval-every 1
```

Each run evaluates 5,000 held-out examples at every step to match the original
protocol. For a faster development pass, use `--eval-every 10`.

Run one configuration:

```bash
muon-train \
  --K 23 \
  --method adaptive_wv_wout \
  --output outputs/K23/adaptive_wv_wout
```

Regenerate plots from completed runs:

```bash
muon-plot --input outputs/experiment_202
```

## Repository layout

```text
src/muon_sparse_attention/
  config.py   experiment configuration
  data.py     sparse signal and Mperp data model
  model.py    attention regression network
  optim.py    full and hard-top-k Adaptive Muon
  train.py    single-run trainer and evaluator
  sweep.py    multi-GPU experiment-202 launcher
  plot.py     CSV summaries and non-overlapping plots
docs/
  EXPERIMENT_202.md
results/experiment_202/
  selected result table and figures from the original workspace
tests/
  test_core.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```
