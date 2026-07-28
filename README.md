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

Each token is a vector in an ambient space of dimension `d1=100`. We sample a
`100 x 100` Gaussian matrix, orthogonalize it by QR decomposition, and split
the resulting orthonormal basis into a signal dictionary and a nuisance
dictionary:

```math
O = [\,M \;\; M_{\perp}\,],
\qquad
M \in \mathbb{R}^{100 \times 20},
\qquad
M_{\perp} \in \mathbb{R}^{100 \times 80}.
```

The 20 signal directions are divided into two groups of 10:

```math
M = [\,M_1 \;\; M_2\,],
\qquad
M_1,M_2 \in \mathbb{R}^{100 \times 10}.
```

Because all columns come from one orthonormal basis,

```math
M^{\mathsf T}M=I,
\qquad
M_{\perp}^{\mathsf T}M_{\perp}=I,
\qquad
M^{\mathsf T}M_{\perp}=0.
```

Therefore:

- `M1` and `M2` contain the useful signal features.
- `Mperp` contains nuisance features orthogonal to every signal feature.
- Every signal and nuisance token has Euclidean norm one.

### Fixed five-signal sampling

One sample is a sequence of `K` row vectors:

```math
X =
\begin{bmatrix}
x_1^{\mathsf T}\\
\vdots\\
x_K^{\mathsf T}
\end{bmatrix}
\in \mathbb{R}^{K \times 100}.
```

Every sample contains **exactly five signal tokens** and `K-5` nuisance
tokens. Increasing `K` therefore adds only nuisance tokens; it never increases
the signal-token count.

For each of the five signal positions, independently choose its group:

```math
\Pr(g_s=M_1)=0.8,
\qquad
\Pr(g_s=M_2)=0.2.
```

Consequently, the number of M1 tokens follows a binomial distribution:

```math
n_{M_1} \sim \mathrm{Binomial}(5,0.8),
\qquad
n_{M_2}=5-n_{M_1},
```

and

```math
\mathbb{E}[n_{M_1}]=4,
\qquad
\mathbb{E}[n_{M_2}]=1.
```

The realized M1/M2 counts vary across samples. After choosing the counts,
feature indices are sampled **without replacement** within each group.
Therefore, all five signal tokens correspond to distinct signal features. A
signal token is one column of `M1` or `M2`:

```math
x_s = M_{:,j_s}.
```

Each nuisance token independently selects one of the 80 columns of
`Mperp`:

```math
r \sim \mathrm{Uniform}(\{1,\ldots,80\}),
\qquad
x_{\mathrm{noise}}=M_{\perp,:,r}.
```

Finally, the five signal tokens and `K-5` nuisance tokens are randomly
permuted. This experiment uses:

- no additive Gaussian noise;
- no amplitude difference between M1, M2, and Mperp tokens;
- no positional encoding.

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

The latent vector records which of the 20 signal features occur in the
sequence:

```math
z \in \{0,1\}^{20},
\qquad
z_j =
\begin{cases}
1 & \text{if signal feature } j \text{ occurs in } X,\\
0 & \text{otherwise}.
\end{cases}
```

All five signal features are distinct, hence

```math
\lVert z\rVert_0=5.
```

Mperp tokens never activate a coordinate of `z`.

The teacher is a fixed binary matrix:

```math
W_{\mathrm{true}} \in \{0,1\}^{50 \times 20}.
```

Each column of `W_true` contains exactly five ones, corresponding to binary
density `0.1`. The regression target is

```math
y = W_{\mathrm{true}}z \in \mathbb{R}^{50}.
```

Thus, `y` is the sum of the five teacher-code columns selected by the active
signal features. M1 and M2 use the same teacher construction and receive no
additional label weighting.

## Model

The network is a single-head softmax attention layer followed by a ReLU value
map, sum pooling, and a linear readout:

```math
Q=XW_Q,
\qquad
K_{\mathrm{att}}=XW_K.
```

The row-wise attention matrix and denoised token representation are

```math
A =
\mathrm{softmax}\!\left(
\frac{QK_{\mathrm{att}}^{\mathsf T}}{\sqrt{d_{qk}}}
\right),
\qquad
\widetilde X=AX.
```

The value map, pooling operation, and prediction are

```math
V=\mathrm{ReLU}(\widetilde XW_V),
\qquad
h=\sum_{k=1}^{K}V_{k,:},
\qquad
\widehat y=hW_{\mathrm{out}}^{\mathsf T}.
```

There is no additional MLP. Pooling is a **sum**, not a mean, so readout scale
can depend on K.

| Quantity | Shape |
|---|---:|
| `X`, `X_tilde` | `K x 100` |
| `WQ`, `WK`, `WV` | `100 x 100` |
| `Q`, `K_att`, `V` | `K x 100` |
| `A` | `K x K` |
| `h` | `100` |
| `Wout` | `50 x 100` |
| `y_hat`, `y` | `50` |

For minibatch size `N=32`, the training objective is

```math
\mathcal{L}
=
\frac{1}{2N}
\sum_{i=1}^{N}
\left\lVert
\widehat y^{(i)}-y^{(i)}
\right\rVert_2^2.
```

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
