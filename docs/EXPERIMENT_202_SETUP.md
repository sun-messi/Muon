# Experiment 202: Data Model and Network

This document specifies the exact data model and network used by the focused
Experiment 202 comparison:

- full-rank Muon;
- hard-top-k Adaptive Muon applied only to `WV` and `Wout`.

The experiment varies sequence length `K` while keeping exactly five signal
tokens per sample. Therefore increasing `K` changes only the number of
orthogonal nuisance tokens.

## 1. Data Generation

### 1.1 Signal and nuisance dictionaries

The ambient token dimension is \(d_1\). The signal subspace has dimension

\[
d = K_1 + K_2.
\]

A Gaussian matrix \(R\in\mathbb{R}^{d_1\times d_1}\) is orthogonalized by QR:

\[
R = \mathcal{O}R_R,\qquad
\mathcal{O} = [\,M\;\;M_\perp\,].
\]

The two dictionaries are

\[
M\in\mathbb{R}^{d_1\times d},
\qquad
M_\perp\in\mathbb{R}^{d_1\times(d_1-d)}.
\]

The first \(K_1\) columns of \(M\) are the M1 features and the next \(K_2\)
columns are the M2 features:

\[
M = [\,M_1\;\;M_2\,],
\qquad
M_1\in\mathbb{R}^{d_1\times K_1},
\qquad
M_2\in\mathbb{R}^{d_1\times K_2}.
\]

All columns of \(M\) and \(M_\perp\) have unit norm, and their subspaces are
orthogonal:

\[
M^\top M=I_d,\qquad
M_\perp^\top M_\perp=I_{d_1-d},\qquad
M^\top M_\perp=0.
\]

### 1.2 Fixed signal count and variable Mperp count

Each sample is a sequence

\[
X=[x_1,\ldots,x_K]^\top\in\mathbb{R}^{K\times d_1}.
\]

Every sample contains exactly

\[
S=5
\]

signal tokens and

\[
K-S=K-5
\]

Mperp tokens. The five signal tokens are generated first; all tokens are then
randomly permuted.

This differs from an iid token-type model. The number of signal tokens is
fixed and does not fluctuate with \(K\).

### 1.3 Random M1/M2 membership of the five signal tokens

For each signal slot \(s\in\{1,\ldots,5\}\), independently sample

\[
\Pr(g_s=\mathrm{M1})=0.8,
\qquad
\Pr(g_s=\mathrm{M2})=0.2.
\]

Consequently,

\[
n_{\mathrm{M1}}\sim\operatorname{Binomial}(5,0.8),
\qquad
n_{\mathrm{M2}}=5-n_{\mathrm{M1}},
\]

and

\[
\mathbb{E}[n_{\mathrm{M1}}]=4,
\qquad
\mathbb{E}[n_{\mathrm{M2}}]=1.
\]

The realized M1/M2 counts vary across samples; only their expectation is 4:1.

Conditional on \(n_{\mathrm{M1}}\) and \(n_{\mathrm{M2}}\):

- choose \(n_{\mathrm{M1}}\) different feature indices without replacement
  from \(\{1,\ldots,K_1\}\);
- choose \(n_{\mathrm{M2}}\) different feature indices without replacement
  from \(\{K_1+1,\ldots,K_1+K_2\}\).

Thus, all five signal features in a sample are distinct. Their tokens are

\[
x_s=M_{:,j_s}.
\]

M1, M2, and Mperp tokens all have the same unit amplitude. There is no
Gaussian additive noise in this experiment.

### 1.4 Mperp tokens

For every nuisance slot, independently sample

\[
r\sim\operatorname{Uniform}\{1,\ldots,d_1-d\},
\qquad
x=M_{\perp,:,r}.
\]

Mperp indices may repeat, but they never contribute to the label.

The K sweep is therefore:

| \(K\) | signal tokens | Mperp tokens | expected M1 | expected M2 | effective Mperp fraction |
|---:|---:|---:|---:|---:|---:|
| 18 | 5 | 13 | 4 | 1 | 0.7222 |
| 20 | 5 | 15 | 4 | 1 | 0.7500 |
| 23 | 5 | 18 | 4 | 1 | 0.7826 |
| 25 | 5 | 20 | 4 | 1 | 0.8000 |
| 28 | 5 | 23 | 4 | 1 | 0.8214 |
| 30 | 5 | 25 | 4 | 1 | 0.8333 |

### 1.5 Latent feature vector and target

The latent vector

\[
z\in\{0,1\}^{d}
\]

records feature presence:

\[
z_j =
\begin{cases}
1,&\text{if signal feature }j\text{ appears in the sequence},\\
0,&\text{otherwise}.
\end{cases}
\]

Because the five signal features are distinct,

\[
\lVert z\rVert_0=5.
\]

The teacher is a fixed binary matrix

\[
W_{\mathrm{true}}\in\{0,1\}^{C\times d}.
\]

Each column contains exactly

\[
\operatorname{round}(0.1C)
\]

ones, placed at randomly selected output coordinates. No column normalization
or M1/M2 reweighting is applied. The target is

\[
y = W_{\mathrm{true}}z\in\mathbb{R}^{C}.
\]

Equivalently, \(y\) is the sum of the five binary code columns associated with
the active signal features. Mperp tokens do not change \(z\) or \(y\).

## 2. Network

### 2.1 Softmax self-attention

For one sample \(X\in\mathbb{R}^{K\times d_1}\), define

\[
Q=XW_Q,\qquad
K_{\mathrm{att}}=XW_K,
\]

where

\[
W_Q,W_K\in\mathbb{R}^{d_1\times d_{qk}}.
\]

The row-normalized attention matrix is

\[
A_{ij}
=
\frac{
\exp\left(q_i^\top k_j/\sqrt{d_{qk}}\right)
}{
\sum_{\ell=1}^{K}
\exp\left(q_i^\top k_\ell/\sqrt{d_{qk}}\right)
},
\qquad
A\in\mathbb{R}^{K\times K}.
\]

The attention-mixed tokens are

\[
\widetilde X=AX\in\mathbb{R}^{K\times d_1}.
\]

### 2.2 Value map, ReLU, and sum pooling

The value representation is

\[
V=\operatorname{ReLU}(\widetilde XW_V)
\in\mathbb{R}^{K\times d_v},
\qquad
W_V\in\mathbb{R}^{d_1\times d_v}.
\]

There is no additional MLP. Token representations are aggregated by a sum:

\[
h=\sum_{k=1}^{K}V_k\in\mathbb{R}^{d_v}.
\]

This is sum pooling, not mean pooling. Therefore the readout scale can depend
on \(K\), which is an important consideration when interpreting the K sweep.

### 2.3 Linear readout

With

\[
W_{\mathrm{out}}\in\mathbb{R}^{C\times d_v},
\]

the prediction is

\[
\widehat y=W_{\mathrm{out}}h\in\mathbb{R}^{C}.
\]

The code uses the row-vector equivalent
\(\widehat y=hW_{\mathrm{out}}^\top\).

## 3. Loss

For minibatch size \(N\), training minimizes

\[
\mathcal{L}
=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{2}
\left\lVert
\widehat y^{(i)}-y^{(i)}
\right\rVert_2^2.
\]

Normalized loss is not used during training. When reported, it is a
post-processing metric:

\[
\mathcal{L}_{\mathrm{norm}}
=
\frac{\mathcal{L}}
{\frac{1}{2}\mathbb{E}\lVert y\rVert_2^2}.
\]

## 4. Exact Dimensions and Training Configuration

| quantity | value |
|---|---:|
| ambient dimension \(d_1\) | 100 |
| M1 dictionary size \(K_1\) | 10 |
| M2 dictionary size \(K_2\) | 10 |
| signal dimension \(d\) | 20 |
| query/key dimension \(d_{qk}\) | 100 |
| value dimension \(d_v\) | 100 |
| output dimension \(C\) | 50 |
| signal tokens per sample | 5 |
| sequence lengths \(K\) | 18, 20, 23, 25, 28, 30 |
| minibatch size \(N\) | 32 |
| training steps | 1000 |
| held-out examples per eval set | 5000 |
| learning rate | 0.03 |
| seed | 0 |

The trainable matrix shapes are

\[
W_Q,W_K,W_V\in\mathbb{R}^{100\times100},
\qquad
W_{\mathrm{out}}\in\mathbb{R}^{50\times100}.
\]

## 5. Optimizer Comparison

### 5.1 Full Muon

For gradient \(G_t\), the momentum buffer is

\[
B_t=0.95B_{t-1}+G_t.
\]

If

\[
B_t=U\operatorname{diag}(\sigma)V^\top,
\]

full Muon uses

\[
\Delta W_t=UV^\top,
\qquad
W_{t+1}=W_t-0.03\Delta W_t.
\]

Full Muon is applied to all four matrices.

### 5.2 Hard-top-k Adaptive Muon on WV and Wout

Adaptive WV+Wout leaves \(W_Q\) and \(W_K\) on full Muon. For \(W_V\) and
\(W_{\mathrm{out}}\), estimate rank from the momentum-buffer singular values.

Let the noise-floor estimate be the median singular value in the bottom half
of the spectrum:

\[
\sigma_{\mathrm{noise}}
=
\operatorname{median}
\left\{
\sigma_i:i>\lfloor r_{\max}/2\rfloor
\right\}.
\]

The raw rank is

\[
k_{\mathrm{raw}}
=
\#\left\{
i:\sigma_i>5\sigma_{\mathrm{noise}}
\right\}.
\]

Rank EMA starts at full rank and is updated by

\[
\bar k_t
=
0.005k_{\mathrm{raw},t}
+0.995\bar k_{t-1},
\qquad
k_t=\operatorname{round}(\bar k_t).
\]

There is no warmup. The hard-truncated update is

\[
\Delta W_t
=
U_{:,1:k_t}V_{:,1:k_t}^\top.
\]

All directions after \(k_t\) receive zero update.

## 6. Evaluation Protocol

For each training K, evaluation uses the same K and the same number \(K-5\) of
Mperp tokens:

| evaluation set | signal composition |
|---|---|
| source | five random M1/M2 tokens, expected 4:1 |
| only M1 | five distinct M1 tokens |
| only M2 | five distinct M2 tokens |
| mixed | exactly four distinct M1 and one distinct M2 |

Each held-out set contains 5000 examples and is fixed throughout training.
The principal summary statistic is mean loss over steps 800-1000.

The only-M2 set is the main minority-feature diagnostic. It tests whether the
model learned M2 features even though the source training distribution contains
only one M2 token in expectation.

## 7. Difference From the Earlier Baseline PDF

The reference baseline PDF and Experiment 202 are not the same data model:

| component | earlier baseline | Experiment 202 |
|---|---|---|
| token-type sampling | every token iid | exactly 5 signal slots |
| M1/M2 counts | controlled by token probabilities | random only within the 5 signal slots |
| signal sparsity as K grows | can grow with K | fixed at \(\lVert z\rVert_0=5\) |
| teacher | normalized Gaussian columns | sparse binary columns |
| value activation | identity | ReLU |
| pooling | mean | sum |
| varying K | changes expected signal and noise counts | changes only Mperp count |

This distinction is essential: Experiment 202 is designed to isolate the
effect of increasing nuisance-token count while keeping the label-generating
signal sparse.
