# Experiment 202: matched-K noise sweep

## Question

When the number of useful signal tokens is fixed, how does adding Mperp tokens
change the relative behavior of full Muon and hard-top-k Adaptive Muon?

## Data

- Ambient dimension: `D1=100`.
- Signal dictionary: 20 orthonormal columns, split into 10 M1 and 10 M2 features.
- Every sample contains exactly five distinct signal features.
- Each signal token independently belongs to M1 with probability 0.8 and M2
  with probability 0.2. Therefore the expected composition is 4 M1 and 1 M2,
  but the realized counts vary.
- The remaining `K-5` tokens are unit-norm columns from the orthogonal Mperp
  subspace.
- `K=[18,20,23,25,28,30]`, giving 13, 15, 18, 20, 23, and 25 Mperp tokens.
- The binary teacher has shape `[50,20]`; each feature activates five outputs.
- The target is `y = z @ W_true.T`, where `z` records feature presence.

## Network

```text
Q = X WQ
K_att = X WK
A = softmax(Q K_att^T / sqrt(d_qk))
X_tilde = A X
V = ReLU(X_tilde WV)
h = sum_tokens(V)
y_hat = h Wout^T
loss = 0.5 * mean_batch ||y_hat - y||_2^2
```

Shapes:

```text
WQ, WK, WV: [100,100]
Wout:        [50,100]
```

## Optimizers

Full Muon uses the momentum buffer `B_t=0.95 B_{t-1}+G_t` and update
`U V^T`, where `B_t=U diag(sigma) V^T`.

Adaptive WV+Wout keeps WQ and WK full rank. For WV and Wout it estimates raw
rank by thresholding singular values against five times the median of the
bottom half of the momentum-buffer spectrum. The selected rank is smoothed
with EMA alpha 0.005, then the update is hard truncated to `U[:,:k] V[:k,:]^T`.

## Evaluation

Training and evaluation use matched K. Four fixed held-out datasets are tracked
at every evaluation step:

- source: random M1/M2 assignment with expected 4:1;
- only M1: five distinct M1 features;
- only M2: five distinct M2 features;
- mixed: exactly four M1 and one M2.

All include exactly `K-5` Mperp tokens. Reported late loss is the mean over
steps 800-1000.

