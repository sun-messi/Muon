# Code map from the research workspace

The original experiment evolved through a long numbered-script chain. The
standalone repository keeps the behavior but removes that inheritance:

| Original responsibility | Standalone module |
|---|---|
| Environment-variable configuration in `config.py` | `config.py` dataclass |
| Data generation and training loop in `run.py` | `data.py` and `train.py` |
| Attention/readout network in `run.py` | `model.py` |
| Muon, rank estimation, rank EMA, hard truncation in `run.py` | `optim.py` |
| Generic matched-K launch/summary logic in numbered 196/198/199/201 scripts | `sweep.py` and `plot.py` |
| Focused K18/20/23/25/28/30 settings and heatmap in numbered 202 | constants in `sweep.py` and `plot.py` |

The exported code intentionally excludes:

- unrelated historical sweeps;
- Gaussian and token-Gaussian noise variants;
- spectral-tail updates;
- fixed-rank controls;
- optimizers other than Muon and Adaptive WV+Wout;
- large per-run metric archives.

Selected experiment-202 CSV and PNG artifacts are retained in
`results/experiment_202/`.
