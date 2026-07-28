"""Adaptive Muon experiments for sparse attention regression."""

from .config import ExperimentConfig
from .model import SparseAttentionRegressor
from .optim import AdaptiveMuon

__all__ = ["AdaptiveMuon", "ExperimentConfig", "SparseAttentionRegressor"]

