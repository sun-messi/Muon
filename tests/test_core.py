import unittest

import torch

from muon_sparse_attention.config import ExperimentConfig
from muon_sparse_attention.data import DataModel
from muon_sparse_attention.model import SparseAttentionRegressor
from muon_sparse_attention.optim import hard_topk_muon_update


class CoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            d_input=12,
            d_signal=6,
            n_m1_features=3,
            n_m2_features=3,
            d_qk=8,
            d_value=7,
            d_output=5,
            sequence_length=7,
            signal_tokens=3,
            eval_size=16,
            device="cpu",
        )
        self.generator = torch.Generator().manual_seed(0)
        self.data = DataModel.create(
            self.config, self.generator, torch.device("cpu")
        )

    def test_source_has_exact_signal_and_noise_counts(self) -> None:
        batch = self.data.sample_source(64, self.generator)
        self.assertTrue(torch.all((batch.groups != 2).sum(dim=1) == 3))
        self.assertTrue(torch.all((batch.groups == 2).sum(dim=1) == 4))
        self.assertTrue(torch.all(batch.z.sum(dim=1) == 3))

    def test_only_m2_uses_only_m2_features(self) -> None:
        batch = self.data.sample_fixed_eval(32, "only_m2", self.generator)
        self.assertTrue(torch.all(batch.z[:, :3] == 0))
        self.assertTrue(torch.all(batch.z[:, 3:].sum(dim=1) == 3))

    def test_model_shapes(self) -> None:
        model = SparseAttentionRegressor(
            self.config, self.generator, torch.device("cpu")
        )
        batch = self.data.sample_source(4, self.generator)
        prediction, attention, X_tilde = model(batch.X)
        self.assertEqual(prediction.shape, (4, 5))
        self.assertEqual(attention.shape, (4, 7, 7))
        self.assertEqual(X_tilde.shape, (4, 7, 12))
        self.assertTrue(
            torch.allclose(
                attention.sum(dim=-1),
                torch.ones(4, 7, dtype=attention.dtype),
            )
        )

    def test_hard_topk_update_has_requested_rank(self) -> None:
        matrix = torch.randn(8, 6, generator=self.generator, dtype=torch.float64)
        update, _ = hard_topk_muon_update(matrix, rank=3)
        rank = int(torch.linalg.matrix_rank(update, tol=1e-6).item())
        self.assertEqual(rank, 3)


if __name__ == "__main__":
    unittest.main()

