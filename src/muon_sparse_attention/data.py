from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig


@dataclass
class Batch:
    X: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    groups: torch.Tensor


@dataclass
class DataModel:
    config: ExperimentConfig
    dictionary: torch.Tensor
    dictionary_perp: torch.Tensor
    teacher: torch.Tensor
    device: torch.device
    dtype: torch.dtype

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> "DataModel":
        basis_seed = torch.randn(
            config.d_input,
            config.d_input,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        basis, _ = torch.linalg.qr(basis_seed, mode="complete")
        dictionary = basis[:, : config.d_signal]
        dictionary_perp = basis[:, config.d_signal :]

        active_rows = max(
            1, min(config.d_output, round(config.d_output * config.binary_teacher_density))
        )
        teacher = torch.zeros(
            config.d_output, config.d_signal, device=device, dtype=dtype
        )
        for feature in range(config.d_signal):
            rows = torch.randperm(
                config.d_output, generator=generator, device=device
            )[:active_rows]
            teacher[rows, feature] = 1.0
        return cls(config, dictionary, dictionary_perp, teacher, device, dtype)

    def sample_source(
        self,
        size: int,
        generator: torch.Generator,
    ) -> Batch:
        """Five distinct signal tokens with random M1/M2 groups, plus Mperp."""
        cfg = self.config
        group_draw = torch.rand(
            size,
            cfg.signal_tokens,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        signal_groups = (group_draw >= cfg.m1_group_probability).long()
        return self._sample_with_groups(signal_groups, generator)

    def sample_fixed_eval(
        self,
        size: int,
        kind: str,
        generator: torch.Generator,
    ) -> Batch:
        cfg = self.config
        if kind == "only_m1":
            signal_groups = torch.zeros(
                size, cfg.signal_tokens, device=self.device, dtype=torch.long
            )
        elif kind == "only_m2":
            signal_groups = torch.ones(
                size, cfg.signal_tokens, device=self.device, dtype=torch.long
            )
        elif kind == "mixed":
            n_m1 = round(cfg.signal_tokens * cfg.m1_group_probability)
            n_m1 = min(cfg.signal_tokens, max(0, n_m1))
            row = torch.cat(
                [
                    torch.zeros(n_m1, device=self.device, dtype=torch.long),
                    torch.ones(
                        cfg.signal_tokens - n_m1,
                        device=self.device,
                        dtype=torch.long,
                    ),
                ]
            )
            signal_groups = row.expand(size, -1)
        else:
            raise ValueError(f"unknown eval kind: {kind}")
        return self._sample_with_groups(signal_groups, generator)

    def _sample_with_groups(
        self,
        signal_groups: torch.Tensor,
        generator: torch.Generator,
    ) -> Batch:
        cfg = self.config
        size = signal_groups.shape[0]
        K = cfg.sequence_length
        n_signal = cfg.signal_tokens

        signal_features = torch.empty(
            size, n_signal, device=self.device, dtype=torch.long
        )
        for row in range(size):
            m1_slots = torch.nonzero(signal_groups[row] == 0, as_tuple=False).flatten()
            m2_slots = torch.nonzero(signal_groups[row] == 1, as_tuple=False).flatten()
            if m1_slots.numel():
                selected = torch.randperm(
                    cfg.n_m1_features, generator=generator, device=self.device
                )[: m1_slots.numel()]
                signal_features[row, m1_slots] = selected
            if m2_slots.numel():
                selected = torch.randperm(
                    cfg.n_m2_features, generator=generator, device=self.device
                )[: m2_slots.numel()]
                signal_features[row, m2_slots] = cfg.n_m1_features + selected

        noise_features = torch.randint(
            self.dictionary_perp.shape[1],
            (size, cfg.noise_tokens),
            generator=generator,
            device=self.device,
        )
        X_signal = self.dictionary.T[signal_features] * cfg.signal_scale
        X_noise = self.dictionary_perp.T[noise_features]
        X = torch.cat([X_signal, X_noise], dim=1)

        groups = torch.cat(
            [
                signal_groups,
                torch.full(
                    (size, cfg.noise_tokens),
                    2,
                    device=self.device,
                    dtype=torch.long,
                ),
            ],
            dim=1,
        )
        permutations = torch.rand(
            size,
            K,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        ).argsort(dim=1)
        X = X.gather(1, permutations.unsqueeze(-1).expand(-1, -1, cfg.d_input))
        groups = groups.gather(1, permutations)

        z = torch.zeros(size, cfg.d_signal, device=self.device, dtype=self.dtype)
        rows = torch.arange(size, device=self.device).repeat_interleave(n_signal)
        z[rows, signal_features.reshape(-1)] = 1.0
        y = z @ self.teacher.T
        return Batch(X=X, y=y, z=z, groups=groups)

