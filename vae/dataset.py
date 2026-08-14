"""
vae/dataset.py — Load rep_width_profiles.txt for VAE training.

Input format (whitespace-separated, no header):
    plant_id  w_1  w_2  ...  w_100

Splits: train 70% / val 20% / test 10%  (fixed seed for reproducibility).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WidthProfileDataset(Dataset):
    """PyTorch Dataset over rows of rep_width_profiles.txt."""

    def __init__(self, profile_path: str | Path):
        self.profile_path = Path(profile_path)
        if not self.profile_path.is_file():
            raise FileNotFoundError(f'Profile file not found: {self.profile_path}')

        raw = np.loadtxt(self.profile_path, dtype=str, ndmin=2)
        if raw.shape[1] < 2:
            raise ValueError(f'Expected >= 2 columns, got {raw.shape[1]}')

        self.labels = raw[:, 0].tolist()
        self.data = raw[:, 1:].astype(np.float32)
        self.input_dim = int(self.data.shape[1])

        if self.input_dim == 0:
            raise ValueError('No numeric width columns found.')

        # Per-column standardisation
        self.col_mean = self.data.mean(axis=0, keepdims=True)
        self.col_std = self.data.std(axis=0, keepdims=True)
        self.col_std[self.col_std < 1e-8] = 1.0
        self.data = (self.data - self.col_mean) / self.col_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'profile': torch.from_numpy(self.data[index]),
            'label': self.labels[index],
        }

    @property
    def num_samples(self) -> int:
        return len(self.labels)

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        return array * self.col_std.squeeze(0) + self.col_mean.squeeze(0)


class _SubsetDataset(Dataset):
    def __init__(self, parent, indices):
        self.parent = parent
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.parent[self.indices[idx]]


def split_dataset(dataset: WidthProfileDataset, seed: int = 42
                  ) -> Tuple[WidthProfileDataset, WidthProfileDataset, WidthProfileDataset]:
    """Return (train, val, test) at 70:20:10 ratio, fixed-seed shuffle."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * 0.10)))
    n_val = max(1, int(round(n * 0.20)))
    n_train = n - n_val - n_test
    return (
        _SubsetDataset(dataset, indices[:n_train].tolist()),
        _SubsetDataset(dataset, indices[n_train:n_train + n_val].tolist()),
        _SubsetDataset(dataset, indices[n_train + n_val:].tolist()),
    )
