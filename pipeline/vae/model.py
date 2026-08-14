"""VAE model definition (Encoder only — for pipeline encoding stage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn


@dataclass
class ProfileVAEConfig:
    input_dim: int = 100
    latent_dim: int = 5
    channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 5
    dropout: float = 0.1


class Encoder(nn.Module):
    def __init__(self, config: ProfileVAEConfig):
        super().__init__()
        k = config.kernel_size
        pad = k // 2

        layers = []
        in_ch = 1
        for out_ch in config.channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=2, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.flat_dim = config.channels[-1] * (config.input_dim // (2 ** len(config.channels)))
        self.fc_mu = nn.Linear(self.flat_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, config.latent_dim)

    def forward(self, x):
        h = self.backbone(x.unsqueeze(1))
        h = torch.flatten(h, start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)
