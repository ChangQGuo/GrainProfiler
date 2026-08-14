"""
vae/model.py — 1D-CNN β-VAE for maize kernel full-width profiles.

Architecture
  Encoder:  100 -> Conv1d(s=2) -> 50 -> Conv1d(s=2) -> 25 -> Flatten(1600) -> FC: μ(5)+logσ²(5) -> z(5)
  Latent:   5-dim vector z sampled from 𝒩(μ,σ²) via reparameterisation trick.
  Decoder:  z(5) → FC(1600) → 25 → ConvTranspose1d(s=2) → 50 → ConvTranspose1d(s=2) → 100
  Loss:  MSE(recon, x) + β · KL( 𝒩(μ,σ²) ‖ 𝒩(0,I) )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Decoder(nn.Module):
    def __init__(self, config: ProfileVAEConfig):
        super().__init__()
        k = config.kernel_size
        pad = k // 2
        self.channels = config.channels
        self.seq_len = config.input_dim // (2 ** len(config.channels))  # 25

        self.fc_in = nn.Linear(config.latent_dim,
                               config.channels[-1] * self.seq_len)  # 5 → 1600

        layers = [
            nn.ConvTranspose1d(64, 32, kernel_size=k, stride=2, padding=pad, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 1, kernel_size=k, stride=2, padding=pad, output_padding=1),
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        h = self.fc_in(z)
        h = h.view(h.shape[0], self.channels[-1], self.seq_len)
        return self.decoder(h).squeeze(1)


class ProfileVAE(nn.Module):
    def __init__(self, config: ProfileVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta: float = 0.001):
    """β-VAE loss = MSE(recon, x) + β · KL(𝒩(μ,σ²) ‖ 𝒩(0,I))."""
    recon_loss = F.mse_loss(recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + float(beta) * kl_loss, recon_loss, kl_loss
