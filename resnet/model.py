"""Small ResNet-style angle regression model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn as nn


@dataclass
class ResNetAngleConfig:
    stem_channels: int = 32
    channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    blocks: tuple[int, int, int, int] = (2, 2, 3, 2)
    dropout: float = 0.20
    in_channels: int = 3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.act(out)
        return out


class ResNetAngleRegressor(nn.Module):
    """RGB image -> cos(theta), sin(theta)."""

    def __init__(self, config: ResNetAngleConfig | None = None):
        super().__init__()
        self.config = config or ResNetAngleConfig()
        stem_channels = self.config.stem_channels

        self.stem = nn.Sequential(
            nn.Conv2d(self.config.in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        in_channels = stem_channels
        stages: List[nn.Module] = []
        for stage_idx, (out_channels, n_blocks) in enumerate(zip(self.config.channels, self.config.blocks)):
            stride = 1 if stage_idx == 0 else 2
            stage, in_channels = self._make_stage(in_channels, out_channels, n_blocks, stride)
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        final_channels = self.config.channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(final_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(128, 2),
        )

    @staticmethod
    def _make_stage(in_channels: int, out_channels: int, n_blocks: int, stride: int) -> tuple[nn.Sequential, int]:
        layers: List[nn.Module] = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers), out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = self.head(x)
        return x


def build_resnet_angle_model(
    stem_channels: int = 32,
    channels: Iterable[int] = (64, 128, 256, 512),
    blocks: Iterable[int] = (2, 2, 3, 2),
    dropout: float = 0.20,
    in_channels: int = 3,
) -> ResNetAngleRegressor:
    config = ResNetAngleConfig(
        stem_channels=int(stem_channels),
        channels=tuple(int(v) for v in channels),
        blocks=tuple(int(v) for v in blocks),
        dropout=float(dropout),
        in_channels=int(in_channels),
    )
    return ResNetAngleRegressor(config)
