"""Data models for kernelmps."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContourData:
    kernel_name: str
    contour: list[list[float]]  # [[x, y], ...] in subimage-local coords


@dataclass
class AxisData:
    kernel_name: str
    bottom: tuple[float, float]
    top: tuple[float, float]
    shape_label: str
    axis_length: float
    max_width: float
    area: float
    perimeter: float
    circularity: float


@dataclass
class YoloDetection:
    kernel_id: int
    box: tuple[int, int, int, int]  # [x1, y1, x2, y2] in full-image coords
    confidence: float = 0.0


@dataclass
class YoloData:
    image_name: str
    image_size: tuple[int, int]  # (width, height)
    accepted_boxes: dict[int, YoloDetection] = field(default_factory=dict)


@dataclass
class LoadResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)
    total_kernels: int = 0
