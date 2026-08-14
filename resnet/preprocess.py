"""Image and mask preprocessing for kernel angle regression."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    return labels == largest_label


def read_main_mask(mask_path: Optional[Path], image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask_path is None or not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    height, width = image_shape
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return largest_component(mask > 127)


def fill_color(background: str) -> np.ndarray:
    if background == "gray":
        return np.array([127, 127, 127], dtype=np.uint8)
    if background == "white":
        return np.array([255, 255, 255], dtype=np.uint8)
    return np.array([0, 0, 0], dtype=np.uint8)


def square_pad(image: np.ndarray, color: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = max(height, width)
    top = (side - height) // 2
    bottom = side - height - top
    left = (side - width) // 2
    right = side - width - left
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=tuple(int(v) for v in color.tolist()),
    )


def resize_long_side_and_pad(image: np.ndarray, size: int, color: np.ndarray) -> np.ndarray:
    """Keep aspect ratio, resize long side to size, then pad short side."""
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    scale = float(size) / float(max(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    output = np.empty((size, size, 3), dtype=np.uint8)
    output[:, :] = color.reshape(1, 1, 3)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    output[top : top + new_height, left : left + new_width] = resized
    return output


def crop_by_mask(image: np.ndarray, mask: np.ndarray, margin_ratio: float, color: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    height, width = image.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    margin = int(round(max(box_w, box_h) * margin_ratio))
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(width - 1, x2 + margin)
    y2 = min(height - 1, y2 + margin)

    masked = image.copy()
    masked[~mask] = color
    return masked[y1 : y2 + 1, x1 : x2 + 1]


def preprocess_rgb(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    imgsz: int,
    mode: str = "crop",
    margin_ratio: float = 0.08,
    background: str = "black",
) -> np.ndarray:
    """Return RGB uint8 image after optional mask cleanup and aspect-preserving resize."""
    color = fill_color(background)
    output = image.copy()

    if mask is not None and mode in {"mask", "crop"}:
        if mode == "mask":
            output[~mask] = color
        elif mode == "crop":
            output = crop_by_mask(output, mask, margin_ratio=margin_ratio, color=color)
    elif mode == "square":
        output = square_pad(output, color)

    if mode == "none":
        return cv2.resize(output, (int(imgsz), int(imgsz)), interpolation=cv2.INTER_AREA)
    return resize_long_side_and_pad(output, int(imgsz), color)


def normalize_to_tensor(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    import torch

    image_f = image.astype(np.float32) / 255.0
    image_f = (image_f - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    image_f = np.transpose(image_f, (2, 0, 1))
    return torch.from_numpy(image_f).float()


def normalize_to_tensor_with_mask(image: np.ndarray, mask_bool: np.ndarray,
                                   mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Return (4, H, W) tensor: 3 RGB channels (ImageNet-norm) + binary mask."""
    import torch

    image_f = image.astype(np.float32) / 255.0
    image_f = (image_f - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    image_f = np.transpose(image_f, (2, 0, 1))

    mask_channel = mask_bool.astype(np.float32)
    mask_channel = np.expand_dims(mask_channel, axis=0)  # (1, H, W)

    combined = np.concatenate([image_f, mask_channel], axis=0)  # (4, H, W)
    return torch.from_numpy(combined).float()


def color_jitter_rgb(image: np.ndarray, brightness: float, contrast: float, rng) -> np.ndarray:
    output = image.astype(np.float32)
    if brightness > 0:
        factor = 1.0 + rng.uniform(-brightness, brightness)
        output *= factor
    if contrast > 0:
        factor = 1.0 + rng.uniform(-contrast, contrast)
        mean = output.mean(axis=(0, 1), keepdims=True)
        output = (output - mean) * factor + mean
    return np.clip(output, 0, 255).astype(np.uint8)
