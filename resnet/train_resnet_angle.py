"""
Train a custom ResNet-style CNN for directed kernel-axis angle regression.

Input : one RGB kernel crop.
Output: cos(theta), sin(theta).

Defaults:
  - train/val/test split = 8:1:1
  - epochs = 300
  - early stopping patience = 30
  - AMP is not used
  - mask-crop preprocessing is enabled to reduce 640x640 background noise
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from model import build_resnet_angle_model
from preprocess import IMAGENET_MEAN, IMAGENET_STD, color_jitter_rgb, normalize_to_tensor, normalize_to_tensor_with_mask, preprocess_rgb, read_main_mask, read_rgb

# Global plot style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
})
BLUE_LIGHT = "#4A90D9"
BLUE_DARK = "#2166AC"
BLUE_PALE = "#A8C8E8"
ORANGE_ACCENT = "#E8963A"


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "image_data"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
DEFAULT_ARGS = {
    "csv": str(DEFAULT_DATA_DIR / "angle_labels_directed.csv"),
    "images": str(DEFAULT_DATA_DIR / "subimages"),
    "masks": "",
    "project": str(SCRIPT_DIR / "runs" / "angle"),
    "name": "train",
    "exist_ok": False,
    "epochs": 300,
    "patience": 30,
    "batch_size": 32,
    "imgsz": 256,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "device": "auto",
    "seed": 42,
    "status": "ok",
    "preprocess": "square",
    "margin": 0.25,
    "background": "black",
    "brightness": 0.04,
    "contrast": 0.04,
    "noise_prob": 0.30,
    "noise_std": 8.0,
    "hflip_prob": 0.0,
    "vflip_prob": 0.0,
    "dropout": 0.20,
    "stem_channels": 32,
    "norm_loss_weight": 0.0,
    "save_samples": 16,
    "in_channels": 3,
}


@dataclass
class LabelRecord:
    image_label: str
    image_path: str
    mask_path: str
    theta_deg: float
    theta_rad: float
    cos_theta: float
    sin_theta: float


def resolve_config_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists() or path.is_absolute():
        return path
    candidate = SCRIPT_DIR / path
    if candidate.exists():
        return candidate
    return path


def config_get(section: Dict, *names, default=None):
    for name in names:
        if name in section and section[name] is not None:
            return section[name]
    return default


def load_config_defaults(config_path: Path) -> Dict:
    defaults = dict(DEFAULT_ARGS)
    if not config_path.exists():
        return defaults

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    data = config.get("data", {}) or {}
    output = config.get("output", {}) or {}
    training = config.get("training", {}) or {}
    preprocess = config.get("preprocess", {}) or {}
    model = config.get("model", {}) or {}

    updates = {
        "csv": config_get(data, "labels_csv", "csv", default=None),
        "images": config_get(data, "images_dir", "images", default=None),
        "masks": config_get(data, "masks_dir", "masks", default=None),
        "status": config_get(data, "status", default=None),
        "project": config_get(output, "project", default=None),
        "name": config_get(output, "name", default=None),
        "exist_ok": config_get(output, "exist_ok", default=None),
        "epochs": config_get(training, "epochs", default=None),
        "patience": config_get(training, "patience", "early_stop_patience", default=None),
        "batch_size": config_get(training, "batch_size", default=None),
        "lr": config_get(training, "lr", "learning_rate", default=None),
        "weight_decay": config_get(training, "weight_decay", default=None),
        "num_workers": config_get(training, "num_workers", default=None),
        "device": config_get(training, "device", default=None),
        "seed": config_get(training, "seed", default=None),
        "save_samples": config_get(training, "save_samples", default=None),
        "preprocess": config_get(preprocess, "mode", "preprocess", default=None),
        "imgsz": config_get(preprocess, "image_size", "imgsz", default=None),
        "margin": config_get(preprocess, "margin_ratio", "margin", default=None),
        "background": config_get(preprocess, "background", default=None),
        "brightness": config_get(preprocess, "brightness", default=None),
        "contrast": config_get(preprocess, "contrast", default=None),
        "noise_prob": config_get(preprocess, "noise_prob", default=None),
        "noise_std": config_get(preprocess, "noise_std", default=None),
        "hflip_prob": config_get(preprocess, "hflip_prob", default=None),
        "vflip_prob": config_get(preprocess, "vflip_prob", default=None),
        "dropout": config_get(model, "dropout", default=None),
        "stem_channels": config_get(model, "stem_channels", default=None),
        "norm_loss_weight": config_get(model, "norm_loss_weight", default=None),
        "in_channels": config_get(model, "in_channels", default=None),
    }
    for key, value in updates.items():
        if value is not None:
            defaults[key] = value
    return defaults


def resolve_path_arg(value: str, config_dir: Path) -> str:
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def parse_args() -> argparse.Namespace:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG_PATH), help="YAML config path.")
    probe_args, _ = config_probe.parse_known_args()
    config_path = resolve_config_path(probe_args.config)
    defaults = load_config_defaults(config_path)

    parser = argparse.ArgumentParser(description="Train a custom ResNet-style angle regressor.")
    parser.add_argument("config", nargs="?", default=str(config_path), help="YAML config path. Example: python train_resnet_angle.py config.yaml")
    parser.add_argument("--csv", default=defaults["csv"], help="Angle-label CSV.")
    parser.add_argument("--images", default=defaults["images"], help="RGB kernel crop directory.")
    parser.add_argument("--masks", default=defaults["masks"], help="Binary mask directory.")
    parser.add_argument("--project", default=defaults["project"], help="Output project directory.")
    parser.add_argument("--name", default=defaults["name"], help="Run name. Existing names are incremented like YOLO.")
    parser.add_argument("--exist-ok", action="store_true", default=bool(defaults["exist_ok"]), help="Allow writing into an existing run directory.")
    parser.add_argument("--epochs", type=int, default=int(defaults["epochs"]), help="Maximum epochs.")
    parser.add_argument("--patience", type=int, default=int(defaults["patience"]), help="Early-stopping patience on validation loss.")
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]), help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=int(defaults["imgsz"]), help="Model input size after preprocessing.")
    parser.add_argument("--lr", type=float, default=float(defaults["lr"]), help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=float(defaults["weight_decay"]), help="AdamW weight decay.")
    parser.add_argument("--num-workers", type=int, default=int(defaults["num_workers"]), help="DataLoader workers.")
    parser.add_argument("--device", default=defaults["device"], help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]), help="Random seed.")
    parser.add_argument("--status", default=defaults["status"], help="CSV status value used for training.")
    parser.add_argument("--preprocess", default=defaults["preprocess"], choices=["crop", "mask", "square", "none"], help="Input preprocessing mode.")
    parser.add_argument("--margin", type=float, default=float(defaults["margin"]), help="Mask crop margin ratio.")
    parser.add_argument("--background", default=defaults["background"], choices=["black", "gray", "white"], help="Background fill color for masked pixels.")
    parser.add_argument("--brightness", type=float, default=float(defaults["brightness"]), help="Training brightness jitter.")
    parser.add_argument("--contrast", type=float, default=float(defaults["contrast"]), help="Training contrast jitter.")
    parser.add_argument("--noise-prob", type=float, default=float(defaults["noise_prob"]), help="Probability of adding Gaussian noise to a train sample.")
    parser.add_argument("--noise-std", type=float, default=float(defaults["noise_std"]), help="Gaussian noise standard deviation in 0-255 pixel units.")
    parser.add_argument("--hflip-prob", type=float, default=float(defaults["hflip_prob"]), help="Random horizontal flip probability; target is adjusted.")
    parser.add_argument("--vflip-prob", type=float, default=float(defaults["vflip_prob"]), help="Random vertical flip probability; target is adjusted.")
    parser.add_argument("--dropout", type=float, default=float(defaults["dropout"]), help="MLP head dropout.")
    parser.add_argument("--stem-channels", type=int, default=int(defaults["stem_channels"]), help="Stem channels.")
    parser.add_argument("--norm-loss-weight", type=float, default=float(defaults["norm_loss_weight"]), help="Small penalty that keeps output vector norm near 1.")
    parser.add_argument("--save-samples", type=int, default=int(defaults["save_samples"]), help="Number of test examples in prediction plot.")
    parser.add_argument("--in-channels", type=int, default=int(defaults["in_channels"]), choices=[3, 4], help="Input channels: 3=RGB, 4=RGB+Mask.")
    args = parser.parse_args()

    args.config = str(resolve_config_path(args.config).resolve())
    config_dir = Path(args.config).parent
    args.csv = resolve_path_arg(args.csv, config_dir)
    args.images = resolve_path_arg(args.images, config_dir)
    args.masks = resolve_path_arg(args.masks, config_dir)
    args.project = resolve_path_arg(args.project, config_dir)
    return args


def increment_path(base_dir: Path, name: str, exist_ok: bool = False) -> Path:
    path = base_dir / name
    if exist_ok or not path.exists():
        return path
    for idx in range(2, 10000):
        candidate = base_dir / f"{name}{idx}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create an incremented run directory under {base_dir}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def load_records(csv_path: Path, image_dir: Path, mask_dir: Optional[Path], status: str = "ok", require_masks: bool = False) -> List[LabelRecord]:
    records: List[LabelRecord] = []
    missing_images = 0
    missing_masks = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"status", "cos_theta", "sin_theta"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            if str(row.get("status", "")).strip().lower() != status.lower():
                continue

            image_label = str(row.get("image_label") or Path(str(row.get("image_path", ""))).name).strip()
            if not image_label:
                continue
            image_path = image_dir / image_label
            if not image_path.exists():
                missing_images += 1
                continue

            mask_path = mask_dir / image_label if mask_dir is not None and require_masks else None
            if require_masks and (mask_path is None or not mask_path.exists()):
                missing_masks += 1
                mask_path_str = ""
            elif mask_path is not None:
                mask_path_str = str(mask_path)
            else:
                mask_path_str = ""

            cos_t = safe_float(row.get("cos_theta"))
            sin_t = safe_float(row.get("sin_theta"))
            norm = math.hypot(cos_t, sin_t)
            if norm < 1e-6:
                continue
            cos_t /= norm
            sin_t /= norm
            theta_rad = math.atan2(sin_t, cos_t) % (2.0 * math.pi)
            records.append(
                LabelRecord(
                    image_label=image_label,
                    image_path=str(image_path),
                    mask_path=mask_path_str,
                    theta_deg=math.degrees(theta_rad),
                    theta_rad=theta_rad,
                    cos_theta=cos_t,
                    sin_theta=sin_t,
                )
            )

    if missing_images:
        print(f"WARNING: skipped {missing_images} rows because images were missing.")
    if require_masks and missing_masks:
        print(f"WARNING: {missing_masks} records have no matching mask; those samples fall back to raw image preprocessing.")
    if not records:
        raise RuntimeError(f"No usable rows found in {csv_path} with status={status!r}")
    return records


def split_records(records: Sequence[LabelRecord], seed: int) -> Dict[str, List[LabelRecord]]:
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    n_total = len(records)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    if n_train + n_val >= n_total:
        n_val = max(1, n_total - n_train - 1)
    return {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }


def write_split_csv(path: Path, split_name: str, records: Sequence[LabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "image_label", "theta_deg", "theta_rad", "cos_theta", "sin_theta", "image_path", "mask_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            row["split"] = split_name
            writer.writerow(row)


def add_gaussian_noise_tensor(image_tensor: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Add pixel-space Gaussian noise to an ImageNet-normalized CHW tensor.

    For 4-channel input (RGB+Mask), noise is only added to RGB channels (0:3);
    the mask channel is left unchanged.
    """
    if noise_std <= 0:
        return image_tensor

    device = image_tensor.device
    dtype = image_tensor.dtype
    n_channels = image_tensor.size(0)

    if n_channels == 4:
        # RGB channels only
        rgb = image_tensor[:3]
        mask_ch = image_tensor[3:4]
        mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=dtype, device=device).view(3, 1, 1)
        rgb_01 = rgb * std + mean
        noise = torch.randn_like(rgb_01) * (float(noise_std) / 255.0)
        rgb_01 = (rgb_01 + noise).clamp(0.0, 1.0)
        rgb = (rgb_01 - mean) / std
        return torch.cat([rgb, mask_ch], dim=0)
    else:
        mean = torch.tensor(IMAGENET_MEAN, dtype=dtype, device=device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=dtype, device=device).view(3, 1, 1)
        image_01 = image_tensor * std + mean
        noise = torch.randn_like(image_01) * (float(noise_std) / 255.0)
        image_01 = (image_01 + noise).clamp(0.0, 1.0)
        return (image_01 - mean) / std


class KernelAngleDataset(Dataset):
    def __init__(self, records: Sequence[LabelRecord], args: argparse.Namespace, train: bool):
        self.records = list(records)
        self.args = args
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = read_rgb(Path(rec.image_path))
        mask = read_main_mask(Path(rec.mask_path) if rec.mask_path else None, image.shape[:2])
        use_mask_channel = self.args.in_channels == 4

        image = preprocess_rgb(
            image, mask,
            imgsz=self.args.imgsz,
            mode=self.args.preprocess,
            margin_ratio=self.args.margin,
            background=self.args.background,
        )

        if use_mask_channel:
            # Mask channel: directly resize the original binary mask.
            # NOTE: when preprocess mode is "square" or "crop", the mask
            # channel will NOT be perfectly aligned with the padded/cropped
            # RGB image.  Use in_channels=4 only with mode="none".
            mask_for_channel = np.zeros((self.args.imgsz, self.args.imgsz), dtype=bool)
            if mask is not None:
                mask_resized = cv2.resize(
                    mask.astype(np.uint8) * 255,
                    (self.args.imgsz, self.args.imgsz),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_for_channel = mask_resized > 127

        cos_t = float(rec.cos_theta)
        sin_t = float(rec.sin_theta)
        if self.train:
            rng = random
            image = color_jitter_rgb(image, brightness=self.args.brightness, contrast=self.args.contrast, rng=rng)
            if rng.random() < self.args.hflip_prob:
                image = np.ascontiguousarray(image[:, ::-1, :])
                if use_mask_channel:
                    mask_for_channel = mask_for_channel[:, ::-1]
                cos_t = -cos_t
            if rng.random() < self.args.vflip_prob:
                image = np.ascontiguousarray(image[::-1, :, :])
                if use_mask_channel:
                    mask_for_channel = mask_for_channel[::-1, :]
                sin_t = -sin_t

        if use_mask_channel:
            image_tensor = normalize_to_tensor_with_mask(image, mask_for_channel)
        else:
            image_tensor = normalize_to_tensor(image)

        if self.train and random.random() < self.args.noise_prob:
            image_tensor = add_gaussian_noise_tensor(image_tensor, self.args.noise_std)

        target = torch.tensor([cos_t, sin_t], dtype=torch.float32)
        return image_tensor, target, index


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=1, eps=1e-6)


def direction_loss(pred: torch.Tensor, target: torch.Tensor, norm_loss_weight: float = 0.0) -> torch.Tensor:
    pred_unit = normalize_vectors(pred)
    target_unit = normalize_vectors(target)
    cosine = (pred_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    loss = (1.0 - cosine).mean()
    if norm_loss_weight > 0:
        loss = loss + norm_loss_weight * (pred.norm(dim=1) - 1.0).pow(2).mean()
    return loss


def angle_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_unit = normalize_vectors(pred.detach())
    target_unit = normalize_vectors(target.detach())
    cosine = (pred_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    pred_angle = torch.atan2(pred_unit[:, 1], pred_unit[:, 0])
    target_angle = torch.atan2(target_unit[:, 1], target_unit[:, 0])
    diff = torch.remainder(pred_angle - target_angle + math.pi, 2.0 * math.pi) - math.pi
    abs_error_deg = diff.abs() * 180.0 / math.pi
    return {
        "mae_deg": float(abs_error_deg.mean().item()),
        "angle_p50": float(torch.quantile(abs_error_deg, 0.50).item()),
        "angle_p90": float(torch.quantile(abs_error_deg, 0.90).item()),
        "mean_cosine": float(cosine.mean().item()),
    }


def run_one_epoch(model, loader, optimizer, device, train: bool, norm_loss_weight: float) -> Dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            preds = model(images)
            loss = direction_loss(preds, targets, norm_loss_weight=norm_loss_weight)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(images)
                loss = direction_loss(preds, targets, norm_loss_weight=norm_loss_weight)

        batch_size = int(images.size(0))
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = angle_metrics(preds_cat, targets_cat)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_mae_deg: float, args: argparse.Namespace, split_sizes: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_mae_deg": float(best_val_mae_deg),
            "early_stop_metric": "val_mae_deg",
            "args": vars(args),
            "model_config": {
                "stem_channels": int(args.stem_channels),
                "channels": [64, 128, 256, 512],
                "blocks": [2, 2, 3, 2],
                "dropout": float(args.dropout),
                "in_channels": int(args.in_channels),
            },
            "imgsz": int(args.imgsz),
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "split_sizes": split_sizes,
        },
        path,
    )


def append_results_csv(path: Path, row: Dict[str, float]) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_mae_deg",
        "val_mae_deg",
        "train_angle_p50",
        "val_angle_p50",
        "train_angle_p90",
        "val_angle_p90",
        "train_mean_cosine",
        "val_mean_cosine",
        "lr",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_results(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    columns: Dict[str, List[float]] = {}
    for key in rows[0].keys() if rows else []:
        columns[key] = [safe_float(row[key]) for row in rows]
    return columns


def smooth_curve(values: Sequence[float], weight: float = 0.72) -> List[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for value in values[1:]:
        smoothed.append(smoothed[-1] * weight + float(value) * (1.0 - weight))
    return smoothed


def plot_metric(ax, epochs: Sequence[float], values: Sequence[float], label: str, color=None, linestyle: str = "-") -> None:
    used_color = color or BLUE_LIGHT
    ax.plot(epochs, values, label="_nolegend_", color=used_color, linestyle=linestyle, alpha=0.22, linewidth=1.0)
    ax.plot(
        epochs,
        smooth_curve(values),
        label=label,
        color=used_color,
        linestyle=linestyle,
        linewidth=2.0,
    )


def plot_results(results_csv: Path, out_path: Path) -> None:
    data = load_results(results_csv)
    if not data:
        return
    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    plot_metric(axes[0], epochs, data["train_loss"], label="train")
    plot_metric(axes[0], epochs, data["val_loss"], label="val")
    axes[0].set_title("Direction loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    plot_metric(axes[1], epochs, data["train_mae_deg"], label="train")
    plot_metric(axes[1], epochs, data["val_mae_deg"], label="val")
    if "val_angle_p50" in data:
        plot_metric(axes[1], epochs, data["val_angle_p50"], label="val p50", linestyle="--")
    if "val_angle_p90" in data:
        plot_metric(axes[1], epochs, data["val_angle_p90"], label="val p90", linestyle=":")
    if data.get("val_mae_deg"):
        best_idx = int(np.argmin(np.asarray(data["val_mae_deg"], dtype=float)))
        axes[1].axvline(epochs[best_idx], color="0.35", linestyle=":", linewidth=1.0)
        axes[1].text(
            epochs[best_idx],
            data["val_mae_deg"][best_idx],
            f" best {data['val_mae_deg'][best_idx]:.2f}",
            fontsize=8,
            color="0.25",
            va="bottom",
        )
    axes[1].set_title("Angle MAE (deg)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plot_metric(axes[2], epochs, data["train_mean_cosine"], label="train")
    plot_metric(axes[2], epochs, data["val_mean_cosine"], label="val")
    axes[2].set_title("Mean cosine")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylim(0.75, 1.005)
    axes[2].legend()

    axes[3].plot(epochs, data["lr"], label="lr")
    axes[3].set_title("Learning rate")
    axes[3].set_xlabel("epoch")
    axes[3].legend()

    for ax in axes:
        ax.grid(True, alpha=0.22, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def draw_angle_inset(ax, target: np.ndarray, pred: np.ndarray, error_deg: float) -> None:
    """Draw GT/pred direction arrows in a fixed lower-left inset."""
    inset = ax.inset_axes([0.035, 0.035, 0.34, 0.34])
    inset.set_facecolor((0.0, 0.0, 0.0, 0.72))
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)
    inset.set_xlim(-1.0, 1.0)
    inset.set_ylim(-1.0, 1.0)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])

    def arrow(vector: np.ndarray, color: str) -> None:
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return
        vector = vector / norm
        # Labels use image coordinates: +y points downward. Matplotlib's inset
        # uses +y upward, so flip y only for display.
        display_vector = np.array([vector[0], -vector[1]], dtype=float)
        inset.arrow(
            0.0,
            0.0,
            display_vector[0] * 0.72,
            display_vector[1] * 0.72,
            color=color,
            width=0.035,
            head_width=0.16,
            length_includes_head=True,
            alpha=0.95,
        )

    arrow(target, "lime")
    arrow(pred, "red")
    inset.text(0.04, 0.94, "GT", transform=inset.transAxes, color="lime", fontsize=7, va="top")
    inset.text(0.32, 0.94, "Pred", transform=inset.transAxes, color="red", fontsize=7, va="top")
    inset.text(0.04, 0.04, f"err {error_deg:.1f} deg", transform=inset.transAxes, color="white", fontsize=7, va="bottom")


def plot_prediction_samples(model, records: Sequence[LabelRecord], args: argparse.Namespace, device, out_path: Path, max_images: int = 25) -> None:
    if not records or max_images <= 0:
        return
    sample_records = list(records)[:max_images]
    model.eval()
    n = len(sample_records)
    cols = min(5, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    with torch.no_grad():
        for ax, rec in zip(axes, sample_records):
            raw_image = read_rgb(Path(rec.image_path))
            mask = read_main_mask(Path(rec.mask_path) if rec.mask_path else None, raw_image.shape[:2])
            image = preprocess_rgb(raw_image, mask, imgsz=args.imgsz, mode=args.preprocess, margin_ratio=args.margin, background=args.background)
            input_tensor = normalize_to_tensor(image).unsqueeze(0).to(device)
            pred = normalize_vectors(model(input_tensor)).cpu().numpy()[0]
            target = np.array([rec.cos_theta, rec.sin_theta], dtype=float)

            ax.imshow(raw_image)
            pred_rad = math.atan2(float(pred[1]), float(pred[0]))
            diff = (pred_rad - rec.theta_rad + math.pi) % (2.0 * math.pi) - math.pi
            error_deg = abs(math.degrees(diff))
            draw_angle_inset(ax, target, pred, error_deg)
            # No per-image title — visible from inset arrows
            ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_test_csv(model, loader, records: Sequence[LabelRecord], device, out_path: Path) -> None:
    """Write per-sample test predictions to CSV."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            preds = model(images)
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    preds_cat = normalize_vectors(torch.cat(all_preds, dim=0))
    targets_cat = normalize_vectors(torch.cat(all_targets, dim=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_label", "true_theta_deg", "pred_theta_deg", "abs_error_deg",
            "true_cos_theta", "true_sin_theta", "pred_cos_theta", "pred_sin_theta",
            "cos_similarity",
        ])
        for i, rec in enumerate(records):
            tc, ts = float(targets_cat[i, 0]), float(targets_cat[i, 1])
            pc, ps = float(preds_cat[i, 0]), float(preds_cat[i, 1])
            true_deg = math.degrees(math.atan2(ts, tc) % (2.0 * math.pi))
            pred_deg = math.degrees(math.atan2(ps, pc) % (2.0 * math.pi))
            diff = (math.atan2(ps, pc) - math.atan2(ts, tc) + math.pi) % (2.0 * math.pi) - math.pi
            abs_err = abs(math.degrees(diff))
            cos_sim = tc * pc + ts * ps
            writer.writerow([
                rec.image_label,
                round(true_deg, 4), round(pred_deg, 4), round(abs_err, 4),
                round(tc, 6), round(ts, 6), round(pc, 6), round(ps, 6),
                round(cos_sim, 6),
            ])
    print(f"Test CSV: {out_path}")


def plot_error_analysis(csv_path: Path, out_path: Path) -> None:
    """4-panel error analysis figure from test_predictions.csv."""
    if not csv_path.exists():
        return

    rows_list = []
    with csv_path.open("r", encoding="utf-8") as f_:
        for row in csv.DictReader(f_):
            rows_list.append(row)
    if not rows_list:
        return

    abs_errs = np.array([float(r["abs_error_deg"]) for r in rows_list])
    true_degs = np.array([float(r["true_theta_deg"]) for r in rows_list])
    pred_degs = np.array([float(r["pred_theta_deg"]) for r in rows_list])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # (a) Error histogram
    ax = axes[0, 0]
    ax.hist(abs_errs, bins=40, color=BLUE_LIGHT, edgecolor="white", alpha=0.85, linewidth=0.5)
    p50, p90, p95 = np.percentile(abs_errs, [50, 90, 95])
    mean_err = float(np.mean(abs_errs))
    for val, label, ls in [(mean_err, "Mean", "-"), (p50, "P50", "--"), (p90, "P90", "-."), (p95, "P95", ":")]:
        ax.axvline(val, color=ORANGE_ACCENT, linestyle=ls, linewidth=1.8,
                    label=f"{label} = {val:.2f}°")
    ax.set_xlabel("Absolute Error (°)")
    ax.set_ylabel("Count")
    ax.set_title("Angle Error Distribution")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # (b) True vs Predicted scatter
    ax = axes[0, 1]
    ax.scatter(true_degs, pred_degs, c=BLUE_LIGHT, s=12, alpha=0.55, edgecolors="none")
    ax.plot([0, 360], [0, 360], color=ORANGE_ACCENT, linewidth=1.5, label="y = x")
    ax.plot([0, 360], [5, 365], color=ORANGE_ACCENT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot([5, 365], [0, 360], color=ORANGE_ACCENT, linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel("True Angle (°)")
    ax.set_ylabel("Predicted Angle (°)")
    ax.set_title(f"True vs Predicted (MAE = {mean_err:.2f}°)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_aspect("equal")

    # (c) CDF of absolute errors
    ax = axes[1, 0]
    sorted_errs = np.sort(abs_errs)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
    ax.plot(sorted_errs, cdf, color=BLUE_LIGHT, linewidth=2.2)
    ax.fill_between(sorted_errs, 0, cdf, color=BLUE_LIGHT, alpha=0.12)
    for pct in [50, 90, 95]:
        val = np.percentile(abs_errs, pct)
        ax.axvline(val, color=ORANGE_ACCENT, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.axhline(pct / 100.0, color=ORANGE_ACCENT, linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(val + 0.3, pct / 100.0 + 0.02, f"P{pct}={val:.2f}°",
                fontsize=8, color=ORANGE_ACCENT, fontweight="bold")
    ax.set_xlabel("Absolute Error (°)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Error Distribution")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # (d) Per-angle-bin error bars
    ax = axes[1, 1]
    bins = np.arange(0, 361, 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = []
    bin_stds = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = (true_degs >= bins[i]) & (true_degs < bins[i + 1])
        bin_means.append(np.mean(abs_errs[mask]) if mask.sum() > 0 else 0)
        bin_stds.append(np.std(abs_errs[mask]) if mask.sum() > 1 else 0)
        bin_counts.append(int(mask.sum()))
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bars = ax.bar(bin_centers, bin_means, width=25, color=BLUE_LIGHT, alpha=0.7,
                   edgecolor=BLUE_DARK, linewidth=0.8)
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="none",
                 ecolor=ORANGE_ACCENT, capsize=4, linewidth=1.4)
    for i, (cx, m, n) in enumerate(zip(bin_centers, bin_means, bin_counts)):
        if n > 0:
            ax.text(cx, m + bin_stds[i] + 0.2, f"n={n}", ha="center", fontsize=7,
                     color=BLUE_DARK, fontweight="bold")
    ax.set_xlabel("True Angle Range (°)")
    ax.set_ylabel("Mean Absolute Error (°)")
    ax.set_title("Error by Angle Range")
    ax.set_xticks(bin_centers)
    ax.set_xticklabels([f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins) - 1)],
                        rotation=30, ha="right", fontsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5, axis="y")

    fig.suptitle("Test Set Error Analysis", fontweight="bold", fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Error analysis: {out_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv)
    image_dir = Path(args.images)
    mask_dir = Path(args.masks) if args.masks else None
    run_dir = increment_path(Path(args.project), args.name, exist_ok=args.exist_ok)
    weights_dir = run_dir / "weights"
    splits_dir = run_dir / "splits"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("AMP: disabled")
    print(f"run_dir: {run_dir}")
    print(f"csv: {csv_path}")
    print(f"images: {image_dir}")
    require_masks = args.preprocess in {"crop", "mask"}
    print(f"masks: {mask_dir if require_masks else 'not used'}")
    print(f"preprocess: {args.preprocess} | imgsz={args.imgsz} | background={args.background}")

    shutil.copy2(csv_path, run_dir / "labels_used.csv")
    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    records = load_records(csv_path, image_dir, mask_dir, status=args.status, require_masks=require_masks)
    splits = split_records(records, seed=args.seed)
    for split_name, split_records_ in splits.items():
        write_split_csv(splits_dir / f"{split_name}.csv", split_name, split_records_)
    split_sizes = {key: len(value) for key, value in splits.items()}
    print("split sizes:", split_sizes)

    train_ds = KernelAngleDataset(splits["train"], args=args, train=True)
    val_ds = KernelAngleDataset(splits["val"], args=args, train=False)
    test_ds = KernelAngleDataset(splits["test"], args=args, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = resolve_device(args.device)
    model = build_resnet_angle_model(stem_channels=args.stem_channels, dropout=args.dropout, in_channels=args.in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"params: {n_params:,}")
    print(f"epochs={args.epochs} | patience={args.patience} | batch={args.batch_size}")

    results_csv = run_dir / "results.csv"
    best_val_mae = float("inf")
    best_epoch = 0
    patience_count = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, train=True, norm_loss_weight=args.norm_loss_weight)
        val_metrics = run_one_epoch(model, val_loader, optimizer, device, train=False, norm_loss_weight=args.norm_loss_weight)
        lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mae_deg": train_metrics["mae_deg"],
            "val_mae_deg": val_metrics["mae_deg"],
            "train_angle_p50": train_metrics["angle_p50"],
            "val_angle_p50": val_metrics["angle_p50"],
            "train_angle_p90": train_metrics["angle_p90"],
            "val_angle_p90": val_metrics["angle_p90"],
            "train_mean_cosine": train_metrics["mean_cosine"],
            "val_mean_cosine": val_metrics["mean_cosine"],
            "lr": lr,
        }
        append_results_csv(results_csv, row)

        improved = val_metrics["mae_deg"] < best_val_mae - 1e-7
        if improved:
            best_val_mae = val_metrics["mae_deg"]
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(weights_dir / "best.pt", model, optimizer, epoch, best_val_mae, args, split_sizes)
        else:
            patience_count += 1

        save_checkpoint(weights_dir / "last.pt", model, optimizer, epoch, best_val_mae, args, split_sizes)

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.5f} val_loss={val_metrics['loss']:.5f} "
            f"train_mae={train_metrics['mae_deg']:.2f} val_mae={val_metrics['mae_deg']:.2f} "
            f"val_p50={val_metrics['angle_p50']:.2f} val_p90={val_metrics['angle_p90']:.2f} "
            f"best_mae={best_val_mae:.2f}@{best_epoch} patience={patience_count}/{args.patience}"
        )

        plot_results(results_csv, run_dir / "results.png")

        if patience_count >= args.patience:
            print(f"Early stopping at epoch {epoch}; no validation MAE improvement for {args.patience} epochs.")
            break

    best_ckpt = torch.load(weights_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, train=False, norm_loss_weight=args.norm_loss_weight)
    with (run_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    test_csv_path = run_dir / "test_predictions.csv"
    generate_test_csv(model, test_loader, splits["test"], device, test_csv_path)
    plot_error_analysis(test_csv_path, run_dir / "test_error_analysis.png")

    plot_prediction_samples(model, splits["test"], args, device, run_dir / "test_predictions.png", max_images=args.save_samples)

    elapsed_min = (time.time() - start) / 60.0
    print("\nTraining complete")
    print(f"run_dir: {run_dir}")
    print(f"best weights: {weights_dir / 'best.pt'}")
    print(f"last weights: {weights_dir / 'last.pt'}")
    print(f"results: {results_csv}")
    print(f"results plot: {run_dir / 'results.png'}")
    print(f"test CSV: {test_csv_path}")
    print(f"error analysis: {run_dir / 'test_error_analysis.png'}")
    print(f"test predictions: {run_dir / 'test_predictions.png'}")
    print(f"test metrics: {test_metrics}")
    print(f"elapsed: {elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
