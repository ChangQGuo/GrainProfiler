"""
Evaluate a trained ResNet-style angle regressor on the saved test split.

Outputs:
  - one validation figure per test kernel
  - test_predictions_result.csv with image label, predicted angle,
    true angle, and angle error

This script does not modify training code or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import build_resnet_angle_model
from preprocess import normalize_to_tensor, preprocess_rgb, read_main_mask, read_rgb


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "runs" / "angle" / "rgb_square256" / "weights" / "best.pt"


@dataclass
class TestRecord:
    image_label: str
    image_path: str
    mask_path: str
    true_cos_theta: float
    true_sin_theta: float
    true_theta_rad: float
    true_theta_deg: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best.pt on the saved test split and export per-kernel figures.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best.pt.")
    parser.add_argument("--test-csv", default="", help="Saved test split CSV. Default: <run_dir>/splits/test.csv.")
    parser.add_argument("--images", default="", help="RGB crop directory. Default: checkpoint args.images, then ../image_data/subimages.")
    parser.add_argument("--masks", default="", help="Optional mask directory. Usually unused for square preprocessing.")
    parser.add_argument("--out-dir", default="", help="Directory for per-kernel validation figures. Default: <run_dir>/test_predictions_images.")
    parser.add_argument("--out-csv", default="", help="Output result CSV. Default: <run_dir>/test_predictions_result.csv.")
    parser.add_argument("--metrics-json", default="", help="Output summary JSON. Default: <run_dir>/test_metrics_result.json.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--imgsz", type=int, default=0, help="Override checkpoint image size. 0 means use checkpoint setting.")
    parser.add_argument("--preprocess", default="", choices=["", "square", "none", "crop", "mask"], help="Override checkpoint preprocessing.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick testing. 0 means all test samples.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def normalize_pair(cos_t: float, sin_t: float) -> tuple[float, float]:
    norm = math.hypot(cos_t, sin_t)
    if norm < 1e-9:
        raise ValueError("Invalid zero-length angle vector.")
    return cos_t / norm, sin_t / norm


def normalize_vectors(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=1, eps=1e-6)


def directed_angle_error_deg(pred_rad: float, true_rad: float) -> float:
    diff = (pred_rad - true_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(diff))


def resolve_image_path(row: Dict[str, str], image_dir: Path) -> Optional[Path]:
    image_label = str(row.get("image_label") or Path(str(row.get("image_path", ""))).name).strip()
    raw_path = str(row.get("image_path", "")).strip()
    candidates: List[Path] = []
    if raw_path:
        path = Path(raw_path)
        candidates.append(path)
        if not path.is_absolute():
            candidates.append((Path.cwd() / path).resolve())
    if image_label:
        candidates.append(image_dir / image_label)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_test_records(test_csv: Path, image_dir: Path, mask_dir: Optional[Path], limit: int = 0) -> List[TestRecord]:
    records: List[TestRecord] = []
    with test_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_label", "cos_theta", "sin_theta"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Test CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            image_label = str(row.get("image_label", "")).strip()
            image_path = resolve_image_path(row, image_dir)
            if image_path is None:
                print(f"WARNING: image not found, skipping: {image_label}")
                continue

            cos_t, sin_t = normalize_pair(safe_float(row.get("cos_theta")), safe_float(row.get("sin_theta")))
            true_rad = math.atan2(sin_t, cos_t) % (2.0 * math.pi)
            mask_path = ""
            raw_mask_path = str(row.get("mask_path", "")).strip()
            if raw_mask_path and Path(raw_mask_path).exists():
                mask_path = raw_mask_path
            elif mask_dir is not None and (mask_dir / image_label).exists():
                mask_path = str(mask_dir / image_label)

            records.append(
                TestRecord(
                    image_label=image_label,
                    image_path=str(image_path),
                    mask_path=mask_path,
                    true_cos_theta=cos_t,
                    true_sin_theta=sin_t,
                    true_theta_rad=true_rad,
                    true_theta_deg=math.degrees(true_rad),
                )
            )
            if limit > 0 and len(records) >= limit:
                break

    if not records:
        raise RuntimeError(f"No usable test records found in {test_csv}")
    return records


class TestDataset(Dataset):
    def __init__(self, records: Sequence[TestRecord], imgsz: int, preprocess_mode: str, margin: float, background: str):
        self.records = list(records)
        self.imgsz = int(imgsz)
        self.preprocess_mode = preprocess_mode
        self.margin = float(margin)
        self.background = background

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = read_rgb(Path(rec.image_path))
        mask = read_main_mask(Path(rec.mask_path) if rec.mask_path else None, image.shape[:2])
        image = preprocess_rgb(
            image,
            mask,
            imgsz=self.imgsz,
            mode=self.preprocess_mode,
            margin_ratio=self.margin,
            background=self.background,
        )
        return normalize_to_tensor(image), index


def draw_angle_inset(ax, target: np.ndarray, pred: np.ndarray, error_deg: float) -> None:
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


def save_validation_figure(rec: TestRecord, pred_cos: float, pred_sin: float, error_deg: float, out_path: Path) -> None:
    image = read_rgb(Path(rec.image_path))
    target = np.array([rec.true_cos_theta, rec.true_sin_theta], dtype=float)
    pred = np.array([pred_cos, pred_sin], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)
    draw_angle_inset(ax, target, pred, error_deg)
    ax.set_title(rec.image_label, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_model_from_checkpoint(checkpoint: Dict, device: torch.device):
    model_config = checkpoint.get("model_config", {})
    model = build_resnet_angle_model(
        stem_channels=int(model_config.get("stem_channels", 32)),
        channels=model_config.get("channels", [64, 128, 256, 512]),
        blocks=model_config.get("blocks", [2, 2, 3, 2]),
        dropout=float(model_config.get("dropout", 0.20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def summarize_errors(errors: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(errors, dtype=float)
    return {
        "count": int(array.size),
        "mae_deg": float(array.mean()),
        "median_p50_deg": float(np.percentile(array, 50)),
        "p90_deg": float(np.percentile(array, 90)),
        "max_deg": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    run_dir = weights_path.parent.parent
    test_csv = Path(args.test_csv) if args.test_csv else run_dir / "splits" / "test.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "test_predictions_images"
    out_csv = Path(args.out_csv) if args.out_csv else run_dir / "test_predictions_result.csv"
    metrics_json = Path(args.metrics_json) if args.metrics_json else run_dir / "test_metrics_result.json"

    device = resolve_device(args.device)
    checkpoint = load_checkpoint(weights_path, device)
    ckpt_args = checkpoint.get("args", {})

    image_dir = Path(args.images or ckpt_args.get("images", DEFAULT_DATA_DIR / "subimages"))
    mask_dir = Path(args.masks) if args.masks else None
    imgsz = int(args.imgsz or checkpoint.get("imgsz", ckpt_args.get("imgsz", 256)))
    preprocess_mode = args.preprocess or ckpt_args.get("preprocess", "square")
    margin = float(ckpt_args.get("margin", 0.25))
    background = str(ckpt_args.get("background", "black"))

    records = load_test_records(test_csv, image_dir=image_dir, mask_dir=mask_dir, limit=args.limit)
    dataset = TestDataset(records, imgsz=imgsz, preprocess_mode=preprocess_mode, margin=margin, background=background)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model_from_checkpoint(checkpoint, device)

    rows: List[Dict[str, str]] = []
    errors: List[float] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            preds = normalize_vectors(model(images)).cpu().numpy()
            for pred, index in zip(preds, indices.numpy().tolist()):
                rec = records[int(index)]
                pred_cos = float(pred[0])
                pred_sin = float(pred[1])
                pred_rad = math.atan2(pred_sin, pred_cos) % (2.0 * math.pi)
                pred_deg = math.degrees(pred_rad)
                error_deg = directed_angle_error_deg(pred_rad, rec.true_theta_rad)
                errors.append(error_deg)

                figure_path = out_dir / f"{Path(rec.image_label).stem}_test_prediction.png"
                save_validation_figure(rec, pred_cos, pred_sin, error_deg, figure_path)

                rows.append(
                    {
                        "image_label": rec.image_label,
                        "pred_theta_deg": f"{pred_deg:.8f}",
                        "true_theta_deg": f"{rec.true_theta_deg:.8f}",
                        "angle_error_deg": f"{error_deg:.8f}",
                        "pred_cos_theta": f"{pred_cos:.10f}",
                        "pred_sin_theta": f"{pred_sin:.10f}",
                        "true_cos_theta": f"{rec.true_cos_theta:.10f}",
                        "true_sin_theta": f"{rec.true_sin_theta:.10f}",
                        "figure_path": str(figure_path),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_label",
        "pred_theta_deg",
        "true_theta_deg",
        "angle_error_deg",
        "pred_cos_theta",
        "pred_sin_theta",
        "true_cos_theta",
        "true_sin_theta",
        "figure_path",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_errors(errors)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"weights: {weights_path}")
    print(f"test split: {test_csv}")
    print(f"images: {image_dir}")
    print(f"preprocess: {preprocess_mode} | imgsz={imgsz}")
    print(f"figures: {out_dir}")
    print(f"result csv: {out_csv}")
    print(f"metrics json: {metrics_json}")
    print(f"test MAE: {summary['mae_deg']:.3f} deg | p50: {summary['median_p50_deg']:.3f} deg | p90: {summary['p90_deg']:.3f} deg")


if __name__ == "__main__":
    main()
