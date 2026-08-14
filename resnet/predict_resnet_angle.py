"""Predict directed kernel-axis angles with a trained custom ResNet-style model."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import build_resnet_angle_model
from preprocess import normalize_to_tensor, preprocess_rgb, read_main_mask, read_rgb
from train_resnet_angle import normalize_vectors, safe_float


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "image_data"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PredictRecord:
    image_label: str
    image_path: str
    mask_path: str
    gt_cos_theta: Optional[float] = None
    gt_sin_theta: Optional[float] = None
    gt_theta_rad: Optional[float] = None
    gt_theta_deg: Optional[float] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict directed kernel-axis angles with a trained ResNet-style checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to weights/best.pt or weights/last.pt.")
    parser.add_argument("--images", default=str(DEFAULT_DATA_DIR / "subimages"), help="RGB kernel crop directory.")
    parser.add_argument("--masks", default=str(DEFAULT_DATA_DIR / "masks_binary"), help="Binary mask directory.")
    parser.add_argument("--csv", default="", help="Optional label CSV for ordering and GT error calculation.")
    parser.add_argument("--out", default="", help="Output predictions CSV. Default: beside weights as predictions.csv.")
    parser.add_argument("--overlay-dir", default="", help="Optional directory for per-image overlay JPGs.")
    parser.add_argument("--grid", default="", help="Output grid image. Default: beside weights as predictions_grid.png.")
    parser.add_argument("--status", default="ok", help="CSV status value to use when --csv is provided.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=0, help="Override checkpoint image size. 0 means use checkpoint imgsz.")
    parser.add_argument("--preprocess", default="", choices=["", "crop", "mask", "square", "none"], help="Override checkpoint preprocessing mode.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--max-overlays", type=int, default=200, help="Maximum per-image overlays to save; use -1 for all.")
    parser.add_argument("--grid-samples", type=int, default=32, help="Number of examples in the grid image.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def directed_angle_error_deg(pred_rad: float, gt_rad: float) -> float:
    diff = (pred_rad - gt_rad + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(diff))


def load_records_from_csv(csv_path: Path, image_dir: Path, mask_dir: Optional[Path], status: str) -> List[PredictRecord]:
    records: List[PredictRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "status" not in (reader.fieldnames or []):
            raise ValueError("CSV is missing required column: status")
        for row in reader:
            if str(row.get("status", "")).strip().lower() != status.lower():
                continue
            image_label = str(row.get("image_label") or Path(str(row.get("image_path", ""))).name).strip()
            image_path = image_dir / image_label
            if not image_path.exists():
                print(f"WARNING: image not found, skipping: {image_path}")
                continue
            mask_path = mask_dir / image_label if mask_dir is not None else None

            gt_cos = gt_sin = gt_rad = gt_deg = None
            if row.get("cos_theta") not in (None, "") and row.get("sin_theta") not in (None, ""):
                gt_cos = safe_float(row.get("cos_theta"))
                gt_sin = safe_float(row.get("sin_theta"))
                norm = math.hypot(gt_cos, gt_sin)
                if norm > 1e-6:
                    gt_cos /= norm
                    gt_sin /= norm
                    gt_rad = math.atan2(gt_sin, gt_cos) % (2.0 * math.pi)
                    gt_deg = math.degrees(gt_rad)

            records.append(
                PredictRecord(
                    image_label=image_label,
                    image_path=str(image_path),
                    mask_path=str(mask_path) if mask_path is not None and mask_path.exists() else "",
                    gt_cos_theta=gt_cos,
                    gt_sin_theta=gt_sin,
                    gt_theta_rad=gt_rad,
                    gt_theta_deg=gt_deg,
                )
            )
    return records


def load_records_from_images(image_dir: Path, mask_dir: Optional[Path]) -> List[PredictRecord]:
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS and path.is_file())
    records: List[PredictRecord] = []
    for path in paths:
        mask_path = mask_dir / path.name if mask_dir is not None else None
        records.append(
            PredictRecord(
                image_label=path.name,
                image_path=str(path),
                mask_path=str(mask_path) if mask_path is not None and mask_path.exists() else "",
            )
        )
    return records


class PredictDataset(Dataset):
    def __init__(self, records: Sequence[PredictRecord], imgsz: int, preprocess_mode: str, margin: float, background: str):
        self.records = list(records)
        self.imgsz = imgsz
        self.preprocess_mode = preprocess_mode
        self.margin = margin
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


def draw_angle_inset(ax, target: Optional[np.ndarray], pred: np.ndarray, error_deg: Optional[float]) -> None:
    """Draw GT/pred direction arrows in a fixed lower-left inset, not on the kernel."""
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

    if target is not None:
        arrow(target, "lime")
        inset.text(0.04, 0.94, "GT", transform=inset.transAxes, color="lime", fontsize=7, va="top")
        inset.text(0.32, 0.94, "Pred", transform=inset.transAxes, color="red", fontsize=7, va="top")
    else:
        inset.text(0.04, 0.94, "Pred", transform=inset.transAxes, color="red", fontsize=7, va="top")
    arrow(pred, "red")
    if error_deg is not None:
        inset.text(0.04, 0.04, f"err {error_deg:.1f} deg", transform=inset.transAxes, color="white", fontsize=7, va="bottom")


def save_overlay(rec: PredictRecord, pred_cos: float, pred_sin: float, pred_deg: float, out_path: Path, error_deg: Optional[float]) -> None:
    image = read_rgb(Path(rec.image_path))
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(image)
    target = None
    if rec.gt_cos_theta is not None and rec.gt_sin_theta is not None:
        target = np.array([rec.gt_cos_theta, rec.gt_sin_theta], dtype=float)
    draw_angle_inset(ax, target, np.array([pred_cos, pred_sin], dtype=float), error_deg)
    title = f"{rec.image_label}\npred={pred_deg:.1f} deg"
    if error_deg is not None:
        title += f" | err={error_deg:.1f} deg"
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(pad=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_grid(rows: Sequence[Dict[str, str]], records_by_name: Dict[str, PredictRecord], out_path: Path, max_items: int) -> None:
    if max_items <= 0 or not rows:
        return
    selected = list(rows)[:max_items]
    n = len(selected)
    cols = min(4, n)
    fig_rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(fig_rows, cols, figsize=(4.2 * cols, 4.2 * fig_rows))
    axes = np.array(axes).reshape(-1)

    for ax, row in zip(axes, selected):
        rec = records_by_name[row["image_label"]]
        image = read_rgb(Path(rec.image_path))
        pred_cos = safe_float(row["pred_cos_theta"])
        pred_sin = safe_float(row["pred_sin_theta"])
        pred_deg = safe_float(row["pred_theta_deg"])
        error_value = row.get("angle_error_deg", "")

        ax.imshow(image)
        target = None
        if rec.gt_cos_theta is not None and rec.gt_sin_theta is not None:
            target = np.array([rec.gt_cos_theta, rec.gt_sin_theta], dtype=float)
        error_deg = safe_float(error_value) if error_value else None
        draw_angle_inset(ax, target, np.array([pred_cos, pred_sin], dtype=float), error_deg)
        title = f"{rec.image_label}\npred={pred_deg:.1f} deg"
        if error_value:
            title += f" | err={safe_float(error_value):.1f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    image_dir = Path(args.images)
    mask_dir = Path(args.masks) if args.masks else None
    device = resolve_device(args.device)

    checkpoint = torch.load(weights_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model_config = checkpoint.get("model_config", {})
    imgsz = int(args.imgsz or checkpoint.get("imgsz", ckpt_args.get("imgsz", 256)))
    preprocess_mode = args.preprocess or ckpt_args.get("preprocess", "crop")
    margin = float(ckpt_args.get("margin", 0.08))
    background = str(ckpt_args.get("background", "black"))

    if args.csv:
        records = load_records_from_csv(Path(args.csv), image_dir, mask_dir, args.status)
    else:
        records = load_records_from_images(image_dir, mask_dir)
    if not records:
        raise RuntimeError("No images found for prediction.")

    dataset = PredictDataset(records, imgsz=imgsz, preprocess_mode=preprocess_mode, margin=margin, background=background)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = build_resnet_angle_model(
        stem_channels=int(model_config.get("stem_channels", 32)),
        channels=model_config.get("channels", [64, 128, 256, 512]),
        blocks=model_config.get("blocks", [2, 2, 3, 2]),
        dropout=float(model_config.get("dropout", 0.20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    out_csv = Path(args.out) if args.out else weights_path.parent.parent / "predictions.csv"
    grid_path = Path(args.grid) if args.grid else weights_path.parent.parent / "predictions_grid.png"
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None

    rows: List[Dict[str, str]] = []
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
                error_deg = None
                if rec.gt_theta_rad is not None:
                    error_deg = directed_angle_error_deg(pred_rad, rec.gt_theta_rad)

                row = {
                    "image_label": rec.image_label,
                    "image_path": rec.image_path,
                    "pred_theta_deg": f"{pred_deg:.8f}",
                    "pred_theta_rad": f"{pred_rad:.10f}",
                    "pred_cos_theta": f"{pred_cos:.10f}",
                    "pred_sin_theta": f"{pred_sin:.10f}",
                    "gt_theta_deg": "" if rec.gt_theta_deg is None else f"{rec.gt_theta_deg:.8f}",
                    "gt_theta_rad": "" if rec.gt_theta_rad is None else f"{rec.gt_theta_rad:.10f}",
                    "gt_cos_theta": "" if rec.gt_cos_theta is None else f"{rec.gt_cos_theta:.10f}",
                    "gt_sin_theta": "" if rec.gt_sin_theta is None else f"{rec.gt_sin_theta:.10f}",
                    "angle_error_deg": "" if error_deg is None else f"{error_deg:.8f}",
                }
                rows.append(row)

                if overlay_dir is not None and (args.max_overlays < 0 or len(rows) <= args.max_overlays):
                    save_overlay(rec, pred_cos, pred_sin, pred_deg, overlay_dir / f"{Path(rec.image_label).stem}_pred.jpg", error_deg)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_label",
        "image_path",
        "pred_theta_deg",
        "pred_theta_rad",
        "pred_cos_theta",
        "pred_sin_theta",
        "gt_theta_deg",
        "gt_theta_rad",
        "gt_cos_theta",
        "gt_sin_theta",
        "angle_error_deg",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    records_by_name = {rec.image_label: rec for rec in records}
    save_grid(rows, records_by_name, grid_path, args.grid_samples)

    errors = [safe_float(row["angle_error_deg"]) for row in rows if row["angle_error_deg"]]
    print(f"weights: {weights_path}")
    print(f"images: {image_dir}")
    print(f"masks: {mask_dir}")
    print(f"imgsz: {imgsz} | preprocess={preprocess_mode}")
    print(f"predictions: {out_csv}")
    print(f"grid: {grid_path}")
    if overlay_dir is not None:
        print(f"overlays: {overlay_dir}")
    if errors:
        print(f"mean angle error: {np.mean(errors):.3f} deg | median: {np.median(errors):.3f} deg")


if __name__ == "__main__":
    main()
