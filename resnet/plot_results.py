"""Regenerate all training result figures from a run directory.

Reads results.csv and test_predictions.csv — no model/PyTorch needed.
Prediction samples are drawn directly from the CSV's stored angle values.

Usage:
  python plot_results.py runs/angle/rgb_square256
  python plot_results.py runs/angle/rgb_square256 --images-dir ../resnet_dataset/subimages
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def smooth_curve(values, weight=0.72):
    if not values:
        return []
    sm = [float(values[0])]
    for v in values[1:]:
        sm.append(sm[-1] * weight + float(v) * (1.0 - weight))
    return sm


# ── Global style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 19,
    "font.weight": "bold",
    "axes.titlesize": 24,  "axes.titleweight": "bold",
    "axes.labelsize": 21,  "axes.labelweight": "bold",
    "xtick.labelsize": 18, "ytick.labelsize": 18,
    "legend.fontsize": 17,
})

# ── Colours ──
BLUE_PRIMARY = "#2C6FAC"
BLUE_LIGHT   = "#7BAFD4"
BLUE_PALE    = "#B8D4EC"
CORAL        = "#E87D4F"
RED_ACCENT   = "#E03030"
RED_BRIGHT   = "#C0392B"
GREEN_MANUAL = "#3CB371"
RED_PRED     = "#E05555"
GREY_GRID    = "#D0D0D0"
ORANGE_MARK  = "#E67E22"
PURPLE_MARK  = "#7D3C98"
BLUE_MARK    = "#1F618D"

# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _load_csv(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols: dict[str, list[float]] = {}
    for key in rows[0].keys():
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (ValueError, TypeError):
                vals.append(float("nan"))
        cols[key] = vals
    return cols


def _curve_pair(ax, epochs, train_vals, val_vals, title, c1, c2):
    ax.plot(epochs, train_vals, color=c1, alpha=0.18, linewidth=0.8)
    ax.plot(epochs, val_vals,   color=c2, alpha=0.18, linewidth=0.8)
    ax.plot(epochs, smooth_curve(train_vals), label="Train", color=c1, linewidth=2.4)
    ax.plot(epochs, smooth_curve(val_vals),   label="Val",   color=c2, linewidth=2.4)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.85)
    ax.grid(True, alpha=0.25, linewidth=0.6, color=GREY_GRID)


def _draw_arrow_inset(ax, manual_cos, manual_sin, pred_cos, pred_sin, error_deg,
                      large=False):
    """Draw manual (green) and predicted (red) direction arrows on an inset."""
    # Larger inset for per-kernel images
    size = 0.40 if large else 0.34
    inset = ax.inset_axes([0.03, 0.03, size, size])
    inset.set_facecolor((0.0, 0.0, 0.0, 0.72))
    for sp in inset.spines.values():
        sp.set_color("white"); sp.set_linewidth(0.8)
    inset.set_xlim(-1.0, 1.0); inset.set_ylim(-1.0, 1.0)
    inset.set_aspect("equal")
    inset.set_xticks([]); inset.set_yticks([])

    def _arrow(cs, color):
        norm = math.hypot(cs[0], cs[1])
        if norm < 1e-9:
            return
        v = (cs[0] / norm, cs[1] / norm)
        dv = (v[0], -v[1])
        inset.arrow(0, 0, dv[0] * 0.72, dv[1] * 0.72,
                    color=color, width=0.035, head_width=0.16,
                    length_includes_head=True, alpha=0.95)

    _arrow((manual_cos, manual_sin), GREEN_MANUAL)
    _arrow((pred_cos, pred_sin), RED_PRED)

    fs = 11 if large else 9
    inset.text(0.04, 0.94, "Manual", transform=inset.transAxes,
               color=GREEN_MANUAL, fontsize=fs, va="top", fontweight="bold")
    inset.text(0.04, 0.82, "Pred", transform=inset.transAxes,
               color=RED_PRED, fontsize=fs, va="top", fontweight="bold")
    inset.text(0.04, 0.04, f"err {error_deg:.1f} deg",
               transform=inset.transAxes, color="white", fontsize=fs, va="bottom")

# ════════════════════════════════════════════════════════════
# 1. Training curves
# ════════════════════════════════════════════════════════════

def plot_results(results_csv: Path, out_path: Path):
    data = _load_csv(results_csv)
    if not data:
        print("  WARNING: cannot read results.csv"); return

    epochs = data["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    _curve_pair(axes[0, 0], epochs, data["train_loss"], data["val_loss"],
                "Loss", BLUE_PRIMARY, CORAL)
    axes[0, 0].set_ylabel("Loss")

    _curve_pair(axes[0, 1], epochs, data["train_mae_deg"], data["val_mae_deg"],
                "MAE (degrees)", BLUE_PRIMARY, CORAL)
    axes[0, 1].set_ylabel("MAE (degrees)")
    if data.get("val_mae_deg"):
        vals = np.asarray(data["val_mae_deg"], dtype=float)
        bi = int(np.argmin(vals))
        best_val = vals[bi]
        axes[0, 1].axvline(epochs[bi], color=CORAL, linestyle=":", linewidth=1.8)
        y_range = max(vals.max() - vals.min(), 0.1)
        y_text = best_val - y_range * 0.08
        axes[0, 1].text(epochs[bi] + 1, y_text,
                        f"Best {best_val:.2f} deg", fontsize=12,
                        color=RED_ACCENT, fontweight="bold")

    _curve_pair(axes[1, 0], epochs, data["train_mean_cosine"],
                data["val_mean_cosine"], "Cosine Similarity", BLUE_PRIMARY, CORAL)
    axes[1, 0].set_ylim(0.65, 1.02)
    axes[1, 0].set_ylabel("Cosine Similarity")

    axes[1, 1].plot(epochs, data["lr"], color=BLUE_PRIMARY, linewidth=2.2)
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].grid(True, alpha=0.25, linewidth=0.6, color=GREY_GRID)

    for ax in axes.ravel():
        ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  results.png → {out_path}")

# ════════════════════════════════════════════════════════════
# 2. Error analysis
# ════════════════════════════════════════════════════════════

def plot_error_analysis(test_csv: Path, out_path: Path):
    if not test_csv.exists():
        print("  WARNING: test_predictions.csv not found"); return

    manual_degs, pred_degs, abs_errs = [], [], []
    with test_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manual_degs.append(float(row["true_theta_deg"]))
            pred_degs.append(float(row["pred_theta_deg"]))
            abs_errs.append(float(row["abs_error_deg"]))
    manual_degs = np.array(manual_degs)
    pred_degs = np.array(pred_degs)
    abs_errs = np.array(abs_errs)
    mean_err = float(np.mean(abs_errs))
    p50, p90, p95 = np.percentile(abs_errs, [50, 90, 95])
    p99 = float(np.percentile(abs_errs, 99))

    fig, axes = plt.subplots(2, 2, figsize=(22, 16.5))

    # (a) Histogram
    ax = axes[0, 0]
    ax.hist(abs_errs, bins=45, color=BLUE_PRIMARY, edgecolor="white",
            alpha=0.85, linewidth=0.5)
    for val, label, ls, c in [
            (mean_err, f"Mean = {mean_err:.2f}°", "-",  RED_BRIGHT),
            (p50,      f"P50  = {p50:.2f}°",     "--", ORANGE_MARK),
            (p90,      f"P90  = {p90:.2f}°",     "-.", PURPLE_MARK),
            (p95,      f"P95  = {p95:.2f}°",     ":",  BLUE_MARK)]:
        ax.axvline(val, color=c, linestyle=ls, linewidth=2.8, label=label)
    ax.set_xlabel("Absolute Error (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Angle Error Distribution")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.25, linewidth=0.5, color=GREY_GRID)

    # Stats box in top-left corner
    frac_lt5  = 100.0 * np.mean(abs_errs < 5.0)
    frac_lt10 = 100.0 * np.mean(abs_errs < 10.0)
    ax.text(0.02, 0.97,
            f"n = {len(abs_errs)}\n{frac_lt5:.1f}%  < 5°\n{frac_lt10:.1f}% < 10°",
            transform=ax.transAxes, ha="left", va="top", fontsize=17,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # (b) Scatter — Manual vs Predicted
    ax = axes[0, 1]
    ax.scatter(manual_degs, pred_degs, c=BLUE_PRIMARY, s=42, alpha=0.45,
               edgecolors="none")
    ax.plot([0, 360], [0, 360], color=RED_BRIGHT, linewidth=2.6)
    ax.plot([0, 360], [5, 365], color=RED_BRIGHT, linewidth=1.0, linestyle="--", alpha=0.45)
    ax.plot([5, 365], [0, 360], color=RED_BRIGHT, linewidth=1.0, linestyle="--", alpha=0.45)
    ax.set_xlim(0, 360); ax.set_ylim(0, 360)
    ax.set_xlabel("Manual Angle (degrees)")
    ax.set_ylabel("Predicted Angle (degrees)")
    ax.set_title("Manual vs Predicted")
    ax.grid(True, alpha=0.25, linewidth=0.5, color=GREY_GRID)
    ax.set_aspect("equal")

    # Stats annotation
    r_pearson = float(np.corrcoef(manual_degs, pred_degs)[0, 1])
    ax.text(0.03, 0.94,
            f"MAE = {mean_err:.2f}°\nPearson r = {r_pearson:.4f}\nn = {len(abs_errs)}",
            transform=ax.transAxes, ha="left", va="top", fontsize=17,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # (c) CDF
    ax = axes[1, 0]
    se = np.sort(abs_errs)
    cdf = np.arange(1, len(se) + 1) / len(se)
    ax.plot(se, cdf, color=BLUE_PRIMARY, linewidth=3.4)
    ax.fill_between(se, 0, cdf, color=BLUE_PALE, alpha=0.30)
    for pct, pval, c in [(50, p50, ORANGE_MARK), (90, p90, PURPLE_MARK),
                         (95, p95, RED_BRIGHT), (99, p99, BLUE_MARK)]:
        ax.axvline(pval, color=c, linestyle="--", linewidth=1.8, alpha=0.75)
        ax.axhline(pct / 100.0, color=c, linestyle="--", linewidth=1.8, alpha=0.75)
        ax.text(pval + 0.6, pct / 100.0 + 0.018, f"P{pct}={pval:.1f}°",
                fontsize=17, color=c, fontweight="bold")
    ax.set_xlabel("Absolute Error (degrees)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("Cumulative Error Distribution")
    ax.set_xlim(left=0); ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.25, linewidth=0.5, color=GREY_GRID)

    # (d) Per-angle-bin
    ax = axes[1, 1]
    bins = np.arange(0, 361, 30)
    centers = (bins[:-1] + bins[1:]) / 2
    means, stds, counts = [], [], []
    for i in range(len(bins) - 1):
        m = (manual_degs >= bins[i]) & (manual_degs < bins[i + 1])
        n = int(m.sum())
        counts.append(n)
        means.append(np.mean(abs_errs[m]) if n > 0 else 0.0)
        stds.append(np.std(abs_errs[m]) if n > 1 else 0.0)
    means = np.array(means); stds = np.array(stds)
    ax.bar(centers, means, width=24, color=BLUE_PRIMARY, alpha=0.80,
           edgecolor="#1A4B73", linewidth=1.2)
    ax.errorbar(centers, means, yerr=stds, fmt="none",
                ecolor=RED_BRIGHT, capsize=7, linewidth=2.5)
    # n-count labels: keep them inside the axes (headroom so they never
    # overlap the top border)
    label_top = max((m + stds[i] + 0.55 for i, m in enumerate(means)
                     if counts[i] > 0), default=1.0)
    ax.set_ylim(0, label_top * 1.08)
    for i, (cx, m) in enumerate(zip(centers, means)):
        if counts[i] > 0:
            ax.text(cx, m + stds[i] + 0.55, f"n={counts[i]}",
                    ha="center", va="bottom", fontsize=16, color="black")
    ax.set_xlabel("Manual Angle Range (degrees)")
    ax.set_ylabel("Mean Absolute Error (degrees)")
    ax.set_title("Error by Angle Range")
    xt = [f"{int(bins[i])}–{int(bins[i+1])}" for i in range(len(bins) - 1)]
    ax.set_xticks(centers)
    ax.set_xticklabels(xt, rotation=30, ha="right", fontsize=16)
    ax.grid(True, alpha=0.25, linewidth=0.5, color=GREY_GRID, axis="y")

    fig.tight_layout(pad=4.0, h_pad=3.4, w_pad=2.4)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  test_error_analysis.png → {out_path}")

# ════════════════════════════════════════════════════════════
# 3. Per-kernel prediction images (every test sample)
# ════════════════════════════════════════════════════════════

def plot_per_kernel(test_csv: Path, images_dir: Path, out_dir: Path):
    """Generate one prediction-overlay image per test kernel."""
    if not _HAS_CV2:
        print("  SKIP: opencv-python not available")
        return
    if not test_csv.exists():
        print("  WARNING: test_predictions.csv not found"); return
    if not images_dir.is_dir():
        print(f"  WARNING: images dir not found: {images_dir}"); return

    rows_list = []
    with test_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_list.append(row)
    if not rows_list:
        print("  WARNING: test CSV is empty"); return

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows_list)
    for i, row in enumerate(rows_list):
        img_name = row["image_label"]
        img_path = images_dir / img_name
        if not img_path.exists():
            candidates = list(images_dir.glob(img_name))
            if not candidates:
                continue
            img_path = candidates[0]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        manual_cos = float(row["true_cos_theta"])
        manual_sin = float(row["true_sin_theta"])
        pred_cos = float(row["pred_cos_theta"])
        pred_sin = float(row["pred_sin_theta"])
        err = float(row["abs_error_deg"])

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img_rgb)
        _draw_arrow_inset(ax, manual_cos, manual_sin, pred_cos, pred_sin,
                          err, large=True)
        ax.axis("off")

        # Legend outside the image area
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=GREEN_MANUAL, alpha=0.7,
                  label=f"Manual ({float(row['true_theta_deg']):.1f} deg)"),
            Patch(facecolor=RED_PRED, alpha=0.7,
                  label=f"Pred ({float(row['pred_theta_deg']):.1f} deg)"),
        ]
        ax.legend(handles=legend_elements, loc="lower center",
                  bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=True,
                  fontsize=13, framealpha=0.9)

        stem = Path(img_name).stem
        out_path = out_dir / f"{stem}_pred.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{total}")

    print(f"  per-kernel predictions → {out_dir}/  ({total} images)")

# ════════════════════════════════════════════════════════════
# 4. Prediction sample grid
# ════════════════════════════════════════════════════════════

def plot_prediction_grid(test_csv: Path, images_dir: Path,
                         out_path: Path, max_images: int = 25):
    """5×5 grid of prediction samples."""
    if not _HAS_CV2:
        print("  SKIP: opencv-python not available")
        return
    if not test_csv.exists():
        print("  WARNING: test_predictions.csv not found"); return
    if not images_dir.is_dir():
        print(f"  WARNING: images dir not found: {images_dir}"); return

    rows_list = []
    with test_csv.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_list.append(row)
    if not rows_list:
        print("  WARNING: test CSV is empty"); return

    # Pick evenly spaced samples across error distribution
    rows_sorted = sorted(rows_list, key=lambda r: float(r["abs_error_deg"]))
    if len(rows_sorted) <= max_images:
        samples = rows_sorted
    else:
        indices = np.linspace(0, len(rows_sorted) - 1, max_images, dtype=int)
        samples = [rows_sorted[i] for i in indices]

    n = len(samples)
    cols = 5
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.8 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, row in zip(axes, samples):
        img_name = row["image_label"]
        img_path = images_dir / img_name
        if not img_path.exists():
            candidates = list(images_dir.glob(img_name))
            if not candidates:
                ax.axis("off")
                continue
            img_path = candidates[0]

        img = cv2.imread(str(img_path))
        if img is None:
            ax.axis("off")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        manual_cos = float(row["true_cos_theta"])
        manual_sin = float(row["true_sin_theta"])
        pred_cos = float(row["pred_cos_theta"])
        pred_sin = float(row["pred_sin_theta"])
        err = float(row["abs_error_deg"])

        ax.imshow(img_rgb)
        _draw_arrow_inset(ax, manual_cos, manual_sin, pred_cos, pred_sin, err)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  test_predictions.png → {out_path}")

# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Regenerate plots from a run directory (no PyTorch needed)")
    p.add_argument("run_dir", help="Run directory, e.g. runs/angle/rgb_square256")
    p.add_argument("--images-dir", default=None,
                   help="Subimages dir (auto-detect if omitted)")
    p.add_argument("--save-samples", type=int, default=25)
    p.add_argument("--per-kernel", action="store_true",
                   help="Also generate one prediction image per test kernel")
    args = p.parse_args()

    rd = Path(args.run_dir)
    if not rd.is_dir():
        print(f"ERROR: {rd} not found"); sys.exit(1)

    print(f"Run: {rd}")

    # Find images
    images_dir = None
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        candidates = [
            SCRIPT_DIR.parent / "resnet_dataset" / "subimages",
            SCRIPT_DIR.parent / "image_data" / "subimages",
        ]
        for c in candidates:
            if c.is_dir():
                images_dir = c
                break

    # 1
    print("\n[1/4] Training curves …")
    plot_results(rd / "results.csv", rd / "results.png")

    # 2
    print("\n[2/4] Error analysis …")
    plot_error_analysis(rd / "test_predictions.csv", rd / "test_error_analysis.png")

    # 3
    print("\n[3/4] Prediction grid …")
    if images_dir is None:
        print("  SKIP: cannot auto-detect images dir. Use --images-dir")
    else:
        print(f"  Images: {images_dir}")
        plot_prediction_grid(rd / "test_predictions.csv", images_dir,
                             rd / "test_predictions.png",
                             max_images=args.save_samples)

    # 4
    if args.per_kernel:
        print("\n[4/4] Per-kernel predictions …")
        if images_dir is None:
            print("  SKIP: cannot auto-detect images dir. Use --images-dir")
        else:
            plot_per_kernel(rd / "test_predictions.csv", images_dir,
                            rd / "per_kernel")
    else:
        print("\n[4/4] Per-kernel predictions … (use --per-kernel to enable)")

    print("\nDone.")

if __name__ == "__main__":
    main()
