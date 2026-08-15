"""
vae/plot_rmse_iou_boxplot.py — Redraw per-test-sample RMSE & IoU boxplots.

Pure matplotlib/numpy — reads test_rmse_per_sample.csv produced by
reconstruct_test_rmse.py, so it runs locally WITHOUT PyTorch.

Output (same folder as the CSV):
  test_rmse_iou_boxplot.png    side-by-side RMSE + IoU boxplots
                               (outliers highlighted in red, no sample names)

Usage:
  python plot_rmse_iou_boxplot.py
  python plot_rmse_iou_boxplot.py <path to test_rmse_per_sample.csv>
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT = "Arial"
BLUE       = "#2166AC"
BOX_FACE   = "#DCE9F7"
MEDIAN     = "#B2182B"
MEAN       = "#D8A03D"
OUTLIER    = "#E03030"
GRID_COLOR = "#C9C9C9"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "xtick.labelsize": 15,
    "ytick.labelsize": 14,
})


def draw_panel(ax, values, xlabel, ylabel, title):
    """One vertical boxplot; outliers drawn in red, no sample names."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.sum((values < lo) | (values > hi)))
    mean = float(np.mean(values))
    std = float(np.std(values))

    bp = ax.boxplot(
        values, patch_artist=True, showmeans=True, widths=0.42,
        medianprops=dict(color=MEDIAN, linewidth=2.2),
        meanprops=dict(marker="D", markerfacecolor=MEAN,
                       markeredgecolor=MEAN, markersize=8),
        flierprops=dict(marker="o", markerfacecolor=OUTLIER,
                        markeredgecolor="#8B1A1A", markeredgewidth=1.2,
                        markersize=7),
        whiskerprops=dict(color=BLUE, linewidth=1.8),
        capprops=dict(color=BLUE, linewidth=1.8),
    )
    bp["boxes"][0].set(facecolor=BOX_FACE, edgecolor=BLUE, linewidth=1.8)

    ax.set_xticks([1])
    ax.set_xticklabels([xlabel], fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.grid(axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Keep all points inside the axes
    span = float(values.max() - values.min())
    ax.set_ylim(values.min() - 0.12 * span, values.max() + 0.12 * span)

    # Stats annotation
    ax.text(0.02, 0.98,
            f"n = {len(values)}\n"
            f"mean = {mean:.4f} ± {std:.4f}\n"
            f"median = {med:.4f}\n"
            f"IQR = [{q1:.4f}, {q3:.4f}]\n"
            f"outliers (red) = {n_out} ({100.0 * n_out / len(values):.1f}%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))


def main():
    p = argparse.ArgumentParser(
        description="Redraw per-test-sample RMSE & IoU boxplots from CSV (no PyTorch).")
    p.add_argument("csv", nargs="?", default=None,
                   help="Path to test_rmse_per_sample.csv "
                        "(default: runs_new/profile_vae_latent5/reconstruction_rmse/)")
    p.add_argument("--out", default=None, help="Output PNG path")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.csv:
        csv_path = Path(args.csv).resolve()
    else:
        csv_path = (script_dir / "runs_new" / "profile_vae_latent5"
                    / "reconstruction_rmse" / "test_rmse_per_sample.csv")
    if not csv_path.is_file():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("ERROR: CSV is empty"); sys.exit(1)
    rmse = np.array([float(r["RMSE"]) for r in rows])
    iou = np.array([float(r["IoU"]) for r in rows])
    print(f"Loaded {len(rows)} test samples from {csv_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), facecolor="white")
    draw_panel(axes[0], rmse,
               xlabel=f"Test set (n = {len(rmse)})",
               ylabel="RMSE (full-width, scaled)",
               title="Reconstruction RMSE per Test Sample")
    draw_panel(axes[1], iou,
               xlabel=f"Test set (n = {len(iou)})",
               ylabel="IoU",
               title="Reconstruction IoU per Test Sample")

    if args.out:
        out_path = Path(args.out).resolve()
    else:
        out_path = csv_path.parent / "test_rmse_iou_boxplot.png"
    fig.tight_layout(pad=2.5)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    main()
