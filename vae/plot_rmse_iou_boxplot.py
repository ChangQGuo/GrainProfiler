"""
vae/plot_rmse_iou_boxplot.py — Redraw per-test-sample RMSE & IoU boxplots.

Pure matplotlib/numpy — reads test_rmse_per_sample.csv produced by
reconstruct_test_rmse.py, so it runs locally WITHOUT PyTorch.

Style: Times New Roman, bold, no titles / no in-figure annotations.
The summary statistics (mean / std / quartiles / outlier count ...) are saved
as a CSV instead of being drawn on the figure.

Output (same folder as the CSV):
  test_rmse_iou_boxplot.png    side-by-side RMSE + IoU boxplots
                               (outliers highlighted in red)
  test_rmse_iou_stats.csv      per-metric summary statistics

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

BLUE       = "#2166AC"
BOX_FACE   = "#DCE9F7"
MEDIAN     = "#B2182B"
MEAN       = "#D8A03D"
OUTLIER    = "#E03030"
GRID_COLOR = "#C9C9C9"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
    "font.size": 16,
    "font.weight": "bold",
    "axes.labelsize": 19,
    "axes.labelweight": "bold",
    "xtick.labelsize": 17,
    "ytick.labelsize": 16,
})


def summarize(values):
    """Return a stats dict for one metric (1.5×IQR outlier rule)."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = values[(values < lo) | (values > hi)]
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(q1),
        "median": float(med),
        "p75": float(q3),
        "max": float(np.max(values)),
        "iqr": float(iqr),
        "n_outliers": int(len(outliers)),
        "outlier_pct": float(100.0 * len(outliers) / len(values)),
    }


def draw_panel(ax, values, xlabel, ylabel):
    """One vertical boxplot; outliers drawn in red, no titles / annotations."""
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
    ax.set_xticklabels([xlabel])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # Keep all points inside the axes
    span = float(values.max() - values.min())
    ax.set_ylim(values.min() - 0.12 * span, values.max() + 0.12 * span)


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

    # ---- Summary statistics → CSV ----
    stats = {"RMSE": summarize(rmse), "IoU": summarize(iou)}
    stats_path = csv_path.parent / "test_rmse_iou_stats.csv"
    fields = ["n", "mean", "std", "min", "p25", "median", "p75", "max",
              "iqr", "n_outliers", "outlier_pct"]
    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + fields)
        for metric in ("RMSE", "IoU"):
            s = stats[metric]
            writer.writerow([metric, s["n"],
                             f"{s['mean']:.6f}", f"{s['std']:.6f}",
                             f"{s['min']:.6f}", f"{s['p25']:.6f}",
                             f"{s['median']:.6f}", f"{s['p75']:.6f}",
                             f"{s['max']:.6f}", f"{s['iqr']:.6f}",
                             s["n_outliers"], f"{s['outlier_pct']:.4f}"])
    print(f"Stats table  → {stats_path}")
    for metric in ("RMSE", "IoU"):
        s = stats[metric]
        print(f"  {metric}: mean={s['mean']:.4f}±{s['std']:.4f}  "
              f"median={s['median']:.4f}  outliers={s['n_outliers']} "
              f"({s['outlier_pct']:.1f}%)")

    # ---- Figure (no titles, no in-figure annotations) ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="white")
    draw_panel(axes[0], rmse,
               xlabel=f"Test set (n = {len(rmse)})",
               ylabel="RMSE (full-width, scaled)")
    draw_panel(axes[1], iou,
               xlabel=f"Test set (n = {len(iou)})",
               ylabel="IoU")

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
