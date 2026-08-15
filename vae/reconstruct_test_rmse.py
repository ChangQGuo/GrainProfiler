"""
vae/reconstruct_test_rmse.py — Per-test-sample reconstruction RMSE: boxplot + table.

For every sample of the held-out TEST split (70/20/10, seed 42 — the exact same
split as train_vae.py), encode → decode → denormalize, then compute the RMSE
(and IoU) between the original and the reconstructed 100-dim full-width profile.

Output (under runs/<run_name>/reconstruction_rmse/):
  test_rmse_per_sample.csv   per-test-sample table: sample_id + RMSE + IoU
  test_rmse_boxplot.png      boxplot of the per-sample RMSE distribution,
                             outliers highlighted in red

Usage:
  python reconstruct_test_rmse.py [run_dir] [--profile <path>] [--device auto]

NOTE: to reproduce the *exact* test split used at training time, --profile must
point to the SAME rep_width_profiles.txt file (same row order) that was used
for training — the split is a fixed-seed shuffle of the row order.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import WidthProfileDataset, split_dataset
from model import ProfileVAE, ProfileVAEConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

FONT = "Arial"
RECON_BLUE = "#2166AC"
RECON_ORANGE = "#D8A03D"
OUTLIER_RED = "#E03030"


def compute_rmse(profile_a, profile_b):
    return float(np.sqrt(np.mean((profile_a - profile_b) ** 2)))


def compute_iou(profile_a, profile_b):
    """IoU of two full-width profiles treated as area under the curve."""
    intersection = np.sum(np.minimum(profile_a, profile_b))
    union = np.sum(np.maximum(profile_a, profile_b))
    return float(intersection / union) if union > 0 else 0.0


def plot_rmse_boxplot(rmse_vals, out_path, test_labels):
    """Vertical boxplot of per-sample RMSE; outliers drawn in red."""
    if not _HAS_MPL:
        return
    rmse = np.asarray(rmse_vals, dtype=float)
    q1, med, q3 = np.percentile(rmse, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.sum((rmse < lo) | (rmse > hi)))
    mean = float(np.mean(rmse))
    std = float(np.std(rmse))

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    bp = ax.boxplot(rmse, patch_artist=True, showmeans=True, widths=0.45,
                    medianprops=dict(color="#B2182B", linewidth=2.4),
                    meanprops=dict(marker="D", markerfacecolor=RECON_BLUE,
                                   markeredgecolor=RECON_BLUE, markersize=8),
                    flierprops=dict(marker="o", markerfacecolor=OUTLIER_RED,
                                    markeredgecolor="#8B1A1A",
                                    markeredgewidth=1.2, markersize=7),
                    whiskerprops=dict(color=RECON_BLUE, linewidth=1.8),
                    capprops=dict(color=RECON_BLUE, linewidth=1.8))
    bp["boxes"][0].set(facecolor="#D8E6F3", edgecolor=RECON_BLUE, linewidth=1.8)

    ax.set_xticks([1])
    ax.set_xticklabels([f"Test set (n = {len(rmse)})"], fontsize=16,
                       fontweight="bold", fontfamily=FONT)
    ax.set_ylabel("RMSE (full-width, scaled)", fontsize=17, fontweight="bold",
                  fontfamily=FONT)
    ax.set_title("Reconstruction RMSE per Test Sample",
                 fontsize=20, fontweight="bold", pad=12, fontfamily=FONT)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.02, 0.98,
            f"mean = {mean:.4f} ± {std:.4f}\n"
            f"median = {med:.4f}\n"
            f"IQR = [{q1:.4f}, {q3:.4f}]\n"
            f"outliers (red) = {n_out} ({100.0 * n_out / len(rmse):.1f}%)",
            transform=ax.transAxes, ha="left", va="top", fontsize=14,
            fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    # Show the worst few samples on the figure
    worst_idx = np.argsort(rmse)[-3:][::-1]
    worst_txt = "worst: " + ", ".join(
        f"{test_labels[i]} ({rmse[i]:.3f})" for i in worst_idx)
    ax.text(0.5, 0.02, worst_txt, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=12, fontfamily=FONT, color="#555555")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Boxplot saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-test-sample reconstruction RMSE: boxplot + table.")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Path to VAE run directory")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to rep_width_profiles.txt (same file/order as training)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = script_dir / "runs_new" / "profile_vae_latent5"
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} not found"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.device != "auto":
        device = torch.device(args.device)

    # ---- Profile file (must match training data & row order) ----
    if args.profile:
        profile_path = Path(args.profile)
    else:
        candidates = [
            script_dir / "data" / "rep_width_profiles.txt",
            script_dir.parent / "HaiNan_results_100images" / "rep_width_profiles.txt",
        ]
        profile_path = next((c for c in candidates if c.is_file()), candidates[0])
    if not profile_path.is_file():
        print(f"ERROR: profile file not found: {profile_path}")
        print("  Pass it explicitly:  --profile <path to rep_width_profiles.txt>")
        sys.exit(1)

    # ---- Load model ----
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    col_mean = ckpt.get("col_mean", None)
    col_std = ckpt.get("col_std", None)
    model = ProfileVAE(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- Rebuild the exact 70/20/10 split (seed 42, same as train_vae.py) ----
    full_ds = WidthProfileDataset(profile_path)
    _, _, test_ds = split_dataset(full_ds)
    n_test = len(test_ds)
    print(f"Run dir     : {run_dir}")
    print(f"Profile     : {profile_path}  ({full_ds.num_samples} samples)")
    print(f"Test split  : {n_test} samples  (seed 42, 10%)")
    print(f"Device      : {device}")

    # ---- Sanity check: profile file should match the training run ----
    args_json = run_dir / "args.json"
    if args_json.is_file():
        import json
        with open(args_json, "r", encoding="utf-8") as f:
            train_meta = json.load(f)
        expected = (train_meta.get("train_samples", 0)
                    + train_meta.get("val_samples", 0)
                    + train_meta.get("test_samples", 0))
        if expected and full_ds.num_samples != expected:
            print(f"WARNING: profile file has {full_ds.num_samples} samples but the "
                  f"training run used {expected}. The test split below is therefore "
                  f"NOT the exact original one — use the same rep_width_profiles.txt "
                  f"(same row order) as training for exact reproduction.")

    # ---- Per-sample encode → decode → RMSE / IoU ----
    rows = []
    rmse_vals = []
    test_labels = []
    with torch.no_grad():
        for i in range(n_test):
            item = test_ds[i]
            x = item["profile"].unsqueeze(0).to(device)
            mu, _ = model.encode(x)
            recon = model.decode(mu).cpu().numpy().squeeze(0)
            orig = item["profile"].numpy()
            if col_mean is not None and col_std is not None:
                orig_phys = orig * col_std.squeeze(0) + col_mean.squeeze(0)
                recon_phys = recon * col_std.squeeze(0) + col_mean.squeeze(0)
            else:
                orig_phys, recon_phys = orig, recon
            rmse = compute_rmse(orig_phys, recon_phys)
            iou = compute_iou(orig_phys, recon_phys)
            rows.append({
                "sample_id": item["label"],
                "RMSE": round(rmse, 6),
                "IoU": round(iou, 6),
            })
            rmse_vals.append(rmse)
            test_labels.append(item["label"])

    rmse_vals = np.asarray(rmse_vals, dtype=float)

    # ---- Summary stats ----
    q1, med, q3 = np.percentile(rmse_vals, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(np.sum((rmse_vals < lo) | (rmse_vals > hi)))
    print(f"\nRMSE over {n_test} test samples:")
    print(f"  mean   = {rmse_vals.mean():.4f} ± {rmse_vals.std():.4f}")
    print(f"  median = {med:.4f}   IQR = [{q1:.4f}, {q3:.4f}]")
    print(f"  outliers (outside 1.5×IQR) = {n_out} ({100.0 * n_out / n_test:.1f}%)")

    # ---- Outputs ----
    out_dir = run_dir / "reconstruction_rmse"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "test_rmse_per_sample.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "RMSE", "IoU"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-sample table → {csv_path}")

    plot_rmse_boxplot(rmse_vals, out_dir / "test_rmse_boxplot.png", test_labels)

    worst = sorted(zip(test_labels, rmse_vals), key=lambda t: t[1], reverse=True)
    print("\nTop-5 worst reconstructions (largest RMSE):")
    for label, v in worst[:5]:
        print(f"  {label}: {v:.4f}")
    print(f"\nDone. All outputs → {out_dir}")


if __name__ == "__main__":
    main()
