"""
vae/reconstruct_samples.py — VAE reconstruction of four representative samples.

For each sample:
  1. Original vs reconstructed profile curve (blue=original, red=VAE)
  2. Reconstructed full-kernel outline with max-width annotation
  3. IoU and RMSE computed between original and reconstructed profiles

Output (under runs/<run_name>/reconstruct_samples/):
  <sample_id>_profile.png    profile overlay curve
  <sample_id>_outline.png    reconstructed outline
  metrics.csv                IoU + RMSE per sample

Usage:
  python reconstruct_samples.py [run_dir]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProfileVAE, ProfileVAEConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

import torch

FONT = "Arial"
BLUE = "#2166AC"
RED  = "#D62728"

TARGET_SAMPLES = [
    "24-HN-795-02",
    "24-HN-172-15",
    "24-HN-086-07",
    "24-HN-673-12",
]

# ---------------------------------------------------------------------------
# Outline
# ---------------------------------------------------------------------------
def reconstruct_full_outline(half_widths, bottom, top):
    axis_vec = top - bottom
    length = float(np.linalg.norm(axis_vec))
    if length < 1e-6:
        return np.array([bottom, top])
    axis_dir = axis_vec / length
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])
    upper, lower = [], []
    n_points = len(half_widths)
    for idx, hw in enumerate(half_widths):
        frac = idx / max(n_points - 1, 1)
        pos = bottom + frac * length * axis_dir
        upper.append(pos + float(hw) * perp_dir)
        lower.append(pos - float(hw) * perp_dir)
    return np.array(upper + lower[::-1])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_iou(profile_a, profile_b):
    """IoU of two full-width profiles treated as area under the curve."""
    intersection = np.sum(np.minimum(profile_a, profile_b))
    union = np.sum(np.maximum(profile_a, profile_b))
    return float(intersection / union) if union > 0 else 0.0


def compute_rmse(profile_a, profile_b):
    return float(np.sqrt(np.mean((profile_a - profile_b) ** 2)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_profile_overlay(original, reconstructed, sample_id, iou, rmse, out_path):
    """Blue = original, red = VAE reconstruction."""
    if not _HAS_MPL:
        return
    x = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")

    ax.plot(x, original, color=BLUE, linewidth=2.5)
    ax.plot(x, reconstructed, color=RED, linewidth=2.2, linestyle="--")
    ax.fill_between(x, original, reconstructed, alpha=0.10, color="#888888")

    ax.set_xlabel("Normalized position along morphological axis (%)",
                  fontsize=16, fontweight="bold", fontfamily=FONT)
    ax.set_ylabel("Full-width (scaled)",
                  fontsize=16, fontweight="bold", fontfamily=FONT)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Metrics annotation
    ax.text(0.97, 0.94, f"IoU = {iou:.4f}\nRMSE = {rmse:.4f}",
            transform=ax.transAxes, fontsize=14, ha="right", va="top",
            fontfamily=FONT, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_title(f"{sample_id}", fontsize=18, fontweight="bold", pad=10, fontfamily=FONT)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_outline(profile, sample_id, out_path):
    """Reconstructed full-kernel outline with max-width annotation."""
    if not _HAS_MPL:
        return
    half_widths = profile / 2.0
    half_widths = np.copy(half_widths)
    half_widths[0] = 0.0
    half_widths[-1] = 0.0
    bottom = np.array([0.0, 0.0])
    top = np.array([100.0, 0.0])
    outline = reconstruct_full_outline(half_widths, bottom, top)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.fill(outline[:, 0], outline[:, 1], alpha=0.60, color="#D8A03D", ec="none")
    ax.plot(outline[:, 0], outline[:, 1], color="#4B3828", linewidth=2.2)
    ax.plot([0, 100], [0, 0], color="#3C3C3C", linewidth=1.8)

    # Max-width annotation
    hw = half_widths
    hmax = float(np.max(hw))
    hmax_idx = int(np.argmax(hw))
    hmax_frac = hmax_idx / max(len(hw) - 1, 1) * 100.0
    ax.plot([hmax_frac, hmax_frac], [-hmax, hmax], color="#C45824",
            linewidth=2.5, alpha=0.85)
    ax.text(hmax_frac, hmax + max(hmax * 0.10, 0.5), f"{hmax_frac:.0f}%",
            color="#C45824", ha="center", va="bottom",
            fontsize=16, fontweight="bold", fontfamily=FONT)

    max_ext = max(np.max(hw) * 1.30, 2.0)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-max_ext, max_ext)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(sample_id, fontsize=18, fontweight="bold", pad=10, fontfamily=FONT)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VAE reconstruction of four representative samples.")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Path to VAE run directory")
    parser.add_argument("--profile", type=str, default=None,
                        help="Path to rep_width_profiles.txt")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Default paths relative to script location (/data/run01/scxj083/SEED_project/vae)
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

    # Load model
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    col_mean = ckpt.get("col_mean", None)
    col_std = ckpt.get("col_std", None)
    model = ProfileVAE(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Load latent traits
    latent_csv = run_dir / "latent_traits.csv"
    with open(latent_csv, "r") as f:
        latent_map = {}
        for row in csv.DictReader(f):
            latent_map[row["plant_id"]] = np.array(
                [float(row[f"latent_{i+1}"]) for i in range(5)], dtype=float)

    # Load original 100-dim profiles
    if args.profile:
        profile_path = Path(args.profile)
    else:
        profile_path = script_dir / "data" / "rep_width_profiles.txt"

    raw = np.loadtxt(str(profile_path), dtype=str, ndmin=2)
    profile_map = {row[0]: row[1:].astype(np.float32) for row in raw}

    out_dir = run_dir / "reconstruct_samples"
    out_dir.mkdir(exist_ok=True)

    metrics_rows = []

    for sample_id in TARGET_SAMPLES:
        if sample_id not in latent_map:
            print(f"WARNING: {sample_id} not found in latent_traits.csv, skipping")
            continue
        if sample_id not in profile_map:
            print(f"WARNING: {sample_id} not found in rep_width_profiles.txt, skipping")
            continue

        print(f"\nProcessing: {sample_id}")

        z = latent_map[sample_id]
        original = profile_map[sample_id]

        # VAE decode (denormalize)
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decode(z_t).cpu().numpy().squeeze(0)
        if col_mean is not None and col_std is not None:
            recon = recon * col_std.squeeze(0) + col_mean.squeeze(0)

        iou = compute_iou(original, recon)
        rmse = compute_rmse(original, recon)
        print(f"  IoU = {iou:.4f}   RMSE = {rmse:.4f}")

        metrics_rows.append({
            "sample_id": sample_id,
            "IoU": round(iou, 4),
            "RMSE": round(rmse, 4),
        })

        safe = sample_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        plot_profile_overlay(original, recon, sample_id, iou, rmse,
                             out_dir / f"{safe}_profile.png")
        plot_outline(recon, sample_id,
                     out_dir / f"{safe}_outline.png")

    # Save metrics
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "IoU", "RMSE"])
        writer.writeheader()
        writer.writerows(metrics_rows)
    print(f"\nMetrics → {out_dir / 'metrics.csv'}")
    print(f"Done. All outputs → {out_dir}")


if __name__ == "__main__":
    main()
