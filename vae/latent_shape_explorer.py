"""
vae/latent_shape_explorer.py — Decode latent extremes into annotated kernel outlines.

For each latent dimension, freeze the other 4 at population median, then decode:
  - z_i = min (lowest across all plants) → outline
  - z_i = max (highest across all plants)  → outline

Output (under runs/<run_name>/latent_analysis/):
  latent_shape_grid.png  annotated outlines: median + 5 dims (low→high overlaid)
  latent_shape_data.csv  raw 100-point profiles for each variant
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProfileVAE, ProfileVAEConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ---------------------------------------------------------------------------
# Outline geometry (mirrors pipeline representative_shape.reconstruct_full_outline)
# ---------------------------------------------------------------------------
def reconstruct_full_outline(half_widths, bottom, top):
    """Rebuild symmetric full-kernel outline from half-width profile."""
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
# Annotated outline plot (pipeline-style, with axis, width markers, labels)
# ---------------------------------------------------------------------------
def draw_annotated_outline(ax, profile, color, alpha_fill=0.55, label="",
                           show_measurements=True, edgecolor=None, fill=True,
                           show_quartile_markers=True, max_width_color=None,
                           show_max_width_label=True):
    """Draw one filled kernel outline with axis and width annotations."""
    half_widths = profile / 2.0
    # Force tip endpoints to zero so the outline polygon closes at both ends
    half_widths = np.copy(half_widths)
    half_widths[0] = 0.0
    half_widths[-1] = 0.0
    bottom = np.array([0.0, 0.0])
    top = np.array([100.0, 0.0])
    outline = reconstruct_full_outline(half_widths, bottom, top)

    if fill and alpha_fill > 0:
        ax.fill(outline[:, 0], outline[:, 1], alpha=alpha_fill, color=color, ec="none")
        outline_color = "#4B3828"
    else:
        outline_color = edgecolor if edgecolor else color

    ax.plot(outline[:, 0], outline[:, 1], color=outline_color, linewidth=2.5, alpha=0.95)
    ax.plot([0, 100], [0, 0], color="#3C3C3C", linewidth=1.6, label="Main axis")

    if show_measurements:
        n = len(half_widths)
        w25 = float(half_widths[int(0.25 * (n - 1))])
        w50 = float(half_widths[int(0.50 * (n - 1))])
        w75 = float(half_widths[int(0.75 * (n - 1))])
        hmax = float(np.max(half_widths))
        hmax_idx = int(np.argmax(half_widths))
        hmax_frac = hmax_idx / max(n - 1, 1) * 100.0
        mw_color = max_width_color if max_width_color else "#C45824"

        if show_quartile_markers:
            for pct, w, lc in [(25, w25, "#3C78A8"), (50, w50, "#2457A6"), (75, w75, "#1F7A6D")]:
                ax.plot([pct, pct], [-w, w], color=lc, linestyle="--", linewidth=1.4, alpha=0.7)
                ax.text(pct, -w - max(hmax * 0.12, 0.5), f"{pct}%", color=lc,
                        ha="center", va="top", fontsize=20)

        ax.plot([hmax_frac, hmax_frac], [-hmax, hmax], color=mw_color,
                linewidth=2.2, alpha=0.9)
        if show_max_width_label:
            ax.text(hmax_frac, hmax + max(hmax * 0.08, 0.3), f"{hmax_frac:.0f}%",
                    color=mw_color, ha="center", va="bottom", fontsize=20)

    max_ext = max(np.max(half_widths) * 1.30, 2.0)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-max_ext, max_ext)
    ax.set_aspect("equal")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=20, fontweight="bold", pad=6)


# ---------------------------------------------------------------------------
# Main grid
# ---------------------------------------------------------------------------
def plot_shape_grid(profiles, z_min, z_max, n_dims, output_path):
    """3×2 annotated outline grid: median + 5 dims (grey low→high)."""
    if not _HAS_MPL:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="white")
    axes = axes.flatten()

    # Panel 0: population median shape (warm gold, filled)
    draw_annotated_outline(axes[0], profiles["median"], "#D8A03D", alpha_fill=0.72,
                           label="Median Shape\n(population median latent)")

    # Panels 1-5: low (blue border) vs high (red border) for each dim
    for dim_idx in range(n_dims):
        ax = axes[dim_idx + 1]
        zname = f"z{dim_idx + 1}"

        lo_prof = profiles[f"{zname}_low"]
        hi_prof = profiles[f"{zname}_high"]

        # Low extreme (blue border, no fill, max-width line only, no label)
        draw_annotated_outline(ax, lo_prof, "#2166AC", alpha_fill=0,
                               fill=False, edgecolor="#2166AC",
                               show_measurements=True,
                               show_quartile_markers=False,
                               max_width_color="#2166AC",
                               show_max_width_label=False)

        # High extreme (red border, no fill, max-width line only, no label)
        draw_annotated_outline(ax, hi_prof, "#D62728", alpha_fill=0,
                               fill=False, edgecolor="#D62728", label="",
                               show_measurements=True,
                               show_quartile_markers=False,
                               max_width_color="#D62728",
                               show_max_width_label=False)

        # Re-compute extent to fit both
        lo_hw = lo_prof / 2.0
        hi_hw = hi_prof / 2.0
        max_ext = max(np.max(lo_hw), np.max(hi_hw)) * 1.30
        ax.set_xlim(-5, 105)
        ax.set_ylim(-max(max_ext, 2.0), max(max_ext, 2.0))

        ax.set_title(
            f"{zname}: [{z_min[dim_idx]:.2f} → {z_max[dim_idx]:.2f}]",
            fontsize=20, fontweight="bold", pad=6,
        )

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            plt.Line2D([0], [0], color="#2166AC", linewidth=3, alpha=0.95, label="low"),
            plt.Line2D([0], [0], color="#D62728", linewidth=3, alpha=0.95, label="high"),
        ]
        ax.legend(handles=legend_elements, fontsize=20, frameon=False,
                  loc="upper right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_latent_traits(csv_path: Path):
    labels, vectors = [], []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            labels.append(row["plant_id"])
            vectors.append([float(row[f"latent_{i+1}"]) for i in range(5)])
    return labels, np.array(vectors, dtype=float)


def decode_to_profile(model, z, device, col_mean, col_std):
    z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model.decode(z_t).cpu().numpy().squeeze(0)
    if col_mean is not None and col_std is not None:
        recon = recon * col_std.squeeze(0) + col_mean.squeeze(0)
    return recon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Decode latent extremes into annotated kernel outlines.")
    parser.add_argument("run_dir", nargs="?", default="runs/profile_vae_latent5")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
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

    # Load latents
    latent_csv = run_dir / "latent_traits.csv"
    if not latent_csv.is_file():
        print(f"ERROR: {latent_csv} not found"); sys.exit(1)
    labels, Z = load_latent_traits(latent_csv)
    n_dims = Z.shape[1]
    print(f"Loaded {len(labels)} plants, {n_dims} latent dims")

    z_median = np.median(Z, axis=0)
    z_min = Z.min(axis=0)
    z_max = Z.max(axis=0)
    print(f"Median z: {[f'{v:.4f}' for v in z_median]}")
    for i in range(n_dims):
        print(f"  z{i+1}: min={z_min[i]:.4f}  max={z_max[i]:.4f}  median={z_median[i]:.4f}")

    out_dir = run_dir / "latent_analysis"
    out_dir.mkdir(exist_ok=True)

    # Decode all variants
    profiles = {}
    profiles["median"] = decode_to_profile(model, z_median, device, col_mean, col_std)

    for dim_idx in range(n_dims):
        zname = f"z{dim_idx + 1}"
        z_lo = z_median.copy(); z_lo[dim_idx] = z_min[dim_idx]
        z_hi = z_median.copy(); z_hi[dim_idx] = z_max[dim_idx]
        profiles[f"{zname}_low"] = decode_to_profile(model, z_lo, device, col_mean, col_std)
        profiles[f"{zname}_high"] = decode_to_profile(model, z_hi, device, col_mean, col_std)

    # Plot grid
    plot_shape_grid(profiles, z_min, z_max, n_dims, out_dir / "latent_shape_grid_new.png")

    # Save CSV
    csv_out = out_dir / "latent_shape_data_new.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "dim", "z_value"] + [f"pos_{i}" for i in range(1, 101)])

        def _w(variant, dim, zv, prof):
            writer.writerow([variant, dim, round(zv, 4)] + [round(float(v), 4) for v in prof])

        _w("median", "", 0.0, profiles["median"])
        for dim_idx in range(n_dims):
            zname = f"z{dim_idx + 1}"
            _w(f"{zname}_low", dim_idx + 1, z_min[dim_idx], profiles[f"{zname}_low"])
            _w(f"{zname}_high", dim_idx + 1, z_max[dim_idx], profiles[f"{zname}_high"])
    print(f"  Data saved → {csv_out}")
    print(f"\nDone. All outputs → {out_dir}")


if __name__ == "__main__":
    main()
