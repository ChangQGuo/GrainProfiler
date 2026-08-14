"""
vae/latent_perturbation_grid.py — Population-median + extreme-kernel perturbation grids.

1. Population-median grid: median latent → outline + 5 perturbation fans.
2. For each latent dimension z_i: find the kernel with the largest |z_i|,
   then produce a 2×3 perturbation grid centred on that kernel's latent vector.

Output (under runs/<run_name>/latent_analysis/):
  latent_perturbation_grid.png           median grid
  extreme_kernels/
    extreme_z1_<label>_perturbation.png  z1 extreme-kernel grid
    ...
    extreme_z5_<label>_perturbation.png  z5 extreme-kernel grid
    extreme_kernels_summary.csv          which kernel was used per dimension
  latent_perturbation_data.csv           raw profiles for median grid

Usage:
  python latent_perturbation_grid.py [run_dir]
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
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

FONT = "Arial"

# ---------------------------------------------------------------------------
# Outline geometry
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


def compute_perturbation(model, z_center, dim_idx, n_steps, device,
                         col_mean, col_std):
    """Perturb z_center[dim_idx] across [-3z, +3z], returning all decoded curves."""
    base_val = float(z_center[dim_idx])
    z_range = np.linspace(-3.0 * abs(base_val), 3.0 * abs(base_val), n_steps)
    if abs(base_val) < 1e-6:
        z_range = np.linspace(-1.0, 1.0, n_steps)

    curves = []
    for val in z_range:
        z_p = z_center.copy()
        z_p[dim_idx] = val
        curves.append(decode_to_profile(model, z_p, device, col_mean, col_std))
    return z_range, np.stack(curves, axis=0)


# ---------------------------------------------------------------------------
# Panel 0: annotated outline (median or extreme kernel)
# ---------------------------------------------------------------------------
def draw_outline_panel(ax, profile, label="", add_colorbar=False, cmap=None, norm=None):
    """Draw a kernel outline with axis.  Optionally place a colorbar below."""
    half_widths = profile / 2.0
    half_widths = np.copy(half_widths)
    half_widths[0] = 0.0
    half_widths[-1] = 0.0
    bottom = np.array([0.0, 0.0])
    top = np.array([100.0, 0.0])
    outline = reconstruct_full_outline(half_widths, bottom, top)

    ax.fill(outline[:, 0], outline[:, 1], alpha=0.72, color="#D8A03D", ec="none")
    ax.plot(outline[:, 0], outline[:, 1], color="#4B3828", linewidth=2.5)
    ax.plot([0, 100], [0, 0], color="#3C3C3C", linewidth=1.8)

    max_ext = max(np.max(half_widths) * 1.30, 2.0)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-max_ext, max_ext)
    ax.set_aspect("equal")
    ax.axis("off")
    if label:
        ax.set_title(label, fontsize=20, fontweight="bold", pad=10, fontfamily=FONT)

    if add_colorbar and cmap is not None and norm is not None:
        # Place a thin horizontal colorbar below the outline
        cbar_ax = ax.inset_axes([0.15, -0.02, 0.70, 0.03])
        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax, orientation="horizontal",
        )
        cb.set_label("Perturbation range", fontsize=14, fontfamily=FONT, fontweight="bold")
        cb.ax.tick_params(labelsize=12)
        cbar_ax.text(0.0, -0.8, "−3z", transform=cbar_ax.transAxes,
                     fontsize=12, ha="center", va="top", fontfamily=FONT)
        cbar_ax.text(1.0, -0.8, "+3z", transform=cbar_ax.transAxes,
                     fontsize=12, ha="center", va="top", fontfamily=FONT)


# ---------------------------------------------------------------------------
# Panels 1-5: perturbation fan
# ---------------------------------------------------------------------------
def draw_perturbation_fan(ax, curves_phys, z_range, dim_idx, original_phys):
    """Draw the perturbation curve fan for one latent dimension."""
    n_steps = len(z_range)
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.viridis
    x_pos = np.arange(1, 101)

    r_min, r_max = z_range[0], z_range[-1]
    z_norm = (z_range - r_min) / (r_max - r_min + 1e-8) * 2 - 1

    for step in range(n_steps):
        color = cmap(norm(z_norm[step]))
        alpha = 0.35 + 0.65 * (step / max(n_steps - 1, 1))
        ax.plot(x_pos, curves_phys[step], color=color, linewidth=2.0, alpha=alpha)

    # Reference profile as dashed black line
    ax.plot(x_pos, original_phys, color="black", linewidth=2.5, linestyle="--",
            label="Reference")

    span = curves_phys.max() - curves_phys.min()
    ax.set_title(f"Latent $z_{{{dim_idx + 1}}}$  (span = {span:.1f})",
                 fontsize=20, fontweight="bold", pad=10, fontfamily=FONT)
    ax.set_xlabel("Position along normalized axis (top → bottom) %",
                  fontsize=16, fontweight="bold", fontfamily=FONT)
    ax.set_ylabel("Full-width (scaled)",
                  fontsize=16, fontweight="bold", fontfamily=FONT)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Range annotation
    ax.text(0.03, 0.95,
            f"$z_{{{dim_idx + 1}}}$ ∈ [{z_range[0]:.2f}, {z_range[-1]:.2f}]",
            transform=ax.transAxes, fontsize=14, va="top", fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------
def build_grid(profiles, all_z_ranges, all_curves_phys, n_dims, output_path,
               title_label, add_colorbar=False):
    """2×3 grid: outline (panel 0) + 5 perturbation fans."""
    if not _HAS_MPL:
        return

    fig, axes = plt.subplots(2, 3, figsize=(21, 15), facecolor="white")
    axes = axes.flatten()

    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.viridis

    draw_outline_panel(axes[0], profiles["ref"],
                       label=title_label,
                       add_colorbar=add_colorbar,
                       cmap=cmap, norm=norm)

    for dim_idx in range(n_dims):
        draw_perturbation_fan(
            axes[dim_idx + 1],
            all_curves_phys[dim_idx],
            all_z_ranges[dim_idx],
            dim_idx,
            profiles["ref"],
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Figure saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Population-median + extreme-kernel latent perturbation grids.")
    parser.add_argument("run_dir", nargs="?", default="runs/profile_vae_latent5")
    parser.add_argument("--n-steps", type=int, default=7,
                        help="Number of perturbation steps (default: 7)")
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
    n_steps = args.n_steps
    print(f"Loaded {len(labels)} plants, {n_dims} latent dims")

    z_median = np.median(Z, axis=0)
    print(f"Population median z: {[f'{v:.4f}' for v in z_median]}")

    out_dir = run_dir / "latent_analysis"
    extreme_dir = out_dir / "extreme_kernels"
    out_dir.mkdir(exist_ok=True)
    extreme_dir.mkdir(exist_ok=True)

    # =====================================================================
    # 1. Median grid
    # =====================================================================
    print("\n--- Median perturbation grid ---")
    profiles = {}
    profiles["ref"] = decode_to_profile(model, z_median, device, col_mean, col_std)

    all_z_ranges, all_curves_phys = [], []
    for dim_idx in range(n_dims):
        z_range, curves_phys = compute_perturbation(
            model, z_median, dim_idx, n_steps, device, col_mean, col_std)
        all_z_ranges.append(z_range)
        all_curves_phys.append(curves_phys)
        print(f"  z_{dim_idx + 1}: span={curves_phys.max() - curves_phys.min():.2f}  "
              f"range=[{z_range[0]:.3f}, {z_range[-1]:.3f}]")

    build_grid(profiles, all_z_ranges, all_curves_phys, n_dims,
               out_dir / "latent_perturbation_grid.png",
               "Median Kernel Shape\n(population median latent)",
               add_colorbar=True)

    # Save median CSV
    _save_csv(out_dir / "latent_perturbation_data.csv",
              profiles, all_z_ranges, all_curves_phys, n_dims, n_steps)

    # =====================================================================
    # 2. Extreme-kernel grids (one per latent dimension)
    # =====================================================================
    extreme_summary = []
    for dim_idx in range(n_dims):
        print(f"\n--- z{dim_idx + 1} extreme-kernel perturbation ---")

        # Find kernel with largest |z_i|
        abs_vals = np.abs(Z[:, dim_idx])
        extreme_idx = int(np.argmax(abs_vals))
        z_extreme = Z[extreme_idx].copy()
        extreme_label = labels[extreme_idx]
        print(f"  Kernel: {extreme_label}  |z_{dim_idx + 1}| = {abs_vals[extreme_idx]:.4f}")
        extreme_summary.append({
            "dim": f"z{dim_idx + 1}",
            "kernel_label": extreme_label,
            "z_value": float(z_extreme[dim_idx]),
            "abs_z_value": float(abs_vals[extreme_idx]),
        })

        profiles_ex = {}
        profiles_ex["ref"] = decode_to_profile(
            model, z_extreme, device, col_mean, col_std)

        all_zr_ex, all_cp_ex = [], []
        for d in range(n_dims):
            zr, cp = compute_perturbation(
                model, z_extreme, d, n_steps, device, col_mean, col_std)
            all_zr_ex.append(zr)
            all_cp_ex.append(cp)

        safe_label = extreme_label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        build_grid(profiles_ex, all_zr_ex, all_cp_ex, n_dims,
                   extreme_dir / f"extreme_z{dim_idx + 1}_{safe_label}.png",
                   f"Extreme Kernel Shape (max |z_{dim_idx + 1}|)\n{extreme_label}",
                   add_colorbar=False)

    # Save extreme-kernel summary
    summary_csv = extreme_dir / "extreme_kernels_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "kernel_label", "z_value", "abs_z_value"])
        writer.writeheader()
        writer.writerows(extreme_summary)
    print(f"\n  Extreme-kernel summary → {summary_csv}")

    print(f"\nDone. All outputs → {out_dir}")


def _save_csv(csv_out, profiles, all_z_ranges, all_curves_phys, n_dims, n_steps):
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "dim", "z_value"]
                        + [f"pos_{i}" for i in range(1, 101)])

        def _w(variant, dim, zv, prof):
            writer.writerow([variant, dim, round(zv, 4)]
                            + [round(float(v), 4) for v in prof])

        _w("ref", "", 0.0, profiles["ref"])
        for dim_idx in range(n_dims):
            for step in range(n_steps):
                _w(f"z{dim_idx + 1}_step{step}", dim_idx + 1,
                   float(all_z_ranges[dim_idx][step]),
                   all_curves_phys[dim_idx][step])
    print(f"  Data saved → {csv_out}")


if __name__ == "__main__":
    main()
