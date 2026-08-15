"""
vae/interpret_latents.py — Latent space interpretability experiment.

For each selected kernel, perturb z_i across [-3z, +3z] (7 steps) while
holding the other 4 dimensions fixed.  Repeating across multiple kernels
reveals whether a latent dimension is collapsed (span ≈ 0 for all kernels)
or genuinely captures shape variation.

Output (under runs/<run_name>/latent_analysis/):
  kernel_<label>_disturbance.png   one 2×3 traversal grid per kernel
  kernel_<label>_disturbance.csv   raw perturbation curves per kernel
  summary_span.png                 curve-span bar chart across kernels × dims
  summary_span.csv                 per-kernel × per-dim span table (collapse check)

Usage:
  python interpret_latents.py [run_dir] --profile <path> --n-kernels 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ProfileVAE, ProfileVAEConfig

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

RECON_BLUE = '#2166AC'
RECON_ORANGE = '#D8A03D'


def load_run(run_dir: Path, device: torch.device):
    """Load model, config, and standardisation params from a training run."""
    ckpt_path = run_dir / 'best.pt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt['config']
    col_mean = ckpt.get('col_mean', None)
    col_std = ckpt.get('col_std', None)
    model = ProfileVAE(config).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, config, col_mean, col_std


def encode_all(ds, model, device):
    """Encode all samples → (Z, labels, original_standardised_profiles)."""
    all_z, all_x, all_labels = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            x = item['profile'].unsqueeze(0).to(device)
            mu, _ = model.encode(x)
            all_z.append(mu.cpu().numpy().squeeze(0))
            all_x.append(item['profile'].numpy())
            all_labels.append(item['label'])
    return np.stack(all_z, axis=0), all_labels, np.stack(all_x, axis=0)


def select_kernels(Z, labels, n_kernels, seed=42):
    """Return indices for N kernels: 1 median + (N-1) random."""
    rng = np.random.default_rng(seed)
    n_total = len(labels)
    n_kernels = min(n_kernels, n_total)

    # 1) Closest to latent median
    z_median = np.median(Z, axis=0)
    dists = np.sqrt(np.sum((Z - z_median) ** 2, axis=1))
    median_idx = int(np.argmin(dists))

    # 2) Random remainder (exclude the median pick)
    pool = [i for i in range(n_total) if i != median_idx]
    random_indices = rng.choice(pool, size=n_kernels - 1, replace=False).tolist()

    all_idx = [median_idx] + sorted(random_indices)
    print(f'Selected {len(all_idx)} kernels: '
          f'1 median ({labels[median_idx]}) + {len(random_indices)} random')
    return all_idx


def perturb_and_decode(model, z_center, dim_idx, n_steps, device,
                       col_mean=None, col_std=None):
    """Perturb z_center[dim_idx] across [-3z, +3z], decode all variants."""
    base_val = float(z_center[dim_idx])
    z_range = np.linspace(-3.0 * abs(base_val), 3.0 * abs(base_val), n_steps)
    if abs(base_val) < 1e-6:
        z_range = np.linspace(-1.0, 1.0, n_steps)

    curves = []
    for dz in z_range:
        z_perturbed = z_center.copy()
        z_perturbed[dim_idx] = dz
        z_t = torch.from_numpy(z_perturbed).float().unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model.decode(z_t).cpu().numpy().squeeze(0)
        curves.append(recon)

    curves_raw = np.stack(curves, axis=0)
    if col_mean is not None and col_std is not None:
        curves_phys = curves_raw * col_std.squeeze(0) + col_mean.squeeze(0)
    else:
        curves_phys = curves_raw
    return z_range, curves_raw, curves_phys


def plot_traversal_grid(original_phys, all_ranges, all_curves_phys,
                        label, output_path):
    """2×3 grid: 5 latent perturbation subplots + 1 legend."""
    if not _HAS_MPL:
        return
    n_dims = len(all_ranges)
    n_steps = len(all_ranges[0])
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
    axes = axes.flatten()

    norm = mcolors.Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.viridis
    x_pos = np.arange(1, 101)

    for dim_idx in range(n_dims):
        ax = axes[dim_idx]
        z_range = all_ranges[dim_idx]
        curves = all_curves_phys[dim_idx]
        r_min, r_max = z_range[0], z_range[-1]
        z_norm = (z_range - r_min) / (r_max - r_min + 1e-8) * 2 - 1

        for step in range(n_steps):
            color = cmap(norm(z_norm[step]))
            alpha = 0.35 + 0.65 * (step / max(n_steps - 1, 1))
            ax.plot(x_pos, curves[step], color=color, linewidth=1.6, alpha=alpha)

        ax.plot(x_pos, original_phys, color=RECON_BLUE, linewidth=2.2,
                linestyle='--', label='Original', zorder=10)
        mid = n_steps // 2
        ax.plot(x_pos, curves[mid], color=RECON_ORANGE, linewidth=2.0,
                label='Reconstructed (z=0)', zorder=9)

        span = curves.max() - curves.min()
        ax.set_title(f'Latent $z_{{{dim_idx + 1}}}$  (span={span:.1f})',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Normalized position along morphological axis (%)', fontsize=9)
        ax.set_ylabel('Full-width (scaled)', fontsize=9)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.03, 0.95,
                f'$z_{{{dim_idx + 1}}}$ ∈ [{z_range[0]:.2f}, {z_range[-1]:.2f}]',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Legend panel
    ax = axes[5]
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', extent=[2, 8, 6, 6.3], cmap=cmap)
    ax.text(5, 6.8, 'Perturbation range', ha='center', fontsize=9, fontweight='bold')
    ax.text(1.5, 6.15, '−3z', ha='center', fontsize=8)
    ax.text(8.5, 6.15, '+3z', ha='center', fontsize=8)
    ax.plot([2, 3], [5, 5], color=RECON_BLUE, linewidth=2.2, linestyle='--')
    ax.text(3.3, 5, 'Original profile', fontsize=9, va='center')
    ax.plot([2, 3], [4.3, 4.3], color=RECON_ORANGE, linewidth=2.0)
    ax.text(3.3, 4.3, 'Reconstructed (z=0)', fontsize=9, va='center')
    ax.plot([2, 3], [3.6, 3.6], color=cmap(0.6), linewidth=1.6, alpha=0.7)
    ax.text(3.3, 3.6, 'Perturbed reconstructions', fontsize=9, va='center')
    ax.text(5, 2.5, 'Sample', ha='center', fontsize=10, fontweight='bold')
    ax.text(5, 1.8, label, ha='center', fontsize=9, color='#555555')

    fig.suptitle(f'Latent Space Traversal — {label}',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_summary_span(span_matrix, labels, output_path):
    """Bar chart: per-dimension curve span across all selected kernels."""
    if not _HAS_MPL:
        return
    n_kernels, n_dims = span_matrix.shape
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='white')
    x = np.arange(n_dims)
    width = 0.8 / n_kernels
    colors = plt.cm.tab10.colors

    for k in range(n_kernels):
        offset = (k - n_kernels / 2 + 0.5) * width
        ax.bar(x + offset, span_matrix[k], width, color=colors[k % 10],
                      alpha=0.85, edgecolor='white', linewidth=0.3,
                      label=labels[k] if k < 12 else None)

    # Highlight collapsed dims (span < 0.5 threshold)
    dim_medians = np.median(span_matrix, axis=0)
    collapsed = np.where(dim_medians < 0.5)[0]
    for dim_idx in collapsed:
        ax.axvspan(dim_idx - 0.4, dim_idx + 0.4, alpha=0.15, color='red', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f'$z_{{{i+1}}}$' for i in range(n_dims)], fontsize=12)
    ax.set_ylabel('Curve span', fontsize=11)
    ax.set_title('Latent Dimension Activity — Curve Span Across Kernels\n'
                 '(red band = potentially collapsed, median span < 0.5)',
                 fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if n_kernels <= 12:
        ax.legend(fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Summary figure saved → {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Latent interpretability: perturb each z_i across N kernels.')
    parser.add_argument('run_dir', nargs='?',
                        default='runs/profile_vae_latent5')
    parser.add_argument('--n-kernels', type=int, default=10,
                        help='Number of kernels to analyse (1 median + N-1 random)')
    parser.add_argument('--n-steps', type=int, default=7,
                        help='Number of perturbation steps (default: 7)')
    parser.add_argument('--profile', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f'ERROR: run directory not found: {run_dir}')
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device != 'auto':
        device = torch.device(args.device)

    if args.profile:
        profile_path = args.profile
    else:
        args_json = run_dir / 'args.json'
        if args_json.is_file():
            with open(args_json, 'r') as f:
                profile_path = json.load(f).get('profile_path', '')
        else:
            profile_path = '/data/home/scxj083/run/SEED_project/vae/data/rep_width_profiles.txt'

    # ---- Output subfolder ----
    out_dir = run_dir / 'latent_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Run dir     : {run_dir}')
    print(f'Profile path: {profile_path}')
    print(f'Device      : {device}')
    print(f'Output      : {out_dir}')
    print(f'Kernels     : {args.n_kernels}')

    # ---- Load data + model ----
    from dataset import WidthProfileDataset
    profile_path = Path(profile_path)
    if not profile_path.is_absolute():
        profile_path = (run_dir.parent.parent / profile_path).resolve()
    ds = WidthProfileDataset(profile_path)
    model, config, col_mean, col_std = load_run(run_dir, device)

    Z, labels, all_x = encode_all(ds, model, device)
    n_dims = config.latent_dim
    print(f'Loaded {len(labels)} samples, {n_dims} latent dims')

    # ---- Select kernels ----
    selected_idx = select_kernels(Z, labels, args.n_kernels)

    # ---- Analyse each kernel ----
    span_matrix = np.zeros((len(selected_idx), n_dims))

    for ki, idx in enumerate(selected_idx):
        label = labels[idx]
        z_center = Z[idx]
        x_orig = all_x[idx]
        orig_phys = (x_orig * col_std.squeeze(0) + col_mean.squeeze(0)
                   if col_mean is not None else x_orig)

        safe_label = label.replace('/', '_').replace('\\', '_').replace(' ', '_')
        print(f'\n[{ki + 1}/{len(selected_idx)}] {label}  '
              f'z = [{", ".join(f"{v:.3f}" for v in z_center)}]')

        all_ranges, all_curves_phys = [], []
        for dim_idx in range(n_dims):
            z_range, _, curves_phys = perturb_and_decode(
                model, z_center, dim_idx, args.n_steps, device,
                col_mean=col_mean, col_std=col_std,
            )
            all_ranges.append(z_range)
            all_curves_phys.append(curves_phys)
            span = curves_phys.max() - curves_phys.min()
            span_matrix[ki, dim_idx] = span
            flag = '  ← COLLAPSED?' if span < 0.5 else ''
            print(f'  z_{dim_idx + 1}: span={span:.2f}  '
                  f'range=[{z_range[0]:.3f}, {z_range[-1]:.3f}]{flag}')

        # Individual kernel figure
        plot_traversal_grid(
            orig_phys, all_ranges, all_curves_phys, label,
            out_dir / f'kernel_{safe_label}_disturbance.png',
        )

        # Individual kernel CSV
        with open(out_dir / f'kernel_{safe_label}_disturbance.csv', 'w',
                  newline='') as f:
            writer = csv.writer(f)
            header = ['dim', 'perturb_step', 'z_value'] + \
                     [f'pos_{i}' for i in range(1, 101)]
            writer.writerow(header)
            for dim_idx in range(n_dims):
                for step in range(args.n_steps):
                    row = [dim_idx + 1, step,
                           round(float(all_ranges[dim_idx][step]), 4)]
                    row += [round(float(v), 4) for v in all_curves_phys[dim_idx][step]]
                    writer.writerow(row)

    # ---- Summary: span matrix ----
    summary_csv = out_dir / 'summary_span.csv'
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['kernel_label'] +
                         [f'z_{i + 1}_span' for i in range(n_dims)])
        for ki, idx in enumerate(selected_idx):
            writer.writerow([labels[idx]] +
                            [round(float(span_matrix[ki, j]), 3)
                             for j in range(n_dims)])

    # Summary figure
    short_labels = [labels[i].replace(' ', '\n') for i in selected_idx]
    plot_summary_span(span_matrix, short_labels,
                      out_dir / 'summary_span.png')

    # Collapse report
    dim_medians = np.median(span_matrix, axis=0)
    print(f'\n{"=" * 55}')
    print('Collapse check (median span across kernels):')
    for i in range(n_dims):
        status = 'COLLAPSED (< 0.5)' if dim_medians[i] < 0.5 else 'OK'
        worst = span_matrix[:, i].min()
        print(f'  z_{i + 1}: median={dim_medians[i]:.2f}  '
              f'min={worst:.2f}  → {status}')
    print(f'{"=" * 55}')
    print(f'\nAll outputs → {out_dir}')


if __name__ == '__main__':
    main()
