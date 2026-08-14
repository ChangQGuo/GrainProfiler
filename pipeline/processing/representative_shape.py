"""
processing/representative_shape.py
==================================
Stage 5 - Compute and visualize the representative kernel shape per plant.

READS:
  - masks_binary/      (one SAM binary mask per kernel)
  - axis_results.json  (per-kernel bottom/top axis from Stage 4 — avoids re-running ResNet)
  - metadata.csv       (plant names linked to tray image names)

WRITES:
  - median_outlines.csv
  - rep_width_profiles.txt
  - representative_shapes/
"""

import json
import os
import sys
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from measurements.kernel_metrics import (
    compute_width_profile,
    load_kernels_from_mask_dir,
    width_at_fraction,
)
from utils.calibration import build_image_scale_lookup, get_original_image_name
from utils.visualization import save_circle_deviation_plot


from utils.config import load_config


def normalize_width_profile_to_length(widths, axis_length, target_length=100.0):
    """Scale widths by the same factor that maps each kernel axis to target_length."""
    axis_length = float(axis_length)
    if axis_length < 1e-6:
        return np.asarray(widths, dtype=float)
    return np.asarray(widths, dtype=float) * (float(target_length) / axis_length)


def median_profile(all_profiles_list):
    """Element-wise median profile across all kernels in one plant."""
    return np.median(np.array(all_profiles_list), axis=0)


def smooth_profile(profile, sigma=1.5):
    """Apply Gaussian smoothing to suppress sampling jitter before VAE."""
    return gaussian_filter1d(profile, sigma=sigma, mode='nearest')


def sanitize_filename(text):
    """Return a filesystem-safe filename stem."""
    text = str(text or 'UNKNOWN').strip()
    text = re.sub(r'[\\/:*?"<>|]+', '_', text)
    text = re.sub(r'\s+', '_', text)
    return text.strip('._') or 'UNKNOWN'


def reconstruct_full_outline(half_widths, bottom, top):
    """Rebuild a symmetric full-kernel outline from a representative half-width profile."""
    axis_vec = top - bottom
    length = np.linalg.norm(axis_vec)
    if length == 0:
        return np.array([bottom, top])

    axis_dir = axis_vec / length
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])

    upper_pts = []
    lower_pts = []
    n_points = len(half_widths)

    for idx, half_width in enumerate(half_widths):
        frac = idx / max(n_points - 1, 1)
        pos = bottom + frac * length * axis_dir
        upper_pts.append(pos + float(half_width) * perp_dir)
        lower_pts.append(pos - float(half_width) * perp_dir)

    return np.array(upper_pts + lower_pts[::-1])




def save_stacked_area_plot(fractions, median_half_widths, all_half_widths_list, plant_name, output_path):
    """Save the length-normalized median half-width profile. Crown at 0%, pedicel at 100%."""
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor='white')
    ax.set_facecolor('white')

    x_pos = fractions * 100

    # Individual kernels: crown→pedicel (same direction as median)
    for half_widths in all_half_widths_list:
        ax.plot(x_pos, half_widths,
                color='black', linewidth=0.3, alpha=0.25)

    # Median profile: red border, no fill
    ax.plot(x_pos, median_half_widths, color='red', linewidth=2.2,
            label='Median half-profile')

    max_idx = int(np.argmax(median_half_widths))
    max_frac = x_pos[max_idx]

    for pct, color, label in [(25, '#3c78a8', 'H25'), (50, '#2457a6', 'H50'), (75, '#1f7a6d', 'H75')]:
        idx = int(pct / 100 * (len(median_half_widths) - 1))
        half_width = median_half_widths[idx]
        ax.plot(
            [pct, pct],
            [0, half_width],
            color=color,
            linestyle='--',
            linewidth=1.5,
            label=f'{label}={half_width:.1f}',
        )

    ax.axvline(
        max_frac,
        color='#c45824',
        linewidth=1.8,
        label=f'Max half-width={np.max(median_half_widths):.1f} @ {max_frac:.0f}%',
    )
    ax.axhline(0, color='#777777', linestyle=':', linewidth=0.8)

    ax.set_title(
        f'Length-Normalized Median Half-Width Profile - {plant_name}  '
        f'(n={len(all_half_widths_list)})',
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel('Position along normalized axis (crown→pedicel) %')
    ax.set_ylabel('Median half-width after scaling main axis to 100')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(axis='y', alpha=0.24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_median_outline_plot(fractions, median_half_widths, plant_name, n_kernels, output_path):
    """Save the reconstructed length-normalized median full-kernel outline (crown at left)."""
    # Profile is crown→pedicel.  Place crown at x=0 (left), pedicel at x=100 (right).
    crown_pt   = np.array([0.0, 0.0])
    pedicel_pt = np.array([100.0, 0.0])
    median_outline = reconstruct_full_outline(median_half_widths, crown_pt, pedicel_pt)

    fig, ax = plt.subplots(figsize=(8.2, 5.1), facecolor='#f7f7f4')
    ax.set_facecolor('#fcfbf7')
    ax.fill(
        median_outline[:, 0],
        median_outline[:, 1],
        alpha=0.72,
        color='#d8a03d',
        label='Median outline',
    )
    ax.plot(median_outline[:, 0], median_outline[:, 1], color='#4b3828', linewidth=2)
    ax.plot([0, 100], [0, 0], color='#3c3c3c', linewidth=1.6, label='Main axis')

    max_half_extent = max(float(np.max(median_half_widths)), 1.0)
    y_pad = max(max_half_extent * 0.18, 1.0)
    for pct, color in [(25, '#3c78a8'), (50, '#2457a6'), (75, '#1f7a6d')]:
        idx = int(round((pct / 100.0) * (len(median_half_widths) - 1)))
        half_width = float(median_half_widths[idx])
        ax.plot(
            [pct, pct],
            [-half_width, half_width],
            color=color,
            linestyle='--',
            linewidth=1.5,
            label=f'W{pct} position',
        )
        ax.text(
            pct,
            -half_width - y_pad,
            f'{pct}%',
            color=color,
            ha='center',
            va='top',
            fontsize=8,
        )

    max_idx = int(np.argmax(median_half_widths))
    max_frac = float(fractions[max_idx] * 100.0)
    max_half_width = float(median_half_widths[max_idx])
    ax.plot(
        [max_frac, max_frac],
        [-max_half_width, max_half_width],
        color='#c45824',
        linewidth=2.2,
        label='Max width position',
    )

    ax.set_aspect('equal')
    fig.suptitle(
        f'Length-Normalized Median Kernel Outline\n{plant_name}  (n={n_kernels})',
        fontsize=12,
        y=0.97,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.86),
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=2.6,
        columnspacing=1.6,
    )
    ax.set_xlim(-3, 103)
    ax.set_ylim(-max_half_extent - 3.0 * y_pad, max_half_extent + 2.0 * y_pad)
    ax.axis('off')
    fig.subplots_adjust(top=0.76, bottom=0.05, left=0.02, right=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run(config_path):
    config = load_config(config_path)

    output_dir = config['output']['base_dir']
    metadata_csv = os.path.join(output_dir, config['output']['metadata_csv'])
    median_outlines_csv = os.path.join(
        output_dir,
        config['output'].get('median_outlines_csv', 'median_outlines.csv'),
    )
    rep_dir = os.path.join(output_dir, config['output']['representative_dir'])
    masks_dir = os.path.join(output_dir, config['output']['masks_binary_dir'])
    rep_cfg = config.get('representative_shape', {})
    smooth_enabled = bool(rep_cfg.get('smooth_enabled', True))
    smooth_sigma   = float(rep_cfg.get('smooth_sigma', 1.5))
    skip_round = bool(rep_cfg.get('skip_round_kernels', False))
    round_threshold = float(rep_cfg.get('round_circularity_threshold', 0.95))
    measurement_cfg = config['measurements']
    contour_cfg = config.get('contour_filtering', {})
    n_samples = int(measurement_cfg['num_width_samples'])
    min_contour_area = int(contour_cfg.get('min_contour_area', 500))
    rep_width_profiles_txt = os.path.join(
        output_dir,
        config['output'].get('rep_width_profiles_txt', 'rep_width_profiles.txt'),
    )
    kernel_width_profiles_txt = os.path.join(
        output_dir,
        config['output'].get('kernel_width_profiles_txt', 'kernel_width_profiles.txt'),
    )
    axis_results_json = os.path.join(output_dir, config['output'].get('axis_results_json', 'axis_results.json'))

    os.makedirs(rep_dir, exist_ok=True)

    if not os.path.isdir(masks_dir):
        print(f'ERROR: {masks_dir} not found. Run segmentation/sam_segment.py first.')
        sys.exit(1)

    if not os.path.exists(metadata_csv):
        print(f'ERROR: {metadata_csv} not found. Run ocr/metadata_extraction.py first.')
        sys.exit(1)

    meta_df = pd.read_csv(metadata_csv)
    name_map = {row['image_name']: row['plant_name'] for _, row in meta_df.iterrows()}
    scale_lookup = build_image_scale_lookup(meta_df)

    kernels, skipped_masks = load_kernels_from_mask_dir(
        masks_dir,
        min_area=min_contour_area,
        config=measurement_cfg,
    )
    print(f'Loaded {len(kernels)} kernels directly from binary masks.')
    if skipped_masks > 0:
        print(f'  Skipped {skipped_masks} mask files with unreadable or invalid contours.')

    # Load pre-computed axis data from Stage 4 (avoids re-running ResNet)
    if not os.path.exists(axis_results_json):
        print(f'ERROR: {axis_results_json} not found. Run measurements/kernel_metrics.py first.')
        sys.exit(1)
    with open(axis_results_json, 'r') as f:
        axis_list = json.load(f)
    axis_lookup = {item['kernel_name']: item for item in axis_list}
    print(f'Loaded {len(axis_lookup)} axis records from Stage 4.')

    plant_groups = {}
    for kernel in kernels:
        kernel_name = kernel['image_name']
        orig_name = get_original_image_name(kernel_name)
        plant_name = name_map.get(orig_name, 'UNKNOWN')
        plant_groups.setdefault(plant_name, []).append({
            'kernel_name': kernel_name,
            'orig_name': orig_name,
            'contour': np.asarray(kernel['contour'], dtype=float),
        })

    print(f'Found {len(plant_groups)} plants, computing representative shapes...')
    summary_records = []
    profile_lines = []  # for rep_width_profiles.txt (PCA input)
    kernel_profile_lines = []  # per-kernel width profiles for GUI interactive hover

    for plant_name, kernels_for_plant in plant_groups.items():
        print(f'\n  Plant: {plant_name}  ({len(kernels_for_plant)} kernels)')
        # Use a filesystem-safe name for the directory; keep the raw plant_name in CSV/VAE outputs
        plant_dir = os.path.join(rep_dir, sanitize_filename(plant_name))
        os.makedirs(plant_dir, exist_ok=True)

        # Raw full-width profiles feed px/mm summaries; normalized half-width profiles feed representative shapes.
        raw_width_profiles_px = []
        normalized_half_width_profiles = []
        total_lengths = []
        max_widths = []
        areas = []
        perimeters = []
        circularities = []
        width_profiles_mm = []
        lengths_mm = []
        max_widths_mm = []
        areas_mm2 = []
        perimeters_mm = []

        # Dominant physical scale for this plant: used to fill mm_per_px for any
        # kernel whose own image had failed calibration, so the px and mm median
        # profiles are computed over the SAME set of kernels (previously such
        # plants ended up with empty mm columns because the counts mismatched).
        plant_scales = []
        for k in kernels_for_plant:
            s = scale_lookup.get(k['orig_name'], {}).get('mm_per_px')
            if s is not None:
                plant_scales.append(float(s))
        dominant_mm_per_px = float(np.median(plant_scales)) if plant_scales else None

        for kernel in kernels_for_plant:
            kernel_name = kernel['kernel_name']
            orig_name = kernel['orig_name']
            contour = kernel['contour']

            # Read pre-computed axis data from Stage 4
            axis_data = axis_lookup.get(kernel_name)
            if axis_data is None:
                print(f'    WARNING: No axis data for {kernel_name}, skipping')
                continue

            axis_length = float(axis_data['axis_length'])
            if axis_length < 1.0:
                continue

            # Skip round kernels when enabled (config: representative_shape.skip_round_kernels)
            if skip_round:
                circ = axis_data.get('circularity')
                if circ is not None and float(circ) > round_threshold:
                    continue

            # axis_data['top'] = crown (flat), axis_data['bottom'] = pedicel (sharp)
            # Pass (crown, pedicel) so the profile is built crown→pedicel
            crown_pt   = np.asarray(axis_data['top'], dtype=float)
            pedicel_pt = np.asarray(axis_data['bottom'], dtype=float)

            fractions, widths, half_widths, _, _ = compute_width_profile(
                contour,
                crown_pt,
                pedicel_pt,
                axis_length,
                n_samples,
            )
            normalized_half_widths = normalize_width_profile_to_length(
                half_widths,
                axis_length,
                target_length=100.0,
            )
            # Skip degenerate profiles where many width samples failed (perpendicular
            # intersection returned 0). Their zeros would bias the plant median.
            widths_arr = np.asarray(widths, dtype=float)
            if widths_arr.size > 0 and float(np.mean(widths_arr <= 0)) > 0.1:
                print(f'    WARNING: {kernel_name} degenerate profile, skipping')
                continue
            raw_width_profiles_px.append(widths)
            normalized_half_width_profiles.append(normalized_half_widths)
            kernel_profile_lines.append(kernel_name + " " + " ".join(f"{w:.2f}" for w in widths))
            total_lengths.append(axis_length)

            max_width_px = float(axis_data['max_width'])
            max_widths.append(max_width_px)

            area = float(axis_data['area'])
            perimeter = float(axis_data['perimeter'])
            areas.append(area)
            perimeters.append(perimeter)
            circularities.append(float(axis_data['circularity']))

            scale_info = scale_lookup.get(orig_name, {})
            mm_per_px = scale_info.get('mm_per_px')
            if mm_per_px is None:
                mm_per_px = dominant_mm_per_px  # fill missing scale to keep px/mm aligned

            if mm_per_px is not None:
                mm_per_px = float(mm_per_px)
                width_profiles_mm.append(widths * mm_per_px)
                lengths_mm.append(axis_length * mm_per_px)
                max_widths_mm.append(float(max_width_px) * mm_per_px)
                areas_mm2.append(float(area) * mm_per_px * mm_per_px)
                perimeters_mm.append(float(perimeter) * mm_per_px)

        if not raw_width_profiles_px:
            print(f'  WARNING: No valid measured kernels for {plant_name}')
            continue

        fractions = np.linspace(0, 1, n_samples)
        representative_widths_px = median_profile(raw_width_profiles_px)
        median_half_widths_smoothed = median_profile(normalized_half_width_profiles)

        if smooth_enabled:
            # Gaussian-smooth interior to suppress sampling jitter.
            # First/last points have no neighbour on one side and would be
            # distorted by padding artefacts — save and restore them.
            tip_crown = median_half_widths_smoothed[0]
            tip_pedicel = median_half_widths_smoothed[-1]
            median_half_widths_smoothed = smooth_profile(median_half_widths_smoothed, sigma=smooth_sigma)
            median_half_widths_smoothed[0] = tip_crown
            median_half_widths_smoothed[-1] = tip_pedicel

        # Profile is already crown→pedicel (compute_width_profile was called with
        # crown_pt as 'bottom' and pedicel_pt as 'top').  No reversal needed.

        # Build one line for rep_width_profiles.txt: plant_name + 100 smoothed full-width values
        full_widths = 2.0 * median_half_widths_smoothed
        profile_line = plant_name + " " + " ".join(f"{w:.2f}" for w in full_widths)
        profile_lines.append(profile_line)

        median_length = float(np.median(total_lengths))
        w25_px = float(representative_widths_px[int(0.25 * (n_samples - 1))])
        w50_px = float(representative_widths_px[int(0.50 * (n_samples - 1))])
        w75_px = float(representative_widths_px[int(0.75 * (n_samples - 1))])
        median_h25 = float(median_half_widths_smoothed[int(0.25 * (n_samples - 1))])
        median_h50 = float(median_half_widths_smoothed[int(0.50 * (n_samples - 1))])
        median_h75 = float(median_half_widths_smoothed[int(0.75 * (n_samples - 1))])
        median_hmax = float(np.max(median_half_widths_smoothed))
        median_hmax_pct = float(fractions[int(np.argmax(median_half_widths_smoothed))] * 100.0)
        median_w25_norm = 2.0 * median_h25
        median_w50_norm = 2.0 * median_h50
        median_w75_norm = 2.0 * median_h75
        median_wmax_norm = 2.0 * median_hmax
        median_mb = float(np.median(max_widths))
        median_area = float(np.median(areas))
        median_perimeter = float(np.median(perimeters))
        median_circularity = float(np.median(circularities))

        if len(width_profiles_mm) == len(raw_width_profiles_px) and width_profiles_mm:
            median_widths_mm = median_profile(width_profiles_mm)
            median_length_mm = float(np.median(lengths_mm))
            w25_mm = float(median_widths_mm[int(0.25 * (n_samples - 1))])
            w50_mm = float(median_widths_mm[int(0.50 * (n_samples - 1))])
            w75_mm = float(median_widths_mm[int(0.75 * (n_samples - 1))])
            median_mb_mm = float(np.median(max_widths_mm))
            median_area_mm2 = float(np.median(areas_mm2))
            median_perimeter_mm = float(np.median(perimeters_mm))
        else:
            median_length_mm = None
            w25_mm = None
            w50_mm = None
            w75_mm = None
            median_mb_mm = None
            median_area_mm2 = None
            median_perimeter_mm = None

        source_images = sorted(set(k['orig_name'] for k in kernels_for_plant))
        summary_records.append({
            'plant_name': plant_name,
            'source_images': ';'.join(source_images),
            'n_kernels': len(raw_width_profiles_px),
            'median_length_px': round(median_length, 2),
            'median_length_mm': round(median_length_mm, 4) if median_length_mm is not None else '',
            'median_max_width_px': round(median_mb, 2),
            'median_max_width_mm': round(median_mb_mm, 4) if median_mb_mm is not None else '',
            'median_width_25pct_px': round(w25_px, 2),
            'median_width_25pct_mm': round(w25_mm, 4) if w25_mm is not None else '',
            'median_width_50pct_px': round(w50_px, 2),
            'median_width_50pct_mm': round(w50_mm, 4) if w50_mm is not None else '',
            'median_width_75pct_px': round(w75_px, 2),
            'median_width_75pct_mm': round(w75_mm, 4) if w75_mm is not None else '',
            'representative_axis_length_norm': 100.0,
            'median_width_25pct_norm': round(median_w25_norm, 2),
            'median_width_50pct_norm': round(median_w50_norm, 2),
            'median_width_75pct_norm': round(median_w75_norm, 2),
            'median_max_width_norm': round(median_wmax_norm, 2),
            'median_max_width_pct': round(median_hmax_pct, 2),
            'median_area_px2': round(median_area, 2),
            'median_area_mm2': round(median_area_mm2, 4) if median_area_mm2 is not None else '',
            'median_perimeter_px': round(median_perimeter, 2),
            'median_perimeter_mm': round(median_perimeter_mm, 4) if median_perimeter_mm is not None else '',
            'median_circularity': round(median_circularity, 4),
        })

        path_area = os.path.join(plant_dir, f'{plant_name}_stacked_area.png')
        save_stacked_area_plot(
            fractions,
            median_half_widths_smoothed,
            normalized_half_width_profiles,
            plant_name,
            path_area,
        )
        print(f'    Saved: {os.path.basename(path_area)}')

        path_outline = os.path.join(plant_dir, f'{plant_name}_median_outline.png')
        save_median_outline_plot(
            fractions,
            median_half_widths_smoothed,
            plant_name,
            len(raw_width_profiles_px),
            path_outline,
        )
        print(f'    Saved: {os.path.basename(path_outline)}')

        # Circle deviation — uses the plant-level median representative outline
        median_outline = reconstruct_full_outline(
            median_half_widths_smoothed,
            np.array([0.0, 0.0]),      # bottom at axis origin
            np.array([100.0, 0.0]),     # top at axis length 100
        )
        centroid_ex = np.array([50.0, 0.0])  # midpoint of the symmetric outline
        bottom_ex = np.array([0.0, 0.0])
        top_ex = np.array([100.0, 0.0])
        path_circle = os.path.join(plant_dir, f'{plant_name}_circle_deviation.png')
        save_circle_deviation_plot(
            median_outline,
            centroid_ex,
            bottom_ex,
            top_ex,
            plant_name,
            path_circle,
        )
        print(f'    Saved: {os.path.basename(path_circle)}')

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(median_outlines_csv, index=False)
    print(f'\nRepresentative shape summaries saved for {len(summary_records)} plants -> {median_outlines_csv}')
    print('Visualization plots saved to:', rep_dir)

    if profile_lines:
        with open(rep_width_profiles_txt, 'w') as f:
            for line in profile_lines:
                f.write(line + '\n')
        print(f'Width profiles for PCA saved: {len(profile_lines)} plants -> {rep_width_profiles_txt}')

    if kernel_profile_lines:
        with open(kernel_width_profiles_txt, 'w') as f:
            for line in kernel_profile_lines:
                f.write(line + '\n')
        print(f'Per-kernel width profiles saved: {len(kernel_profile_lines)} kernels -> {kernel_width_profiles_txt}')


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run(config_path)
