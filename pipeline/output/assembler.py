"""
output/assembler.py
===================
Stage 7 - Export human-friendly result tables.

READS:
  - measurements.csv        (per-kernel morphological data, Stage 4)
  - median_outlines.csv     (plant-level median shape summaries, Stage 5)
  - metadata.csv            (plant name + weight per image, Stage 1)

WRITES:
  - final_output_individual.csv   (per-kernel table with metadata)
  - final_output_plant_median.csv (plant-level median table)
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calibration import get_original_image_name


DIAGNOSTIC_COLUMNS_TO_DROP = {
    'bottom_source',
    'axis_bottom_x_px',
    'axis_bottom_y_px',
    'axis_top_x_px',
    'axis_top_y_px',
    'skip_local_refine',
    'bottom_refine_axis_angle_gap_deg',
    'bottom_refine_radial_distance_ratio',
    'axis_selection_circularity',
    'axis_selection_reason',
    'axis_score',
    'candidate_count',
    'candidate_score_margin',
    'area_balance_penalty',
    'width_symmetry_penalty',
    'color_symmetry_penalty',
    'length_penalty',
    'endpoint_penalty',
    'round_circularity_threshold',
    'round_circularity_ok',
    'round_filter_enabled',
    'low_confidence_chord_moments_max_angle_deg',
}
DIAGNOSTIC_COLUMN_PREFIXES = (
    'bottom_roughness_',
    'moments_axis_',
    'longest_chord_',
    'symmetric_chord_',
    'round_spoke_',
    'round_color_',
)


from utils.config import load_config


def select_columns(df, columns):
    """Return only the requested columns that exist in the DataFrame."""
    selected = [col for col in columns if col in df.columns]
    return df.loc[:, selected]


def reorder_individual_columns(df):
    """Keep only analysis-facing per-kernel phenotype columns."""
    diagnostic_cols = [
        col for col in df.columns
        if col in DIAGNOSTIC_COLUMNS_TO_DROP
        or any(str(col).startswith(prefix) for prefix in DIAGNOSTIC_COLUMN_PREFIXES)
    ]
    df = df.drop(columns=diagnostic_cols, errors='ignore')
    keep_columns = [
        'kernel_name',
        'plant_name',
        'weight_g',
        'length_px',
        'length_mm',
        'max_width_px',
        'max_width_mm',
        'width_25pct_px',
        'width_25pct_mm',
        'width_50pct_px',
        'width_50pct_mm',
        'length_width_ratio',
        'eccentricity',
        'width_75pct_px',
        'width_75pct_mm',
        'area_px2',
        'area_mm2',
        'perimeter_px',
        'perimeter_mm',
        'circularity',
    ]
    return select_columns(df, keep_columns)


def reorder_plant_median_columns(df):
    """Keep only analysis-facing plant-level phenotype columns."""
    keep_columns = [
        'plant_name',
        'weight_g',
        'n_kernels',
        'median_length_px',
        'median_length_mm',
        'median_max_width_px',
        'median_max_width_mm',
        'median_width_25pct_px',
        'median_width_25pct_mm',
        'median_width_50pct_px',
        'median_width_50pct_mm',
        'median_width_75pct_px',
        'median_width_75pct_mm',
        'representative_axis_length_norm',
        'median_width_25pct_norm',
        'median_width_50pct_norm',
        'median_width_75pct_norm',
        'median_max_width_norm',
        'median_max_width_pct',
        'median_area_px2',
        'median_area_mm2',
        'median_perimeter_px',
        'median_perimeter_mm',
        'median_circularity',
        'weight_per_kernel_g',
    ]
    return select_columns(df, keep_columns)


def run(config_path):
    config = load_config(config_path)

    output_dir = config['output']['base_dir']
    measurements_csv = os.path.join(output_dir, config['output']['measurements_csv'])
    metadata_csv = os.path.join(output_dir, config['output']['metadata_csv'])
    median_outlines_csv = os.path.join(
        output_dir,
        config['output'].get('median_outlines_csv', 'median_outlines.csv'),
    )
    final_individual_csv = os.path.join(
        output_dir,
        config['output'].get('final_individual_csv', 'final_output_individual.csv'),
    )
    final_plant_median_csv = os.path.join(
        output_dir,
        config['output'].get('final_plant_median_csv', 'final_output_plant_median.csv'),
    )

    if not os.path.exists(measurements_csv):
        print(f'ERROR: {measurements_csv} not found. Run measurements/kernel_metrics.py first.')
        sys.exit(1)
    if not os.path.exists(metadata_csv):
        print(f'ERROR: {metadata_csv} not found. Run ocr/metadata_extraction.py first.')
        sys.exit(1)

    meas_df = pd.read_csv(measurements_csv)
    meta_df = pd.read_csv(metadata_csv)

    # --- Individual kernel table ---
    meta_lookup = {
        row['image_name']: (row['plant_name'], str(row['weight_g']))
        for _, row in meta_df.iterrows()
    }

    plant_names = []
    weights = []
    for _, row in meas_df.iterrows():
        orig = get_original_image_name(row['kernel_name'])
        plant_name, weight_g = meta_lookup.get(orig, ('UNKNOWN', 'UNKNOWN'))
        plant_names.append(plant_name)
        weights.append(weight_g)

    individual_df = meas_df.copy()
    individual_df['plant_name'] = plant_names
    individual_df['weight_g'] = weights
    individual_df = individual_df.fillna('')
    individual_df = reorder_individual_columns(individual_df)
    individual_df.to_csv(final_individual_csv, index=False)

    print(f'Individual output saved -> {final_individual_csv}')
    print(f'  Individual kernel rows : {len(individual_df)}')

    # --- Plant median table ---
    if not os.path.exists(median_outlines_csv):
        print(f'WARNING: {median_outlines_csv} not found, skipping plant median output.')
        print(f'  Run processing/representative_shape.py first.')
        return

    plant_weight_map = {}
    for _, row in meta_df.iterrows():
        pn = str(row.get('plant_name', '')).strip()
        if pn and pn not in plant_weight_map:
            plant_weight_map[pn] = str(row.get('weight_g', 'UNKNOWN'))

    plant_df = pd.read_csv(median_outlines_csv)
    plant_df['weight_g'] = plant_df['plant_name'].map(
        lambda pn: plant_weight_map.get(str(pn).strip(), 'UNKNOWN')
    )

    def _weight_per_kernel(weight_g, n_kernels):
        try:
            w = float(weight_g)
            n = int(n_kernels)
            if n > 0:
                return round(w / n, 4)
        except (TypeError, ValueError):
            pass
        return ''

    plant_df['weight_per_kernel_g'] = plant_df.apply(
        lambda r: _weight_per_kernel(r.get('weight_g'), r.get('n_kernels')), axis=1)
    plant_df = plant_df.fillna('')
    plant_df = reorder_plant_median_columns(plant_df)
    plant_df.to_csv(final_plant_median_csv, index=False)

    print(f'Plant median output saved -> {final_plant_median_csv}')
    print(f'  Plant rows : {len(plant_df)}')


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run(config_path)
