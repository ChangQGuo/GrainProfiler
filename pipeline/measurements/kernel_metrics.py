"""
measurements/kernel_metrics.py
================================
Stage 4 — Compute morphological measurements for every individual kernel.

READS:   masks_binary/                  (one binary mask per kernel)
         WRITES:  measurements.csv               (per-kernel measurements)
         QC images showing measurement lines overlaid on each kernel

Measurements computed per kernel:
  - Area              : pixel area of the kernel contour (using the shoelace formula)
  - Perimeter         : total length of the kernel outline
  - Circularity       : 4π × Area / Perimeter²  (1.0 = circle, lower = elongated)
  - Length            : distance from bottom tip to top point along the main axis
  - Max Width          : widest perpendicular measurement across the full kernel
  - Width_25pct       : width at 25% of kernel length (perpendicular to axis)
  - Width_50pct       : width at 50% of kernel length (the midpoint)
  - Width_75pct       : width at 75% of kernel length
  - Length/Width_50   : axis length divided by Width_50pct
  - Eccentricity      : ellipse eccentricity estimated from moments

Width measurements use Shapely LineString intersection — a perpendicular line
is drawn at the measurement position and intersected with the actual contour.
This is more accurate than bounding box width.

The updated measurement pipeline for each kernel:
  1. Read and clean the SAM binary mask, then compute the centroid with cv2.moments
  2. Predict the directed main-axis vector with the trained ResNet angle model
  3. Intersect the model axis through the centroid with the cleaned contour to
     recover bottom/top endpoints
  4. Measure widths, area, perimeter, circularity, and roundness score

RUN WITH: SAM2 conda environment (has numpy, shapely, pandas, matplotlib, cv2)
  conda activate SAM2
  python measurements/kernel_metrics.py config.yaml
"""

import os
import sys
import math
import json
import importlib.util
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.visualization import (
    draw_all_measurements,
    draw_axis,
    draw_axis_points,
    draw_bottom_point,
    draw_centroid,
    draw_contour,
    draw_qc_legend,
    draw_text_with_outline,
    draw_top_point,
    save_image,
)
from utils.calibration import (
    build_image_scale_lookup,
    convert_area_px2_to_mm2,
    convert_px_to_mm,
    get_original_image_name,
)
from utils.kernel_id import parse_kernel_id_from_name
from processing.contour_extraction import extract_largest_contour
from measurements.geometry import (
    build_axis_frame,
    build_multiscale_sharpness_profiles,
    calculate_centroid,
    calculate_mask_centroid,
    calculate_principal_axis_from_moments,
    clean_kernel_mask,
    coerce_point,
    compute_area_perimeter,
    compute_circularity,
    compute_eccentricity,
    compute_kernel_measurement_bundle,
    compute_local_sharpness,
    compute_major_axis_metrics,
    compute_width_profile,
    contour_to_filled_mask,
    deduplicate_intersections,
    find_contour_intersections,
    get_analysis_contour,
    get_bottom_refinement_hint_contour,
    get_open_contour,
    get_outline_count_points,
    load_binary_mask,
    load_kernels_from_mask_dir,
    max_width_from_contour,
    nearest_contour_index,
    point_is_valid,
    resample_closed_contour,
    sample_outline_count_points,
    sharpness_around_point,
    smooth_closed_contour,
    summarize_axis_widths,
    width_at_fraction,
)
from measurements.axis import (
    score_bottom_endpoint_candidate,
    choose_bottom_top_from_endpoints,
    build_axis_from_direction,
    build_axis_from_model_direction,
    build_fallback_axis,
    derive_axis_from_mask_and_contour,
    derive_axis_from_resnet_model,
    classify_kernel_shape,
    build_axis_from_bottom_and_centroid,
    build_axis_from_anchor_and_direction,
    angle_between_axis_dirs_deg,
    compute_bottom_roughness_metrics,
    score_axis_for_local_refinement,
    score_local_bottom_refinement_candidate,
    score_method_axis_endpoint_refinement,
    locally_refine_bottom_point,
    refine_bottom_point,
    safe_candidate_point,
    safe_candidate_float,
    normalize_axis_angle_deg,
    axis_angle_distance_deg,
    direction_from_angle_deg,
    angle_from_direction,
    robust_penalty_stat,
    mask_contains_point,
    ray_distance_to_mask,
    centroid_axis_endpoints,
    build_axis_candidate_from_angle,
    maybe_refine_candidate_axis_endpoint,
    compute_mask_area_balance,
    compute_width_symmetry_penalty,
    build_color_symmetry_context,
    sample_lab_at,
    anomaly_at,
    compute_color_symmetry_penalty,
    endpoint_local_shape_metrics,
    compute_endpoint_penalty,
    scan_centroid_axis_summaries,
    get_enabled_axis_methods,
    collect_base_axis_angles,
    score_axis_candidates,
    generate_scored_axis_candidates,
    merge_near_duplicate_method_axes,
    select_best_scored_axis,
    measure_kernel_geometry,
)
from measurements.qc import (
    save_outline_count_qc,
    save_measurement_qc,
    save_axis_candidate_image,
    get_accepted_detections_for_overlay,
    load_detection_box_lookup,
    subimage_point_to_full_image,
    overlay_scale,
    save_measurement_overlay_images,
    save_circularity_overlay_images,
)








from utils.config import load_config


PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(PIPELINE_DIR)
RESNET_DIR = os.path.join(PROJECT_DIR, 'resnet')


def resolve_path_from_config(path_text, config_path=None, require_exists=False):
    """Resolve absolute or relative config paths.

    By default the returned path need not exist (useful for probing candidate
    directories).  When ``require_exists`` is True and nothing is found, a
    FileNotFoundError is raised listing every location tried, so misconfigured
    model/checkpoint paths fail here with a clear message instead of deep inside
    a model loader.
    """
    if path_text is None:
        return ''
    path_text = str(path_text).strip()
    if not path_text:
        return ''
    if os.path.isabs(path_text):
        if require_exists and not os.path.exists(path_text):
            raise FileNotFoundError(f'Configured path does not exist: {path_text}')
        return path_text

    candidates = []
    if config_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(config_path)), path_text))
    candidates.append(os.path.join(PROJECT_DIR, path_text))
    candidates.append(os.path.join(PIPELINE_DIR, path_text))
    candidates.append(os.path.abspath(path_text))

    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)

    if require_exists:
        raise FileNotFoundError(
            'Configured path not found. Checked:\n    '
            + '\n    '.join(os.path.abspath(c) for c in candidates)
        )
    return os.path.abspath(candidates[0])


def resolve_runtime_device(config, device_value):
    if device_value in (None, '', 'null'):
        return (config.get('runtime', {}) or {}).get('gpu_device', 'auto') or 'auto'
    return str(device_value)


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load Python module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def find_resnet_code_dir(weights_path, configured_code_dir=None, config_path=None):
    """Find the resnet source directory containing model.py and preprocess.py."""
    candidates = []
    if configured_code_dir:
        candidates.append(resolve_path_from_config(configured_code_dir, config_path=config_path))

    weights_dir = os.path.dirname(os.path.abspath(weights_path))
    candidates.extend([
        RESNET_DIR,
        os.path.abspath(os.path.join(weights_dir, '..', '..')),
        os.path.abspath(os.path.join(weights_dir, '..')),
        weights_dir,
    ])

    checked = []
    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(str(candidate))
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if (
            os.path.isfile(os.path.join(candidate, 'model.py'))
            and os.path.isfile(os.path.join(candidate, 'preprocess.py'))
        ):
            return candidate

    checked_text = '\n    '.join(checked)
    raise FileNotFoundError(
        'ResNet axis model code not found. This is different from the .pt checkpoint: '
        'the pipeline also needs the training source files model.py and preprocess.py.\n'
        'Copy them into a_aguo_test_new/resnet/ on the server, or set '
        'models.resnet_code_dir / measurements.axis_model_code_dir in config.yaml.\n'
        f'Checked:\n    {checked_text}'
    )


class ResNetAxisPredictor:
    """Small wrapper around the trained directed-axis ResNet model."""

    def __init__(
        self,
        weights_path,
        device='auto',
        imgsz=0,
        preprocess_mode='',
        margin=None,
        background='',
        code_dir=None,
        config_path=None,
    ):
        weights_path = os.path.abspath(str(weights_path))
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f'ResNet axis weights not found: {weights_path}. '
                'Set models.resnet_axis in config.yaml to the server checkpoint path.'
            )

        self.code_dir = find_resnet_code_dir(
            weights_path,
            configured_code_dir=code_dir,
            config_path=config_path,
        )

        import torch
        import torch.nn.functional as torch_f

        model_module = load_module_from_file(
            'kernel_axis_resnet_model',
            os.path.join(self.code_dir, 'model.py'),
        )
        preprocess_module = load_module_from_file(
            'kernel_axis_resnet_preprocess',
            os.path.join(self.code_dir, 'preprocess.py'),
        )
        build_resnet_angle_model = model_module.build_resnet_angle_model
        normalize_to_tensor = preprocess_module.normalize_to_tensor
        preprocess_rgb = preprocess_module.preprocess_rgb

        self.torch = torch
        self.torch_f = torch_f
        self.normalize_to_tensor = normalize_to_tensor
        self.preprocess_rgb = preprocess_rgb
        self.weights_path = weights_path
        self.device = self._resolve_device(str(device or 'auto'))

        checkpoint = torch.load(weights_path, map_location=self.device)
        ckpt_args = checkpoint.get('args', {}) or {}
        model_config = checkpoint.get('model_config', {}) or {}

        self.imgsz = int(imgsz or checkpoint.get('imgsz', ckpt_args.get('imgsz', 256)))
        self.preprocess_mode = str(preprocess_mode or ckpt_args.get('preprocess', 'square'))
        self.margin = float(margin if margin is not None else ckpt_args.get('margin', 0.25))
        self.background = str(background or ckpt_args.get('background', 'gray'))
        self.in_channels = int(model_config.get('in_channels', 3))

        # Build model kwargs — only pass in_channels when explicitly
        # present in the checkpoint (backward-compat with old checkpoints
        # and old resnet/model.py that lacks the parameter).
        build_kwargs = dict(
            stem_channels=int(model_config.get('stem_channels', 32)),
            channels=model_config.get('channels', [64, 128, 256, 512]),
            blocks=model_config.get('blocks', [2, 2, 3, 2]),
            dropout=float(model_config.get('dropout', 0.20)),
        )
        if 'in_channels' in model_config:
            build_kwargs['in_channels'] = self.in_channels

        self.model = build_resnet_angle_model(**build_kwargs).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

    def _resolve_device(self, device_arg):
        if device_arg == 'auto':
            return self.torch.device('cuda' if self.torch.cuda.is_available() else 'cpu')
        if device_arg.startswith('cuda') and not self.torch.cuda.is_available():
            print(f'WARNING: requested axis model device {device_arg}, but CUDA is unavailable; using CPU.')
            return self.torch.device('cpu')
        return self.torch.device(device_arg)

    def _make_4channel_tensor(self, image_rgb, mask_channel):
        """Build (4, H, W) tensor: 3 RGB (ImageNet-norm) + binary mask channel."""
        import torch
        image_f = image_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_f = (image_f - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
        image_f = np.transpose(image_f, (2, 0, 1))  # (3, H, W)
        mask_f = np.expand_dims(mask_channel, axis=0)  # (1, H, W)
        combined = np.concatenate([image_f, mask_f], axis=0)  # (4, H, W)
        return torch.from_numpy(combined).float()

    def predict(self, image_bgr, mask_binary=None):
        if image_bgr is None:
            raise ValueError('ResNet axis prediction requires a readable BGR subimage.')

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask_bool = None
        if mask_binary is not None:
            mask_bool = np.asarray(mask_binary) > 0

        image_rgb = self.preprocess_rgb(
            image_rgb,
            mask_bool,
            imgsz=self.imgsz,
            mode=self.preprocess_mode,
            margin_ratio=self.margin,
            background=self.background,
        )

        if self.in_channels == 4 and mask_bool is not None:
            mask_channel = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
            if mask_bool is not None:
                mask_resized = cv2.resize(
                    mask_bool.astype(np.uint8) * 255,
                    (self.imgsz, self.imgsz),
                    interpolation=cv2.INTER_NEAREST,
                )
                mask_channel = (mask_resized > 127).astype(np.float32)
            image_tensor = self._make_4channel_tensor(image_rgb, mask_channel)
        else:
            image_tensor = self.normalize_to_tensor(image_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            pred = self.model(image_tensor)
            pred = self.torch_f.normalize(pred, dim=1, eps=1e-6)[0].detach().cpu().numpy()

        pred_cos = float(pred[0])
        pred_sin = float(pred[1])
        pred_rad = math.atan2(pred_sin, pred_cos) % (2.0 * math.pi)
        pred_deg = math.degrees(pred_rad)
        vector_norm = float(math.hypot(pred_cos, pred_sin))

        return {
            'cos_theta': pred_cos,
            'sin_theta': pred_sin,
            'theta_rad': float(pred_rad),
            'theta_deg': float(pred_deg),
            'vector_norm': vector_norm,
            'direction': np.array([pred_cos, pred_sin], dtype=float),
            'weights_path': self.weights_path,
            'imgsz': int(self.imgsz),
            'preprocess': self.preprocess_mode,
        }


def load_axis_predictor_from_config(config, config_path=None):
    measurement_cfg = config.get('measurements', {}) or {}
    axis_source = str(measurement_cfg.get('axis_source', 'resnet')).strip().lower()
    if axis_source not in {'resnet', 'resnet_axis_model', 'model'}:
        return None

    model_cfg = config.get('models', {}) or {}
    weights = (
        measurement_cfg.get('axis_model_weights')
        or model_cfg.get('resnet_axis')
        or model_cfg.get('axis_resnet')
        or ''
    )
    if not str(weights).strip():
        raise ValueError(
            'measurements.axis_source is resnet, but no checkpoint path was set. '
            'Set models.resnet_axis or measurements.axis_model_weights in config.yaml.'
    )
    weights = resolve_path_from_config(weights, config_path=config_path, require_exists=True)
    code_dir = (
        measurement_cfg.get('axis_model_code_dir')
        or model_cfg.get('resnet_code_dir')
        or ''
    )
    device = resolve_runtime_device(config, measurement_cfg.get('axis_model_device'))
    return ResNetAxisPredictor(
        weights,
        device=device,
        imgsz=int(measurement_cfg.get('axis_model_imgsz', 0) or 0),
        preprocess_mode=str(measurement_cfg.get('axis_model_preprocess', '') or ''),
        margin=measurement_cfg.get('axis_model_margin', None),
        background=str(measurement_cfg.get('axis_model_background', '') or ''),
        code_dir=code_dir,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# Geometry helpers — these are the core measurement functions
# ---------------------------------------------------------------------------

def format_optional_float(value, digits):
    """Round a numeric value or return an empty string."""
    if value is None or value == '':
        return ''
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ''
    if not math.isfinite(numeric):
        return ''
    return round(numeric, digits)


def build_measurement_record_base(image_name, bottom, bottom_info,
                                  mm_per_px, tray_side_px, tray_size_mm):
    """Shared internal fields for measured and skipped kernels."""
    centroid = coerce_point(bottom_info.get('centroid'), fallback=bottom)
    top = coerce_point(bottom_info.get('top'), fallback=bottom)

    return {
        'kernel_name': image_name,
        'shape_label': bottom_info.get('shape_label', ''),
        'measurement_status': bottom_info.get('measurement_status', 'ok'),
        'shape_reason': bottom_info.get('shape_reason', ''),
        'mm_per_px': format_optional_float(mm_per_px, 6),
        'tray_side_px': format_optional_float(tray_side_px, 3),
        'tray_size_mm': format_optional_float(tray_size_mm, 3),
        'bottom_x_px': round(float(bottom[0]), 2),
        'bottom_y_px': round(float(bottom[1]), 2),
        'top_x_px': round(float(top[0]), 2),
        'top_y_px': round(float(top[1]), 2),
        'centroid_x_px': round(float(centroid[0]), 2),
        'centroid_y_px': round(float(centroid[1]), 2),
        'centroid_method': bottom_info.get('centroid_source', ''),
        'axis_angle_deg': format_optional_float(bottom_info.get('axis_angle_deg'), 3),
        'axis_method': bottom_info.get('axis_method', ''),
        'axis_model_pred_theta_deg': format_optional_float(bottom_info.get('axis_model_pred_theta_deg'), 3),
        'axis_model_pred_cos_theta': format_optional_float(bottom_info.get('axis_model_pred_cos_theta'), 6),
        'axis_model_pred_sin_theta': format_optional_float(bottom_info.get('axis_model_pred_sin_theta'), 6),
        'low_confidence_axis': bool(bottom_info.get('low_confidence_axis', False)),
        'low_confidence_reason': bottom_info.get('low_confidence_reason', ''),
        'length_width_ratio': format_optional_float(bottom_info.get('length_width_ratio'), 4),
        'eccentricity': format_optional_float(bottom_info.get('eccentricity'), 4),
    }


def select_measurement_export_columns(df):
    """Keep measurements.csv focused on exported phenotypes, not QC/internal state."""
    keep_columns = [
        'kernel_name',
        'mm_per_px',
        'tray_side_px',
        'tray_size_mm',
        'length_width_ratio',
        'eccentricity',
        'length_px',
        'length_mm',
        'max_width_px',
        'max_width_mm',
        'width_25pct_px',
        'width_25pct_mm',
        'width_50pct_px',
        'width_50pct_mm',
        'width_75pct_px',
        'width_75pct_mm',
        'area_px2',
        'area_mm2',
        'perimeter_px',
        'perimeter_mm',
        'circularity',
    ]
    selected = [col for col in keep_columns if col in df.columns]
    return df.loc[:, selected]


# ---------------------------------------------------------------------------
# QC image saving
# ---------------------------------------------------------------------------

def point_xy_fields(prefix, point):
    if not point_is_valid(point):
        return {f'{prefix}_x': None, f'{prefix}_y': None}
    point = np.asarray(point, dtype=float)
    return {f'{prefix}_x': float(point[0]), f'{prefix}_y': float(point[1])}


def flatten_axis_candidate_records(kernel_name, bottom_info):
    """Return one compact CSV row for the final selected main axis."""
    candidates = (
        bottom_info.get('method_axis_candidate_scores')
        or bottom_info.get('candidate_scores', [])
        or []
    )
    selected = next((c for c in candidates if bool(c.get('selected', False))), None)
    if selected is None and candidates:
        selected = sorted(
            candidates,
            key=lambda item: float(item.get('total_score', 1.0)),
        )[0]
    selected = selected or {}

    bottom = selected.get('bottom', bottom_info.get('bottom'))
    top = selected.get('top', bottom_info.get('top'))
    row = {
        'kernel_name': kernel_name,
        'axis_length_px': format_optional_float(bottom_info.get('axis_length', selected.get('length_px')), 4),
        'pred_theta_deg': format_optional_float(bottom_info.get('axis_model_pred_theta_deg', selected.get('pred_theta_deg')), 4),
    }
    row.update(point_xy_fields('bottom', bottom))
    row.update(point_xy_fields('top', top))
    row.update(point_xy_fields('centroid', bottom_info.get('centroid')))
    return [row]


def run(config_path):
    import cv2  # imported here for QC

    config = load_config(config_path)

    output_dir      = config['output']['base_dir']
    os.makedirs(output_dir, exist_ok=True)  # support standalone run into a fresh dir
    measurements_csv = os.path.join(output_dir, config['output']['measurements_csv'])
    bbox_json       = os.path.join(output_dir, config['output']['bounding_boxes_json'])
    metadata_csv    = os.path.join(output_dir, config['output']['metadata_csv'])
    subimages_dir   = os.path.join(output_dir, config['output']['subimages_dir'])
    masks_dir       = os.path.join(output_dir, config['output']['masks_binary_dir'])
    qc_base_dir     = os.path.join(output_dir, config['output']['qc_dir'])
    qc_dir          = os.path.join(qc_base_dir, 'measurements')
    measurement_overlay_dir = os.path.join(
        output_dir,
        config['output'].get('measurement_overlay_dir', 'measurement_overlay'),
    )
    image_dir       = config['input']['image_dir']
    save_qc         = bool(config.get('visualization', {}).get('save_qc', True))
    measurement_cfg = config['measurements']
    calibration_cfg = config.get('calibration', {})
    contour_cfg     = config.get('contour_filtering', {})
    segmentation_cfg = config.get('segmentation', {})
    n_width_samples = measurement_cfg['num_width_samples']
    crop_padding    = int(segmentation_cfg.get('padding', 20))
    min_contour_area = int(contour_cfg.get('min_contour_area', 500))
    axis_predictor = load_axis_predictor_from_config(config, config_path=config_path)
    save_axis_candidate_images   = bool(measurement_cfg.get('save_axis_candidate_images', False))
    save_circularity_overlay     = bool(measurement_cfg.get('save_circularity_overlay', False))
    save_count_overlay_flag      = bool(measurement_cfg.get('save_count_overlay', False))

    if save_qc:
        os.makedirs(qc_dir, exist_ok=True)

    if not os.path.isdir(masks_dir):
        print(f'ERROR: Binary masks directory not found: {masks_dir}')
        print('Run segmentation/sam_segment.py first.')
        sys.exit(1)

    print(f'Loading kernels directly from binary masks in {masks_dir}...')
    kernels, skipped_masks = load_kernels_from_mask_dir(
        masks_dir,
        min_area=min_contour_area,
        config=measurement_cfg,
    )
    print(f'Loaded {len(kernels)} kernels from binary masks.')
    if skipped_masks > 0:
        print(f'  Skipped {skipped_masks} mask files with unreadable or invalid contours.')
    detection_box_lookup = load_detection_box_lookup(bbox_json)
    if axis_predictor is not None:
        print(
            'Using trained ResNet model for directed main-axis prediction: '
            f'{axis_predictor.weights_path}'
        )
        print(
            f'  axis model preprocess={axis_predictor.preprocess_mode} '
            f'imgsz={axis_predictor.imgsz} device={axis_predictor.device}'
        )
    else:
        print('Using contour-derived axis fallback because no ResNet axis predictor was loaded.')

    scale_lookup = {}
    if bool(calibration_cfg.get('enabled', False)):
        if os.path.exists(metadata_csv):
            meta_df = pd.read_csv(metadata_csv)
            scale_lookup = build_image_scale_lookup(meta_df)
            print(f'Loaded calibration info for {len(scale_lookup)} images from metadata.csv.')
        else:
            print('WARNING: metadata.csv not found; physical mm conversion will be skipped.')

    records = []
    candidate_score_records = []
    axis_results = []       # per-kernel axis data for downstream Stage 5
    overlay_items_by_image = {}

    for k in kernels:
        image_name = k['image_name']
        contour = get_open_contour(k['contour'])
        mask_binary = k['mask_binary']
        subimage_path = os.path.join(subimages_dir, image_name)

        if save_qc and save_count_overlay_flag:
            os.makedirs(os.path.join(qc_dir, 'counts'), exist_ok=True)
            count_qc_path = os.path.join(qc_dir, 'counts', f'count_{image_name}')
            save_outline_count_qc(subimage_path, contour, count_qc_path, measurement_cfg)

        # Step 1: Find centroid, final bottom/top axis, and measurements.
        bottom, bottom_info, measurement_bundle = measure_kernel_geometry(
            contour,
            mask_binary,
            measurement_cfg,
            n_width_samples,
            kernel_name=image_name,
            subimage_path=subimage_path,
            axis_predictor=axis_predictor,
        )
        candidate_score_records.extend(flatten_axis_candidate_records(image_name, bottom_info))

        # Save axis data so Stage 5 (representative_shape) can reuse it without re-running ResNet
        axis_results.append({
            'kernel_name': image_name,
            'bottom': [float(bottom[0]), float(bottom[1])],
            'top': [float(bottom_info['top'][0]), float(bottom_info['top'][1])],
            'shape_label': bottom_info.get('shape_label', 'Normal'),
            'measurement_status': bottom_info.get('measurement_status', 'ok'),
            'axis_length': float(measurement_bundle['axis_length']),
            'max_width': float(measurement_bundle['max_width']),
            'area': float(measurement_bundle['area']),
            'perimeter': float(measurement_bundle['perimeter']),
            'circularity': float(measurement_bundle['circularity']),
        })

        if save_axis_candidate_images:
            os.makedirs(os.path.join(qc_dir, 'axis_candidate'), exist_ok=True)
            candidate_image_path = os.path.join(qc_dir, 'axis_candidate', f'axis_candidate_{image_name}')
            save_axis_candidate_image(
                subimage_path,
                contour,
                bottom_info,
                candidate_image_path,
            )

        # Step 2: Read the refined axis geometry.
        top = bottom_info['top']
        axis_length = float(measurement_bundle['axis_length'])
        axis_dir = measurement_bundle['axis_dir']
        if axis_length < 1:
            print(f'  WARNING: Degenerate axis for {image_name} — skipping')
            continue
        perp_dir = np.array([-axis_dir[1], axis_dir[0]])  # 90° rotation

        orig_image_name = get_original_image_name(image_name)
        kernel_id = parse_kernel_id_from_name(image_name)
        det_box = None
        if kernel_id is not None:
            det_box = detection_box_lookup.get(orig_image_name, {}).get(kernel_id)
        scale_info = scale_lookup.get(orig_image_name, {})
        mm_per_px = scale_info.get('mm_per_px')
        tray_side_px = scale_info.get('tray_side_px')
        tray_size_mm = scale_info.get('tray_size_mm')
        measurement_status = bottom_info.get('measurement_status', 'ok')
        if det_box is not None:
            overlay_items_by_image.setdefault(orig_image_name, []).append({
                'kernel_name': image_name,
                'box': det_box,
                'measurement_status': measurement_status,
                'shape_label': bottom_info.get('shape_label', ''),
                'bottom_corrected': np.asarray(bottom, dtype=float),
                'top': np.asarray(top, dtype=float),
                'centroid': np.asarray(bottom_info.get('centroid', bottom), dtype=float),
                'circularity': float(bottom_info.get('round_circularity', 0.0) or 0.0),
                'is_round': bool(bottom_info.get('shape_label') == 'Round'),
            })
        # Keep the original kernel_name for ALL kernels (round ones included) so
        # measurements.csv keys match subimages/ and axis_results.json. Roundness
        # is flagged by the shape_label / measurement_status columns instead.
        base_record = build_measurement_record_base(
            image_name,
            bottom,
            bottom_info,
            mm_per_px,
            tray_side_px,
            tray_size_mm,
        )

        if bottom_info.get('shape_label') == 'Round':
            records.append({
                **base_record,
                'length_px': '',
                'length_mm': '',
                'max_width_px': '',
                'max_width_mm': '',
                'width_25pct_px': '',
                'width_25pct_mm': '',
                'width_50pct_px': '',
                'width_50pct_mm': '',
                'width_75pct_px': '',
                'width_75pct_mm': '',
                'area_px2': '',
                'area_mm2': '',
                'perimeter_px': '',
                'perimeter_mm': '',
                'circularity': format_optional_float(bottom_info.get('round_circularity'), 4),
            })

            print(
                f'  {image_name}: classified as Round -> skipping measurements '
                f'(Circularity={bottom_info.get("round_circularity", 0.0):.4f} > '
                f'{bottom_info.get("round_circularity_threshold", 0.0):.4f}, '
                f'reasons={bottom_info.get("shape_reason", "")})'
            )

            if save_qc:
                subimage_path = os.path.join(subimages_dir, image_name)
                qc_path = os.path.join(qc_dir, f'meas_{image_name}')
                save_measurement_qc(
                    subimage_path,
                    contour,
                    bottom,
                    top,
                    qc_path,
                    refine_info=bottom_info,
                    measurement_bundle=measurement_bundle,
                )
            continue

        w25 = float(measurement_bundle['width_25pct'])
        w50 = float(measurement_bundle['width_50pct'])
        w75 = float(measurement_bundle['width_75pct'])
        max_width = float(measurement_bundle['max_width'])
        area = float(measurement_bundle['area'])
        perimeter = float(measurement_bundle['perimeter'])
        circularity = float(measurement_bundle['circularity'])
        length_width_ratio = measurement_bundle.get('length_width_ratio')
        eccentricity = float(measurement_bundle['eccentricity'])

        records.append({
            **base_record,
            'length_px':    round(axis_length, 2),
            'length_mm':    convert_px_to_mm(axis_length, mm_per_px),
            'max_width_px': round(max_width, 2),
            'max_width_mm': convert_px_to_mm(max_width, mm_per_px),
            'width_25pct_px': round(w25, 2),
            'width_25pct_mm': convert_px_to_mm(w25, mm_per_px),
            'width_50pct_px': round(w50, 2),
            'width_50pct_mm': convert_px_to_mm(w50, mm_per_px),
            'width_75pct_px': round(w75, 2),
            'width_75pct_mm': convert_px_to_mm(w75, mm_per_px),
            'area_px2':     round(area, 2),
            'area_mm2':     convert_area_px2_to_mm2(area, mm_per_px),
            'perimeter_px': round(perimeter, 2),
            'perimeter_mm': convert_px_to_mm(perimeter, mm_per_px),
            'circularity':  round(circularity, 4),
            'length_width_ratio': round(length_width_ratio, 4) if length_width_ratio is not None else '',
            'eccentricity': round(eccentricity, 4),
        })

        print(f'  {image_name}: L={axis_length:.1f}  MB={max_width:.1f}  '
              f'W50={w50:.1f}  Circ={circularity:.3f}  '
              f'L/W50={length_width_ratio if length_width_ratio is not None else 0.0:.3f}  '
              f'Ecc={eccentricity:.3f}  '
              f'PredTheta={bottom_info.get("axis_model_pred_theta_deg", 0.0) if bottom_info.get("axis_model_pred_theta_deg") is not None else 0.0:.2f}deg')

        if save_qc:
            subimage_path = os.path.join(subimages_dir, image_name)
            qc_path = os.path.join(qc_dir, f'meas_{image_name}')
            save_measurement_qc(
                subimage_path,
                contour,
                bottom,
                top,
                qc_path,
                refine_info=bottom_info,
                measurement_bundle=measurement_bundle,
            )

    # Save to CSV
    df = select_measurement_export_columns(pd.DataFrame(records))
    df.to_csv(measurements_csv, index=False)

    # Also save as Parquet for fast GUI loading (optional — skips if pyarrow missing)
    measurements_parquet = os.path.join(output_dir,
                                        config['output'].get('measurements_parquet', 'measurements.parquet'))
    try:
        df.to_parquet(measurements_parquet, index=False)
        print(f'  Parquet saved: {measurements_parquet}')
    except ImportError:
        print('  Parquet skipped: install pyarrow or fastparquet for faster GUI loading')

    # Per-kernel axis results for downstream Stage 5 (avoids re-running ResNet)
    # Compact format: no indent, comma-separated — reduces file size ~3× vs pretty-printed
    axis_results_json = os.path.join(output_dir, config['output'].get('axis_results_json', 'axis_results.json'))
    with open(axis_results_json, 'w') as f:
        json.dump(axis_results, f, separators=(',', ':'))
    print(f'Axis results saved: {len(axis_results)} kernels -> {axis_results_json}')

    # Measurement overlays
    overlay_count = save_measurement_overlay_images(
        overlay_items_by_image,
        image_dir,
        measurement_overlay_dir,
        crop_padding,
    )

    # Circularity overlays (gated)
    if save_circularity_overlay:
        circularity_overlay_dir = os.path.join(qc_dir, 'circularity_overlay')
        os.makedirs(circularity_overlay_dir, exist_ok=True)
        circularity_count = save_circularity_overlay_images(
            overlay_items_by_image, image_dir, circularity_overlay_dir
        )

    print(f'\nMeasurements saved for {len(records)} kernels → {measurements_csv}')
    print(f'Axis candidate scores processed: {len(candidate_score_records)} kernels')
    if save_qc:
        print('Check measurement QC images in:', qc_dir)
    if save_axis_candidate_images:
        print('Axis candidate images saved to:', os.path.join(qc_dir, 'axis_candidate'))
    if save_count_overlay_flag:
        print('Outline count QC images in:', os.path.join(qc_dir, 'counts'))
    if save_circularity_overlay:
        print(f'Circularity overlays saved for {circularity_count} images')
    if overlay_count > 0:
        print(f'Measurement overlays saved for {overlay_count} images -> {measurement_overlay_dir}')
    print('Verify that the centroid, axis, and top/bottom annotations follow each kernel correctly.')

    # --- QC cleanup ---
    _cleanup_qc(config, output_dir)


def _cleanup_qc(config, output_dir):
    import shutil
    vis = config.get('visualization', {})

    def _rm(path):
        if os.path.exists(path):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
            print(f'  Cleaned: {os.path.basename(path)}')

    # Delete ONLY the measurements QC subfolder, not the whole qc_visualizations/
    # (which also holds Stage 2 detection QC and other stages' outputs).
    qc_base = os.path.join(output_dir, config['output'].get('qc_dir', 'qc_visualizations'))
    if not vis.get('save_measurement_per_kernel_qc', True):
        _rm(os.path.join(qc_base, 'measurements'))
    if not vis.get('save_measurement_overlay', True):
        _rm(os.path.join(output_dir, config['output'].get('measurement_overlay_dir', 'measurement_overlay')))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run(config_path)
