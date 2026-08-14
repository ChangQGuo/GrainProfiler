"""axis.py - directed-axis candidate generation, scoring, selection and refinement.

Extracted from kernel_metrics.py.  Depends on measurements.geometry (and numpy /
cv2 / shapely), but NOT on ResNetAxisPredictor (it is passed in as a parameter),
so this module forms a clean middle layer between geometry and orchestration.
"""

import math
import os

import cv2
import numpy as np

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


def score_bottom_endpoint_candidate(contour, candidate_bottom, candidate_top, endpoint_sharpness, config):
    """
    Score one axis endpoint as the bottom tip.

    Bottom/top orientation is driven mainly by endpoint sharpness, with a
    taper-based stabilizer: the true bottom should stay narrow near the tip and
    widen as we move upward along the axis.
    """
    widths = summarize_axis_widths(contour, candidate_bottom, candidate_top, config)
    near_width = float(widths.get('near_width', 0.0) or 0.0)
    mid_width = float(widths.get('mid_width', 0.0) or 0.0)
    upper_width = float(widths.get('upper_width', 0.0) or 0.0)

    taper_mid = 0.0
    taper_upper = 0.0
    if mid_width > 1e-6:
        taper_mid = 1.0 - min(2.0, near_width / mid_width)
    if upper_width > 1e-6:
        taper_upper = 1.0 - min(2.0, near_width / upper_width)

    taper_score = float(0.45 * taper_mid + 0.55 * taper_upper)

    if near_width <= mid_width <= upper_width:
        monotonic_score = 1.0
    elif near_width <= mid_width or near_width <= upper_width:
        monotonic_score = 0.5
    else:
        monotonic_score = 0.0

    sharpness_weight = float(config.get('bottom_sharpness_score_weight', 1.0))
    taper_weight = float(config.get('bottom_taper_score_weight', 0.35))
    monotonic_weight = float(config.get('bottom_monotonic_score_weight', 0.15))

    score = (
        sharpness_weight * float(endpoint_sharpness)
        + taper_weight * taper_score
        + monotonic_weight * monotonic_score
    )

    return {
        'score': float(score),
        'sharpness': float(endpoint_sharpness),
        'taper_score': taper_score,
        'monotonic_score': float(monotonic_score),
        'near_width': near_width,
        'mid_width': mid_width,
        'upper_width': upper_width,
    }




def choose_bottom_top_from_endpoints(contour, point_a, point_b, sharpness_profiles, config):
    """Orient one axis by choosing which endpoint is the sharper, more tapered bottom."""
    base_sharpness = sharpness_profiles[0][1] if sharpness_profiles else np.zeros(len(contour), dtype=float)
    sharpness_radius = int(config.get('endpoint_sharpness_neighbor_radius', 2))

    sharpness_a, idx_a = sharpness_around_point(
        contour,
        base_sharpness,
        point_a,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )
    sharpness_b, idx_b = sharpness_around_point(
        contour,
        base_sharpness,
        point_b,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )

    cand_a = score_bottom_endpoint_candidate(contour, point_a, point_b, sharpness_a, config)
    cand_b = score_bottom_endpoint_candidate(contour, point_b, point_a, sharpness_b, config)

    if cand_a['score'] >= cand_b['score']:
        return {
            'bottom': np.asarray(point_a, dtype=float),
            'top': np.asarray(point_b, dtype=float),
            'bottom_idx': int(idx_a),
            'top_idx': int(idx_b),
            'bottom_sharpness': float(sharpness_a),
            'top_sharpness': float(sharpness_b),
            'sharpness_gap': float(sharpness_a - sharpness_b),
            'bottom_candidate_score': float(cand_a['score']),
            'top_candidate_score': float(cand_b['score']),
            'bottom_taper_score': float(cand_a['taper_score']),
            'top_taper_score': float(cand_b['taper_score']),
            'bottom_monotonic_score': float(cand_a['monotonic_score']),
            'top_monotonic_score': float(cand_b['monotonic_score']),
        }

    return {
        'bottom': np.asarray(point_b, dtype=float),
        'top': np.asarray(point_a, dtype=float),
        'bottom_idx': int(idx_b),
        'top_idx': int(idx_a),
        'bottom_sharpness': float(sharpness_b),
        'top_sharpness': float(sharpness_a),
        'sharpness_gap': float(sharpness_b - sharpness_a),
        'bottom_candidate_score': float(cand_b['score']),
        'top_candidate_score': float(cand_a['score']),
        'bottom_taper_score': float(cand_b['taper_score']),
        'top_taper_score': float(cand_a['taper_score']),
        'bottom_monotonic_score': float(cand_b['monotonic_score']),
        'top_monotonic_score': float(cand_a['monotonic_score']),
    }




def build_axis_from_direction(contour, origin, direction, sharpness_profiles, config,
                              method):
    """
    Build an axis from a known principal direction and origin.

    The direction comes from mask moments or a centroid chord; we recover the
    actual boundary endpoints by intersecting that line with the contour
    polyline and assign bottom/top by endpoint sharpness.
    """

    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm
    origin = np.asarray(origin, dtype=float)

    half_extent = max(float(np.linalg.norm(np.ptp(contour, axis=0))), 1.0) * 2.0 + 20.0
    from shapely.geometry import LineString
    line = LineString([
        origin - float(half_extent) * direction,
        origin + float(half_extent) * direction,
    ])
    intersections = find_contour_intersections(contour, line)
    intersections = deduplicate_intersections(intersections, origin, direction)
    if len(intersections) < 2:
        return None

    point_a = np.asarray(intersections[0], dtype=float)
    point_b = np.asarray(intersections[-1], dtype=float)
    axis_length = float(np.linalg.norm(point_b - point_a))
    if axis_length < 1.0:
        return None

    endpoint_choice = choose_bottom_top_from_endpoints(
        contour, point_a, point_b, sharpness_profiles, config
    )
    bottom = endpoint_choice['bottom']
    top = endpoint_choice['top']
    bottom_idx = endpoint_choice['bottom_idx']
    top_idx = endpoint_choice['top_idx']
    bottom_sharpness = endpoint_choice['bottom_sharpness']
    top_sharpness = endpoint_choice['top_sharpness']

    axis_vec, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    axis_angle_deg = (math.degrees(math.atan2(axis_vec[1], axis_vec[0])) + 360.0) % 180.0

    return {
        'bottom': bottom,
        'top': top,
        'bottom_idx': int(bottom_idx),
        'top_idx': int(top_idx),
        'bottom_sharpness': float(bottom_sharpness),
        'top_sharpness': float(top_sharpness),
        'sharpness_gap': float(endpoint_choice['sharpness_gap']),
        'length_px': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'axis_angle_deg': float(axis_angle_deg),
        'score': float(endpoint_choice['bottom_candidate_score']),
        'bottom_candidate_score': float(endpoint_choice['bottom_candidate_score']),
        'top_candidate_score': float(endpoint_choice['top_candidate_score']),
        'bottom_taper_score': float(endpoint_choice['bottom_taper_score']),
        'top_taper_score': float(endpoint_choice['top_taper_score']),
        'method': method,
    }




def build_axis_from_model_direction(contour, centroid, prediction, sharpness_profiles, config):
    """
    Build the final bottom/top axis from the directed ResNet prediction.

    The model predicts an image-coordinate vector [cos(theta), sin(theta)]
    pointing from top (crown, wider, embryo-bearing) to bottom (pedicel,
    narrower, attachment end), matching the manual labeler arrow convention.
    The axis line passes through the mask centroid; endpoints are the two
    contour intersections. The intersection with positive projection along
    the predicted direction is assigned as bottom (pedicel end); the
    intersection with negative projection is assigned as top (crown end).
    """
    direction = np.asarray(prediction.get('direction'), dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        raise ValueError('ResNet axis prediction returned a near-zero direction vector.')
    direction = direction / norm
    centroid = np.asarray(centroid, dtype=float)

    half_extent = max(float(np.linalg.norm(np.ptp(contour, axis=0))), 1.0) * 2.0 + 20.0
    from shapely.geometry import LineString
    line = LineString([
        centroid - float(half_extent) * direction,
        centroid + float(half_extent) * direction,
    ])
    intersections = find_contour_intersections(contour, line)
    intersections = deduplicate_intersections(intersections, centroid, direction)

    if len(intersections) >= 2:
        ordered = sorted(
            [(float(np.dot(np.asarray(pt, dtype=float) - centroid, direction)), np.asarray(pt, dtype=float))
             for pt in intersections],
            key=lambda item: item[0],
        )
        top = ordered[0][1]                                 # negative → against d → crown
        bottom = ordered[-1][1]                              # positive → with d → pedicel
    else:
        projections = np.dot(contour - centroid, direction)
        top = contour[int(np.argmin(projections))]           # projection min → crown
        bottom = contour[int(np.argmax(projections))]        # projection max → pedicel

    axis_vec, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1.0:
        raise ValueError('ResNet axis contour intersections produced a degenerate axis.')

    base_sharpness = sharpness_profiles[0][1] if sharpness_profiles else np.zeros(len(contour), dtype=float)
    sharpness_radius = int(config.get('endpoint_sharpness_neighbor_radius', 2))
    bottom_sharpness, bottom_idx = sharpness_around_point(
        contour,
        base_sharpness,
        bottom,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )
    top_sharpness, top_idx = sharpness_around_point(
        contour,
        base_sharpness,
        top,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )
    bottom_score = score_bottom_endpoint_candidate(contour, bottom, top, bottom_sharpness, config)
    top_score = score_bottom_endpoint_candidate(contour, top, bottom, top_sharpness, config)
    axis_angle_deg = (math.degrees(math.atan2(axis_vec[1], axis_vec[0])) + 360.0) % 180.0

    candidate = {
        'bottom': np.asarray(bottom, dtype=float),
        'top': np.asarray(top, dtype=float),
        'bottom_idx': int(bottom_idx),
        'top_idx': int(top_idx),
        'bottom_sharpness': float(bottom_sharpness),
        'top_sharpness': float(top_sharpness),
        'sharpness_gap': float(bottom_sharpness - top_sharpness),
        'length_px': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'axis_angle_deg': float(axis_angle_deg),
        'score': 0.0,
        'total_score': 0.0,
        'bottom_candidate_score': float(bottom_score['score']),
        'top_candidate_score': float(top_score['score']),
        'bottom_taper_score': float(bottom_score['taper_score']),
        'top_taper_score': float(top_score['taper_score']),
        'bottom_monotonic_score': float(bottom_score['monotonic_score']),
        'top_monotonic_score': float(top_score['monotonic_score']),
        'method': 'resnet_axis_model',
        'base_source': 'resnet_axis_model',
        'selected': True,
        'rank': 1,
        'pred_theta_deg': float(prediction.get('theta_deg', 0.0)),
        'pred_theta_rad': float(prediction.get('theta_rad', 0.0)),
        'pred_cos_theta': float(prediction.get('cos_theta', direction[0])),
        'pred_sin_theta': float(prediction.get('sin_theta', direction[1])),
        'pred_vector_norm': float(prediction.get('vector_norm', 1.0)),
    }
    return candidate




def build_fallback_axis(contour, centroid, sharpness_profiles, config):
    """Fallback axis from the contour PCA major direction when moments are unavailable."""
    pts = get_open_contour(contour)
    pca_metrics = compute_major_axis_metrics(pts, centroid)
    major_axis = np.asarray(pca_metrics['major_axis'], dtype=float)

    fallback = build_axis_from_direction(
        pts,
        centroid,
        major_axis,
        sharpness_profiles,
        config,
        method='pca_fallback',
    )
    if fallback is not None:
        return fallback

    projections = np.dot(pts - centroid, major_axis)
    point_a = pts[int(np.argmin(projections))]
    point_b = pts[int(np.argmax(projections))]

    endpoint_choice = choose_bottom_top_from_endpoints(
        pts, point_a, point_b, sharpness_profiles, config
    )
    bottom = endpoint_choice['bottom']
    top = endpoint_choice['top']
    bottom_idx = endpoint_choice['bottom_idx']
    top_idx = endpoint_choice['top_idx']
    bottom_sharpness = endpoint_choice['bottom_sharpness']
    top_sharpness = endpoint_choice['top_sharpness']

    axis_vec, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    axis_angle_deg = (math.degrees(math.atan2(axis_vec[1], axis_vec[0])) + 360.0) % 180.0

    return {
        'bottom': bottom,
        'top': top,
        'bottom_idx': int(bottom_idx),
        'top_idx': int(top_idx),
        'bottom_sharpness': float(bottom_sharpness),
        'top_sharpness': float(top_sharpness),
        'sharpness_gap': float(endpoint_choice['sharpness_gap']),
        'length_px': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'axis_angle_deg': float(axis_angle_deg),
        'score': float(endpoint_choice['bottom_candidate_score']),
        'bottom_candidate_score': float(endpoint_choice['bottom_candidate_score']),
        'top_candidate_score': float(endpoint_choice['top_candidate_score']),
        'bottom_taper_score': float(endpoint_choice['bottom_taper_score']),
        'top_taper_score': float(endpoint_choice['top_taper_score']),
        'method': 'pca_fallback',
    }




def derive_axis_from_mask_and_contour(contour, mask_binary, config, image_bgr=None):
    """Contour-derived fallback for bottom/top when ResNet axis mode is disabled."""
    pts = get_analysis_contour(contour, config)
    centroid, centroid_source = calculate_mask_centroid(mask_binary, pts)
    sharpness_profiles = build_multiscale_sharpness_profiles(
        pts,
        base_window=int(config.get('bottom_sharpness_window', 5)),
        scale_factors=config.get('endpoint_sharpness_scale_factors', [1.0, 2.0, 3.5]),
    )

    area, perimeter = compute_area_perimeter(pts)
    axis_selection_circularity = compute_circularity(area, perimeter)
    candidates, base_axes, method_axis_candidates = generate_scored_axis_candidates(
        pts,
        mask_binary,
        centroid,
        sharpness_profiles,
        config,
        image_bgr=image_bgr,
    )

    if not candidates:
        fallback = build_fallback_axis(pts, centroid, sharpness_profiles, config)
        candidates = score_axis_candidates(
            pts,
            mask_binary,
            centroid,
            [fallback],
            sharpness_profiles,
            config,
            image_bgr=image_bgr,
        )
        method_axis_candidates = candidates

    best, low_confidence, low_confidence_reason = select_best_scored_axis(candidates, config)
    if best is None:
        best = build_fallback_axis(pts, centroid, sharpness_profiles, config)
        best['total_score'] = 1.0
        best['score'] = 1.0
        low_confidence = True
        low_confidence_reason = 'no_valid_scored_candidate'

    method = best.get('method', 'scored_centroid_axis')

    moment_axis = next((c for c in candidates if c.get('base_source') == 'moments_axis'), None)
    longest_axis = next((c for c in candidates if c.get('base_source') == 'longest_centroid_chord'), None)
    shortest_axis = next((c for c in candidates if c.get('base_source') == 'shortest_centroid_chord'), None)

    return {
        'centroid': centroid,
        'centroid_source': centroid_source,
        'bottom': np.asarray(best['bottom'], dtype=float),
        'top': np.asarray(best['top'], dtype=float),
        'axis_dir': np.asarray(best['axis_dir'], dtype=float),
        'perp_dir': np.asarray(best['perp_dir'], dtype=float),
        'axis_length': float(best['length_px']),
        'axis_angle_deg': float(best['axis_angle_deg']),
        'bottom_sharpness': float(best.get('bottom_sharpness', 0.0)),
        'top_sharpness': float(best.get('top_sharpness', 0.0)),
        'sharpness_gap': float(best.get('sharpness_gap', 0.0)),
        'score': float(best.get('total_score', best.get('score', 1.0))),
        'bottom_candidate_score': float(best.get('bottom_candidate_score', 0.0)),
        'top_candidate_score': float(best.get('top_candidate_score', 0.0)),
        'bottom_taper_score': float(best.get('bottom_taper_score', 0.0)),
        'top_taper_score': float(best.get('top_taper_score', 0.0)),
        'candidate_count': int(len(candidates)),
        'method': method,
        'axis_selection_reason': 'lowest_weighted_candidate_score',
        'axis_selection_circularity': float(axis_selection_circularity),
        'skip_local_refine': bool(config.get('candidate_axis_skip_extra_bottom_refine', True)),
        'low_confidence_axis': bool(low_confidence),
        'low_confidence_reason': low_confidence_reason,
        'candidate_score_margin': best.get('score_margin_to_second'),
        'candidate_rank': int(best.get('rank', 1)),
        'candidate_scores': candidates,
        'method_axis_candidate_scores': method_axis_candidates,
        'candidate_base_axes': base_axes,
        'area_balance_penalty': best.get('area_balance_penalty'),
        'width_symmetry_penalty': best.get('width_symmetry_penalty'),
        'color_symmetry_penalty': best.get('color_symmetry_penalty'),
        'length_penalty': best.get('length_penalty'),
        'endpoint_penalty': best.get('endpoint_penalty'),
        'color_valid_pairs': int(best.get('color_valid_pairs', 0) or 0),
        'width_valid_samples': int(best.get('width_valid_samples', 0) or 0),
        'moments_axis_bottom': safe_candidate_point(moment_axis, 'bottom'),
        'moments_axis_top': safe_candidate_point(moment_axis, 'top'),
        'moments_axis_length_px': safe_candidate_float(moment_axis, 'length_px'),
        'moments_axis_angle_deg': safe_candidate_float(moment_axis, 'axis_angle_deg'),
        'moments_axis_total_score': safe_candidate_float(moment_axis, 'total_score'),
        'longest_chord_bottom': safe_candidate_point(longest_axis, 'bottom'),
        'longest_chord_top': safe_candidate_point(longest_axis, 'top'),
        'longest_chord_length_px': safe_candidate_float(longest_axis, 'length_px'),
        'longest_chord_angle_deg': safe_candidate_float(longest_axis, 'axis_angle_deg'),
        'longest_chord_total_score': safe_candidate_float(longest_axis, 'total_score'),
        'shortest_chord_bottom': safe_candidate_point(shortest_axis, 'bottom'),
        'shortest_chord_top': safe_candidate_point(shortest_axis, 'top'),
        'shortest_chord_length_px': safe_candidate_float(shortest_axis, 'length_px'),
        'shortest_chord_angle_deg': safe_candidate_float(shortest_axis, 'axis_angle_deg'),
        'shortest_chord_total_score': safe_candidate_float(shortest_axis, 'total_score'),
    }




def derive_axis_from_resnet_model(contour, mask_binary, config, image_bgr, axis_predictor):
    """Find centroid, predict directed axis with ResNet, and recover contour endpoints."""
    if axis_predictor is None:
        raise RuntimeError(
            'measurements.axis_source is set to resnet, but no ResNetAxisPredictor was loaded.'
        )
    if image_bgr is None:
        raise FileNotFoundError('ResNet axis prediction requires the cropped subimage file.')

    pts = get_analysis_contour(contour, config)
    centroid, centroid_source = calculate_mask_centroid(mask_binary, pts)
    sharpness_profiles = build_multiscale_sharpness_profiles(
        pts,
        base_window=int(config.get('bottom_sharpness_window', 5)),
        scale_factors=config.get('endpoint_sharpness_scale_factors', [1.0, 2.0, 3.5]),
    )
    area, perimeter = compute_area_perimeter(pts)
    axis_selection_circularity = compute_circularity(area, perimeter)
    prediction = axis_predictor.predict(image_bgr, mask_binary=mask_binary)
    best = build_axis_from_model_direction(
        pts,
        centroid,
        prediction,
        sharpness_profiles,
        config,
    )

    return {
        'centroid': centroid,
        'centroid_source': centroid_source,
        'bottom': np.asarray(best['bottom'], dtype=float),
        'top': np.asarray(best['top'], dtype=float),
        'axis_dir': np.asarray(best['axis_dir'], dtype=float),
        'perp_dir': np.asarray(best['perp_dir'], dtype=float),
        'axis_length': float(best['length_px']),
        'axis_angle_deg': float(best['axis_angle_deg']),
        'bottom_sharpness': float(best.get('bottom_sharpness', 0.0)),
        'top_sharpness': float(best.get('top_sharpness', 0.0)),
        'sharpness_gap': float(best.get('sharpness_gap', 0.0)),
        'score': 0.0,
        'bottom_candidate_score': float(best.get('bottom_candidate_score', 0.0)),
        'top_candidate_score': float(best.get('top_candidate_score', 0.0)),
        'bottom_taper_score': float(best.get('bottom_taper_score', 0.0)),
        'top_taper_score': float(best.get('top_taper_score', 0.0)),
        'bottom_monotonic_score': float(best.get('bottom_monotonic_score', 0.0)),
        'top_monotonic_score': float(best.get('top_monotonic_score', 0.0)),
        'candidate_count': 1,
        'method': 'resnet_axis_model',
        'axis_selection_reason': 'resnet_directed_axis_prediction',
        'axis_selection_circularity': float(axis_selection_circularity),
        'skip_local_refine': not bool(config.get('axis_model_allow_local_bottom_refine', False)),
        'low_confidence_axis': False,
        'low_confidence_reason': '',
        'candidate_score_margin': None,
        'candidate_rank': 1,
        'candidate_scores': [best],
        'method_axis_candidate_scores': [best],
        'candidate_base_axes': [{
            'method': 'resnet_axis_model',
            'angle_deg': normalize_axis_angle_deg(best.get('axis_angle_deg', 0.0)),
            'pred_theta_deg': float(prediction.get('theta_deg', 0.0)),
            'pred_cos_theta': float(prediction.get('cos_theta', 0.0)),
            'pred_sin_theta': float(prediction.get('sin_theta', 0.0)),
        }],
        'area_balance_penalty': 0.0,
        'width_symmetry_penalty': 0.0,
        'color_symmetry_penalty': 0.0,
        'length_penalty': 0.0,
        'endpoint_penalty': 0.0,
        'color_valid_pairs': 0,
        'width_valid_samples': 0,
        'axis_model_pred_theta_deg': float(prediction.get('theta_deg', 0.0)),
        'axis_model_pred_theta_rad': float(prediction.get('theta_rad', 0.0)),
        'axis_model_pred_cos_theta': float(prediction.get('cos_theta', 0.0)),
        'axis_model_pred_sin_theta': float(prediction.get('sin_theta', 0.0)),
        'axis_model_pred_vector_norm': float(prediction.get('vector_norm', 1.0)),
        'axis_model_weights_path': prediction.get('weights_path', ''),
        'axis_model_imgsz': prediction.get('imgsz', ''),
        'axis_model_preprocess': prediction.get('preprocess', ''),
        'moments_axis_bottom': None,
        'moments_axis_top': None,
        'moments_axis_length_px': None,
        'moments_axis_angle_deg': None,
        'moments_axis_total_score': None,
        'longest_chord_bottom': None,
        'longest_chord_top': None,
        'longest_chord_length_px': None,
        'longest_chord_angle_deg': None,
        'longest_chord_total_score': None,
        'shortest_chord_bottom': None,
        'shortest_chord_top': None,
        'shortest_chord_length_px': None,
        'shortest_chord_angle_deg': None,
        'shortest_chord_total_score': None,
    }




def classify_kernel_shape(contour, bottom_info, config, measurement_bundle=None,
                          mask_binary=None, image_bgr=None):
    """
    Flag kernels that should be excluded from downstream measurements.
    """
    if measurement_bundle is None:
        bottom = coerce_point(bottom_info.get('bottom'), fallback=bottom_info.get('centroid'))
        top = coerce_point(bottom_info.get('top'), fallback=bottom)
        measurement_bundle = compute_kernel_measurement_bundle(
            contour,
            bottom,
            top,
            bottom_info=bottom_info,
            num_width_samples=max(8, int(config.get('num_width_samples', 100))),
            config=config,
        )

    circularity = measurement_bundle.get('circularity')
    length_width_ratio = measurement_bundle.get('length_width_ratio')
    eccentricity = measurement_bundle.get('eccentricity')
    enabled = bool(config.get('round_kernel_filter_enabled', False))
    circularity_threshold = float(config.get('round_circularity_threshold', 0.90))
    circularity_ok = (
        enabled
        and circularity is not None
        and circularity_threshold > 0.0
        and float(circularity) > circularity_threshold
    )
    is_round = bool(circularity_ok)
    reason_bits = []
    if is_round:
        reason_bits.append(f'circularity>{circularity_threshold:.4f}')

    if is_round:
        shape_label = 'Round'
        measurement_status = 'skipped_round'
    else:
        shape_label = ''
        measurement_status = 'ok'

    if not is_round and bool(bottom_info.get('low_confidence_axis', False)):
        reason = str(bottom_info.get('low_confidence_reason', '') or 'candidate_scores_ambiguous')
        return {
            'shape_label': 'LowConfidenceAxis',
            'measurement_status': 'low_confidence_axis',
            'shape_reason': reason,
            'round_circularity': circularity,
            'round_circularity_threshold': circularity_threshold,
            'length_width_ratio': length_width_ratio,
            'eccentricity': eccentricity,
            'round_filter_enabled': enabled,
            'round_circularity_ok': bool(circularity_ok),
        }

    return {
        'shape_label': shape_label,
        'measurement_status': measurement_status,
        'shape_reason': '|'.join(reason_bits),
        'round_circularity': circularity,
        'round_circularity_threshold': circularity_threshold,
        'length_width_ratio': length_width_ratio,
        'eccentricity': eccentricity,
        'round_filter_enabled': enabled,
        'round_circularity_ok': bool(circularity_ok),
    }




def build_axis_from_bottom_and_centroid(contour, bottom_hint, centroid):
    """
    Build a refined axis constrained to pass through the centroid and a candidate bottom.

    This is a local correction step around the already-determined principal axis:
    we preserve the centroid anchor, move the bottom slightly along the nearby tip
    contour, and recompute the opposite endpoint from the same line.
    """
    bottom_hint = np.asarray(bottom_hint, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    direction = centroid - bottom_hint
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm

    half_extent = max(float(np.linalg.norm(np.ptp(contour, axis=0))), 1.0) * 2.0 + 20.0
    from shapely.geometry import LineString
    line = LineString([
        centroid - float(half_extent) * direction,
        centroid + float(half_extent) * direction,
    ])
    intersections = find_contour_intersections(contour, line)
    intersections = deduplicate_intersections(intersections, centroid, direction)
    if len(intersections) < 2:
        return None

    ordered = []
    for pt in intersections:
        proj = float(np.dot(np.asarray(pt, dtype=float) - centroid, direction))
        ordered.append((proj, np.asarray(pt, dtype=float)))
    ordered.sort(key=lambda item: item[0])

    negative_side = [(proj, pt) for proj, pt in ordered if proj <= 0.0]
    positive_side = [(proj, pt) for proj, pt in ordered if proj >= 0.0]

    if negative_side:
        bottom = min(negative_side, key=lambda item: np.linalg.norm(item[1] - bottom_hint))[1]
    else:
        bottom = min(ordered, key=lambda item: np.linalg.norm(item[1] - bottom_hint))[1]

    if positive_side:
        top = max(positive_side, key=lambda item: item[0])[1]
    else:
        top = max(ordered, key=lambda item: np.linalg.norm(item[1] - bottom))[1]

    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1.0:
        return None

    return {
        'bottom': np.asarray(bottom, dtype=float),
        'top': np.asarray(top, dtype=float),
        'axis_length': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
    }




def build_axis_from_anchor_and_direction(contour, bottom_hint, direction):
    """
    Build an axis through a bottom hint while keeping an existing axis direction.

    This keeps the moments-derived direction as the geometric authority. A tip
    candidate may move the line onto the visible tip, but it cannot rotate the
    main axis toward a small protrusion or noise point.
    """
    bottom_hint = np.asarray(bottom_hint, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm

    half_extent = max(float(np.linalg.norm(np.ptp(contour, axis=0))), 1.0) * 2.0 + 20.0
    from shapely.geometry import LineString
    line = LineString([
        bottom_hint - float(half_extent) * direction,
        bottom_hint + float(half_extent) * direction,
    ])
    intersections = find_contour_intersections(contour, line)
    intersections = deduplicate_intersections(intersections, bottom_hint, direction)
    if len(intersections) < 2:
        return None

    ordered = []
    for pt in intersections:
        pt = np.asarray(pt, dtype=float)
        proj = float(np.dot(pt - bottom_hint, direction))
        ordered.append((proj, pt))
    ordered.sort(key=lambda item: item[0])

    bottom = min(ordered, key=lambda item: np.linalg.norm(item[1] - bottom_hint))[1]
    top = max(ordered, key=lambda item: np.linalg.norm(item[1] - bottom))[1]

    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1.0:
        return None

    return {
        'bottom': np.asarray(bottom, dtype=float),
        'top': np.asarray(top, dtype=float),
        'axis_length': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
    }




def angle_between_axis_dirs_deg(axis_a, axis_b):
    """Smallest angle between two axis direction vectors."""
    try:
        a = np.asarray(axis_a, dtype=float)
        b = np.asarray(axis_b, dtype=float)
    except Exception:
        return None
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < 1e-6 or norm_b < 1e-6:
        return None
    cosang = float(np.clip(abs(np.dot(a / norm_a, b / norm_b)), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))




def compute_bottom_roughness_metrics(contour, axis_bottom, axis_top, centroid, config):
    """
    Penalize small ragged protrusions that look sharp but are not stable tips.

    The main signals are:
      - distance from raw contour to a smoothed contour at the same outline index
      - isolated radial spike compared with neighboring outline points
      - very narrow support/neck width near the candidate bottom
    """
    enabled = bool(config.get('bottom_roughness_filter_enabled', True))
    default = {
        'bottom_roughness_enabled': bool(enabled),
        'bottom_roughness_penalty': 0.0,
        'bottom_roughness_indicator_count': 0,
        'bottom_roughness_reject': False,
        'bottom_roughness_smooth_offset_px': 0.0,
        'bottom_roughness_smooth_offset_fraction': 0.0,
        'bottom_roughness_radial_spike_px': 0.0,
        'bottom_roughness_radial_spike_fraction': 0.0,
        'bottom_roughness_support_ratio': 1.0,
        'bottom_roughness_neck_score': 0.0,
    }
    if not enabled:
        return default

    pts = get_open_contour(contour)
    if pts is None or len(pts) < 8:
        return default

    axis_bottom = np.asarray(axis_bottom, dtype=float)
    axis_top = np.asarray(axis_top, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(axis_bottom, axis_top)
    if axis_length < 1.0:
        return default

    idx = nearest_contour_index(pts, axis_bottom)
    smooth_pts = smooth_closed_contour(
        pts,
        window=int(config.get('bottom_smooth_contour_window', 11)),
        iterations=int(config.get('bottom_smooth_contour_iterations', 2)),
    )
    smooth_ref = smooth_pts[idx] if len(smooth_pts) == len(pts) else smooth_pts[nearest_contour_index(smooth_pts, axis_bottom)]
    smooth_offset_px = float(np.linalg.norm(pts[idx] - smooth_ref))
    smooth_offset_fraction = float(smooth_offset_px / max(axis_length, 1e-6))

    neighbor_window = max(3, int(config.get('bottom_roughness_neighbor_window', 14)))
    skip_center = max(1, int(config.get('bottom_roughness_skip_center_points', 2)))
    radial = np.linalg.norm(pts - centroid, axis=1)
    neighbor_values = []
    for offset in range(-neighbor_window, neighbor_window + 1):
        if abs(offset) <= skip_center:
            continue
        neighbor_values.append(float(radial[(idx + offset) % len(radial)]))
    neighbor_median = float(np.median(neighbor_values)) if neighbor_values else float(radial[idx])
    radial_spike_px = max(0.0, float(radial[idx]) - neighbor_median)
    radial_spike_fraction = float(radial_spike_px / max(axis_length, 1e-6))

    support_fractions = config.get('bottom_roughness_support_fractions', [0.06, 0.12])
    support_widths = []
    for fraction in support_fractions:
        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            continue
        width, _, _ = width_at_fraction(pts, axis_bottom, axis_dir, perp_dir, fraction, axis_length)
        support_widths.append(float(width))
    mid_width, _, _ = width_at_fraction(pts, axis_bottom, axis_dir, perp_dir, 0.50, axis_length)
    if mid_width > 1e-6 and support_widths:
        support_ratio = float(max(support_widths) / max(float(mid_width), 1e-6))
    else:
        support_ratio = 1.0

    smooth_threshold = max(1e-6, float(config.get('bottom_roughness_smooth_offset_fraction', 0.035)))
    spike_threshold = max(1e-6, float(config.get('bottom_roughness_radial_spike_fraction', 0.030)))
    min_support_ratio = max(1e-6, float(config.get('bottom_roughness_min_support_ratio', 0.12)))

    smooth_score = max(0.0, smooth_offset_fraction / smooth_threshold - 1.0)
    spike_score = max(0.0, radial_spike_fraction / spike_threshold - 1.0)
    neck_score = max(0.0, (min_support_ratio - support_ratio) / min_support_ratio)
    indicator_count = int(smooth_score > 0.0) + int(spike_score > 0.0) + int(neck_score > 0.0)

    penalty_weight = float(config.get('bottom_roughness_penalty_weight', 0.80))
    smooth_weight = float(config.get('bottom_roughness_smooth_offset_weight', 0.45))
    spike_weight = float(config.get('bottom_roughness_radial_spike_weight', 0.35))
    neck_weight = float(config.get('bottom_roughness_neck_weight', 0.20))
    penalty = penalty_weight * (
        smooth_weight * smooth_score
        + spike_weight * spike_score
        + neck_weight * neck_score
    )

    min_bad_metrics = int(config.get('bottom_roughness_reject_min_bad_metrics', 2))
    severe_score = float(config.get('bottom_roughness_reject_severe_score', 2.0))
    reject = bool(
        config.get('bottom_roughness_reject_enabled', True)
        and (
            indicator_count >= min_bad_metrics
            or max(smooth_score, spike_score) >= severe_score
        )
    )

    return {
        'bottom_roughness_enabled': True,
        'bottom_roughness_penalty': float(penalty),
        'bottom_roughness_indicator_count': int(indicator_count),
        'bottom_roughness_reject': bool(reject),
        'bottom_roughness_smooth_offset_px': float(smooth_offset_px),
        'bottom_roughness_smooth_offset_fraction': float(smooth_offset_fraction),
        'bottom_roughness_radial_spike_px': float(radial_spike_px),
        'bottom_roughness_radial_spike_fraction': float(radial_spike_fraction),
        'bottom_roughness_support_ratio': float(support_ratio),
        'bottom_roughness_neck_score': float(neck_score),
    }




def score_axis_for_local_refinement(contour, axis_bottom, axis_top, initial_bottom,
                                    initial_top, centroid, sharpness_profiles, config):
    """Score one candidate axis for the local bottom-refinement step."""
    axis_bottom = np.asarray(axis_bottom, dtype=float)
    axis_top = np.asarray(axis_top, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    if np.linalg.norm(axis_top - axis_bottom) < 1.0:
        return None

    base_sharpness = sharpness_profiles[0][1] if sharpness_profiles else np.zeros(len(contour), dtype=float)
    sharpness_radius = int(config.get('endpoint_sharpness_neighbor_radius', 2))
    bottom_sharpness, bottom_idx = sharpness_around_point(
        contour,
        base_sharpness,
        axis_bottom,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )
    top_sharpness, top_idx = sharpness_around_point(
        contour,
        base_sharpness,
        axis_top,
        neighbor_radius=sharpness_radius,
        multiscale_profiles=sharpness_profiles,
    )
    base_score = score_bottom_endpoint_candidate(
        contour,
        axis_bottom,
        axis_top,
        bottom_sharpness,
        config,
    )

    _, axis_length, axis_dir, perp_dir = build_axis_frame(axis_bottom, axis_top)
    _, initial_axis_length, initial_axis_dir, _ = build_axis_frame(
        np.asarray(initial_bottom, dtype=float),
        np.asarray(initial_top, dtype=float),
    )
    if initial_axis_length < 1e-6:
        initial_axis_length = axis_length

    distance_px = float(np.linalg.norm(axis_bottom - np.asarray(initial_bottom, dtype=float)))
    max_distance_px = max(
        float(config.get('bottom_local_refine_max_distance_px', 0.0)),
        float(initial_axis_length) * float(config.get('bottom_local_refine_max_distance_fraction', 0.10)),
        1.0,
    )
    distance_penalty = min(2.0, distance_px / max(max_distance_px, 1e-6))

    axis_length_ratio = float(axis_length / max(initial_axis_length, 1e-6))
    axis_alignment = float(abs(np.dot(axis_dir, initial_axis_dir)))
    axis_angle_gap = angle_between_axis_dirs_deg(axis_dir, initial_axis_dir)
    max_angle_gap = float(config.get('bottom_local_refine_max_angle_gap_deg', 20.0))
    if axis_angle_gap is not None and axis_angle_gap > max_angle_gap:
        return None

    initial_radial = float(np.linalg.norm(np.asarray(initial_bottom, dtype=float) - centroid))
    radial_distance = float(np.linalg.norm(axis_bottom - centroid))
    radial_ratio = float(radial_distance / max(initial_radial, 1e-6))
    roughness_metrics = compute_bottom_roughness_metrics(
        contour,
        axis_bottom,
        axis_top,
        centroid,
        config,
    )
    if bool(roughness_metrics.get('bottom_roughness_reject', False)):
        return None

    distance_weight = float(config.get('bottom_local_refine_distance_weight', 0.35))
    length_weight = float(config.get('bottom_local_refine_length_weight', 0.20))
    alignment_weight = float(config.get('bottom_local_refine_alignment_weight', 0.20))
    radial_weight = float(config.get('bottom_local_refine_radial_distance_weight', 0.25))

    total_score = (
        float(base_score['score'])
        + length_weight * axis_length_ratio
        + alignment_weight * axis_alignment
        + radial_weight * min(1.25, radial_ratio)
        - distance_weight * distance_penalty
        - float(roughness_metrics.get('bottom_roughness_penalty', 0.0))
    )

    result = {
        'bottom': axis_bottom,
        'top': axis_top,
        'bottom_idx': int(bottom_idx),
        'top_idx': int(top_idx),
        'bottom_sharpness': float(bottom_sharpness),
        'top_sharpness': float(top_sharpness),
        'sharpness_gap': float(bottom_sharpness - top_sharpness),
        'bottom_taper_score': float(base_score['taper_score']),
        'top_taper_score': 0.0,
        'bottom_monotonic_score': float(base_score['monotonic_score']),
        'top_monotonic_score': 0.0,
        'bottom_candidate_score': float(total_score),
        'top_candidate_score': 0.0,
        'score': float(total_score),
        'axis_length': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'distance_px': float(distance_px),
        'axis_length_ratio': float(axis_length_ratio),
        'axis_alignment': float(axis_alignment),
        'axis_angle_gap_deg': axis_angle_gap,
        'radial_distance_px': float(radial_distance),
        'radial_distance_ratio': float(radial_ratio),
    }
    result.update(roughness_metrics)
    return result




def score_local_bottom_refinement_candidate(contour, candidate_bottom, centroid, initial_bottom,
                                            initial_top, sharpness_profiles, config):
    """Score a nearby contour point as a local bottom correction candidate."""
    if bool(config.get('bottom_local_refine_lock_axis_direction', True)):
        _, _, initial_axis_dir, _ = build_axis_frame(
            np.asarray(initial_bottom, dtype=float),
            np.asarray(initial_top, dtype=float),
        )
        candidate_axis = build_axis_from_anchor_and_direction(contour, candidate_bottom, initial_axis_dir)
    else:
        candidate_axis = build_axis_from_bottom_and_centroid(contour, candidate_bottom, centroid)
    if candidate_axis is None:
        return None

    return score_axis_for_local_refinement(
        contour,
        candidate_axis['bottom'],
        candidate_axis['top'],
        initial_bottom,
        initial_top,
        centroid,
        sharpness_profiles,
        config,
    )




def score_method_axis_endpoint_refinement(contour, candidate_bottom, candidate_top,
                                          initial_bottom, initial_top, centroid,
                                          sharpness_profiles, config,
                                          max_distance_px, max_angle_gap):
    """
    Score one corrected method-axis proposal against the original method chord.

    The original method axis is treated as an unordered pair of endpoints here:
    if the sharper corrected endpoint is closer to the initial top than the
    initial bottom, we still allow it. This lets each raw method axis move toward a
    nearby sharp/concave bottom point without being trapped by the first
    endpoint orientation.
    """
    candidate_bottom = np.asarray(candidate_bottom, dtype=float)
    candidate_top = np.asarray(candidate_top, dtype=float)
    initial_bottom = np.asarray(initial_bottom, dtype=float)
    initial_top = np.asarray(initial_top, dtype=float)

    dist_to_bottom = float(np.linalg.norm(candidate_bottom - initial_bottom))
    dist_to_top = float(np.linalg.norm(candidate_bottom - initial_top))
    if dist_to_bottom <= dist_to_top:
        reference_bottom = initial_bottom
        reference_top = initial_top
        endpoint_distance = dist_to_bottom
    else:
        reference_bottom = initial_top
        reference_top = initial_bottom
        endpoint_distance = dist_to_top

    if endpoint_distance > max_distance_px:
        return None

    _, _, candidate_axis_dir, _ = build_axis_frame(candidate_bottom, candidate_top)
    _, _, reference_axis_dir, _ = build_axis_frame(reference_bottom, reference_top)
    angle_gap = angle_between_axis_dirs_deg(candidate_axis_dir, reference_axis_dir)
    if angle_gap is not None and angle_gap > max_angle_gap:
        return None

    refine_config = dict(config)
    refine_config['bottom_local_refine_max_distance_px'] = float(max_distance_px)
    refine_config['bottom_local_refine_max_angle_gap_deg'] = float(max_angle_gap)
    scored = score_axis_for_local_refinement(
        contour,
        candidate_bottom,
        candidate_top,
        reference_bottom,
        reference_top,
        centroid,
        sharpness_profiles,
        refine_config,
    )
    if scored is None:
        return None
    scored['endpoint_refine_distance_px'] = float(endpoint_distance)
    scored['endpoint_refine_angle_gap_deg'] = angle_gap
    return scored




def locally_refine_bottom_point(contour, centroid, initial_info, config):
    """Apply a small, score-based local correction around the current bottom tip."""
    if not bool(config.get('bottom_local_refine_enabled', True)):
        return None

    initial_bottom = np.asarray(initial_info.get('bottom'), dtype=float)
    initial_top = np.asarray(initial_info.get('top'), dtype=float)
    if np.linalg.norm(initial_top - initial_bottom) < 1.0:
        return None

    pts = get_open_contour(contour)
    sharpness_profiles = build_multiscale_sharpness_profiles(
        pts,
        base_window=int(config.get('bottom_sharpness_window', 5)),
        scale_factors=config.get('endpoint_sharpness_scale_factors', [1.0, 2.0, 3.5]),
    )
    baseline = score_axis_for_local_refinement(
        pts,
        initial_bottom,
        initial_top,
        initial_bottom,
        initial_top,
        centroid,
        sharpness_profiles,
        config,
    )
    if baseline is None:
        return None

    neighbor_window = max(1, int(config.get('bottom_local_refine_neighbor_window', 16)))
    hint_pts = get_bottom_refinement_hint_contour(pts, config)
    if hint_pts is None or len(hint_pts) != len(pts):
        hint_pts = pts
    initial_idx = nearest_contour_index(hint_pts, initial_bottom)
    best = baseline

    for offset in range(-neighbor_window, neighbor_window + 1):
        idx = (initial_idx + offset) % len(pts)
        candidate_bottom = np.asarray(hint_pts[idx], dtype=float)
        candidate = score_local_bottom_refinement_candidate(
            pts,
            candidate_bottom,
            centroid,
            initial_bottom,
            initial_top,
            sharpness_profiles,
            config,
        )
        if candidate is None:
            continue
        if candidate['distance_px'] > max(
            float(config.get('bottom_local_refine_max_distance_px', 0.0)),
            float(baseline['axis_length']) * float(config.get('bottom_local_refine_max_distance_fraction', 0.10)),
            1.0,
        ):
            continue
        if candidate['score'] > best['score']:
            best = candidate

    min_gain = float(config.get('bottom_local_refine_min_score_gain', 0.02))
    if best['score'] <= baseline['score'] + min_gain:
        return None

    best['method'] = f"{initial_info.get('method', 'principal_axis')}_local_bottom_refine"
    best['refined'] = True
    best['baseline_score'] = float(baseline['score'])
    return best




def refine_bottom_point(contour, config, mask_binary=None, image_bgr=None,
                        axis_predictor=None):
    """
    Determine bottom/top from the SAM mask centroid and configured axis source.

    In the active pipeline the direction comes from the ResNet axis model.
    A contour-derived fallback is kept only for deliberately non-ResNet configs.
    """
    pts = get_analysis_contour(contour, config)
    axis_source = str(config.get('axis_source', 'resnet')).strip().lower()
    if axis_source in {'resnet', 'resnet_axis_model', 'model'}:
        axis_info = derive_axis_from_resnet_model(
            pts,
            mask_binary,
            config,
            image_bgr=image_bgr,
            axis_predictor=axis_predictor,
        )
    else:
        axis_info = derive_axis_from_mask_and_contour(
            pts,
            mask_binary,
            config,
            image_bgr=image_bgr,
        )
    centroid = np.asarray(axis_info['centroid'], dtype=float)

    refined_axis = None
    if not bool(axis_info.get('skip_local_refine', False)):
        refined_axis = locally_refine_bottom_point(pts, centroid, axis_info, config)
    if refined_axis is not None:
        axis_info = {
            **axis_info,
            **refined_axis,
            'axis_angle_deg': (
                math.degrees(math.atan2(refined_axis['axis_dir'][1], refined_axis['axis_dir'][0])) + 360.0
            ) % 180.0,
        }

    bottom = np.asarray(axis_info['bottom'], dtype=float)
    top = np.asarray(axis_info['top'], dtype=float)
    axis_widths = summarize_axis_widths(pts, bottom, top, config)

    return bottom, {
        'method': axis_info.get('method', 'moments_principal_axis'),
        'bottom': bottom,
        'top': top,
        'centroid': centroid,
        'centroid_source': axis_info.get('centroid_source', ''),
        'axis_dir': axis_widths['axis_dir'],
        'perp_dir': axis_widths['perp_dir'],
        'axis_length': axis_widths['axis_length'],
        'axis_angle_deg': axis_info.get('axis_angle_deg'),
        'score': axis_info.get('score'),
        'bottom_candidate_score': axis_info.get('bottom_candidate_score'),
        'top_candidate_score': axis_info.get('top_candidate_score'),
        'near_width': axis_widths['near_width'],
        'mid_width': axis_widths['mid_width'],
        'upper_width': axis_widths['upper_width'],
        'candidate_count': int(axis_info.get('candidate_count', 0)),
        'top_method': 'sharpness_plus_taper_on_principal_axis',
        'bottom_sharpness': axis_info.get('bottom_sharpness'),
        'top_sharpness': axis_info.get('top_sharpness'),
        'sharpness_gap': axis_info.get('sharpness_gap'),
        'bottom_taper_score': axis_info.get('bottom_taper_score'),
        'top_taper_score': axis_info.get('top_taper_score'),
        'bottom_monotonic_score': axis_info.get('bottom_monotonic_score'),
        'top_monotonic_score': axis_info.get('top_monotonic_score'),
        'bottom_refined': bool(axis_info.get('refined', False)),
        'baseline_score': axis_info.get('baseline_score'),
        'bottom_refine_distance_px': axis_info.get('distance_px'),
        'bottom_refine_axis_length_ratio': axis_info.get('axis_length_ratio'),
        'bottom_refine_axis_alignment': axis_info.get('axis_alignment'),
        'bottom_refine_axis_angle_gap_deg': axis_info.get('axis_angle_gap_deg'),
        'bottom_refine_radial_distance_ratio': axis_info.get('radial_distance_ratio'),
        'bottom_roughness_penalty': axis_info.get('bottom_roughness_penalty'),
        'bottom_roughness_indicator_count': axis_info.get('bottom_roughness_indicator_count'),
        'bottom_roughness_reject': axis_info.get('bottom_roughness_reject'),
        'bottom_roughness_smooth_offset_px': axis_info.get('bottom_roughness_smooth_offset_px'),
        'bottom_roughness_smooth_offset_fraction': axis_info.get('bottom_roughness_smooth_offset_fraction'),
        'bottom_roughness_radial_spike_px': axis_info.get('bottom_roughness_radial_spike_px'),
        'bottom_roughness_radial_spike_fraction': axis_info.get('bottom_roughness_radial_spike_fraction'),
        'bottom_roughness_support_ratio': axis_info.get('bottom_roughness_support_ratio'),
        'bottom_roughness_neck_score': axis_info.get('bottom_roughness_neck_score'),
        'skip_local_refine': bool(axis_info.get('skip_local_refine', False)),
        'axis_selection_reason': axis_info.get('axis_selection_reason', ''),
        'low_confidence_axis': bool(axis_info.get('low_confidence_axis', False)),
        'low_confidence_reason': axis_info.get('low_confidence_reason', ''),
        'candidate_score_margin': axis_info.get('candidate_score_margin'),
        'candidate_rank': axis_info.get('candidate_rank'),
        'candidate_scores': axis_info.get('candidate_scores', []),
        'method_axis_candidate_scores': axis_info.get('method_axis_candidate_scores', []),
        'candidate_base_axes': axis_info.get('candidate_base_axes', []),
        'area_balance_penalty': axis_info.get('area_balance_penalty'),
        'width_symmetry_penalty': axis_info.get('width_symmetry_penalty'),
        'color_symmetry_penalty': axis_info.get('color_symmetry_penalty'),
        'length_penalty': axis_info.get('length_penalty'),
        'endpoint_penalty': axis_info.get('endpoint_penalty'),
        'color_valid_pairs': axis_info.get('color_valid_pairs'),
        'width_valid_samples': axis_info.get('width_valid_samples'),
        'axis_selection_circularity': axis_info.get('axis_selection_circularity'),
        'axis_model_pred_theta_deg': axis_info.get('axis_model_pred_theta_deg'),
        'axis_model_pred_theta_rad': axis_info.get('axis_model_pred_theta_rad'),
        'axis_model_pred_cos_theta': axis_info.get('axis_model_pred_cos_theta'),
        'axis_model_pred_sin_theta': axis_info.get('axis_model_pred_sin_theta'),
        'axis_model_pred_vector_norm': axis_info.get('axis_model_pred_vector_norm'),
        'axis_model_weights_path': axis_info.get('axis_model_weights_path'),
        'axis_model_imgsz': axis_info.get('axis_model_imgsz'),
        'axis_model_preprocess': axis_info.get('axis_model_preprocess'),
        'moments_axis_bottom': axis_info.get('moments_axis_bottom'),
        'moments_axis_top': axis_info.get('moments_axis_top'),
        'moments_axis_length_px': axis_info.get('moments_axis_length_px'),
        'moments_axis_angle_deg': axis_info.get('moments_axis_angle_deg'),
        'moments_axis_total_score': axis_info.get('moments_axis_total_score'),
        'longest_chord_bottom': axis_info.get('longest_chord_bottom'),
        'longest_chord_top': axis_info.get('longest_chord_top'),
        'longest_chord_length_px': axis_info.get('longest_chord_length_px'),
        'longest_chord_angle_deg': axis_info.get('longest_chord_angle_deg'),
        'longest_chord_total_score': axis_info.get('longest_chord_total_score'),
        'shortest_chord_bottom': axis_info.get('shortest_chord_bottom'),
        'shortest_chord_top': axis_info.get('shortest_chord_top'),
        'shortest_chord_length_px': axis_info.get('shortest_chord_length_px'),
        'shortest_chord_angle_deg': axis_info.get('shortest_chord_angle_deg'),
        'shortest_chord_total_score': axis_info.get('shortest_chord_total_score'),
    }




def safe_candidate_point(candidate, key):
    if not candidate:
        return None
    value = candidate.get(key)
    if not point_is_valid(value):
        return None
    return np.asarray(value, dtype=float)




def safe_candidate_float(candidate, key):
    if not candidate:
        return None
    value = candidate.get(key)
    if value is None or value == '':
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None




def normalize_axis_angle_deg(angle_deg):
    return float(float(angle_deg) % 180.0)




def axis_angle_distance_deg(angle_a, angle_b):
    diff = abs(normalize_axis_angle_deg(angle_a) - normalize_axis_angle_deg(angle_b))
    return float(min(diff, 180.0 - diff))




def direction_from_angle_deg(angle_deg):
    theta = math.radians(normalize_axis_angle_deg(angle_deg))
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)




def angle_from_direction(direction):
    direction = np.asarray(direction, dtype=float)
    if np.linalg.norm(direction) < 1e-6:
        return 0.0
    return normalize_axis_angle_deg(math.degrees(math.atan2(direction[1], direction[0])))




def robust_penalty_stat(values, trim_fraction=0.15, mode='trimmed_mean'):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    if str(mode).lower() == 'median':
        return float(np.median(arr))
    arr = np.sort(arr)
    trim_count = int(math.floor(arr.size * float(np.clip(trim_fraction, 0.0, 0.45))))
    if trim_count > 0 and arr.size - trim_count >= 1:
        arr = arr[:arr.size - trim_count]
    return float(np.mean(arr))




def mask_contains_point(mask_binary, point):
    if mask_binary is None or not point_is_valid(point):
        return False
    h, w = mask_binary.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask_binary[y, x] > 0)




def ray_distance_to_mask(mask_binary, origin, direction, max_distance, step=1.0):
    if mask_binary is None:
        return 0.0
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return 0.0
    direction = direction / norm
    max_distance = max(float(max_distance), 0.0)
    step = max(float(step), 0.25)
    last_inside = 0.0
    distance = step
    h, w = mask_binary.shape[:2]
    while distance <= max_distance + 1e-6:
        point = origin + distance * direction
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if x < 0 or y < 0 or x >= w or y >= h or mask_binary[y, x] == 0:
            break
        last_inside = distance
        distance += step
    return float(last_inside)




def centroid_axis_endpoints(contour, centroid, angle_deg):
    """Intersect one centroid-crossing angle with the contour and return two endpoints."""
    pts = get_open_contour(contour)
    if pts is None or len(pts) < 3 or not point_is_valid(centroid):
        return None

    centroid = np.asarray(centroid, dtype=float)
    direction = direction_from_angle_deg(angle_deg)
    half_extent = max(float(np.linalg.norm(np.ptp(pts, axis=0))), 1.0) * 2.0 + 20.0

    from shapely.geometry import LineString
    line = LineString([
        centroid - float(half_extent) * direction,
        centroid + float(half_extent) * direction,
    ])
    intersections = find_contour_intersections(pts, line)
    intersections = deduplicate_intersections(intersections, centroid, direction)
    if len(intersections) < 2:
        return None

    ordered = sorted(
        [
            (float(np.dot(np.asarray(pt, dtype=float) - centroid, direction)),
             np.asarray(pt, dtype=float))
            for pt in intersections
        ],
        key=lambda item: item[0],
    )
    negative = [(proj, pt) for proj, pt in ordered if proj <= 0.0]
    positive = [(proj, pt) for proj, pt in ordered if proj >= 0.0]
    if negative and positive:
        point_a = negative[0][1]
        point_b = positive[-1][1]
    else:
        point_a = ordered[0][1]
        point_b = ordered[-1][1]

    length = float(np.linalg.norm(point_b - point_a))
    if length < 1.0:
        return None
    return point_a, point_b, direction, length




def build_axis_candidate_from_angle(contour, centroid, angle_deg, sharpness_profiles,
                                    config, base_source, base_angle_deg, scan_offset_deg,
                                    base_rank=0):
    endpoint_info = centroid_axis_endpoints(contour, centroid, angle_deg)
    if endpoint_info is None:
        return None
    point_a, point_b, _, chord_length = endpoint_info
    endpoint_choice = choose_bottom_top_from_endpoints(
        contour,
        point_a,
        point_b,
        sharpness_profiles,
        config,
    )
    bottom = endpoint_choice['bottom']
    top = endpoint_choice['top']
    axis_vec, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1.0:
        return None

    return {
        'bottom': bottom,
        'top': top,
        'bottom_idx': int(endpoint_choice['bottom_idx']),
        'top_idx': int(endpoint_choice['top_idx']),
        'bottom_sharpness': float(endpoint_choice['bottom_sharpness']),
        'top_sharpness': float(endpoint_choice['top_sharpness']),
        'sharpness_gap': float(endpoint_choice['sharpness_gap']),
        'length_px': float(axis_length),
        'raw_chord_length_px': float(chord_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'axis_angle_deg': angle_from_direction(axis_dir),
        'raw_angle_deg': normalize_axis_angle_deg(angle_deg),
        'base_angle_deg': normalize_axis_angle_deg(base_angle_deg),
        'scan_offset_deg': float(scan_offset_deg),
        'base_source': str(base_source),
        'base_rank': int(base_rank),
        'method': f'{base_source}_corrected_axis',
        'score': float(endpoint_choice.get('bottom_candidate_score', 0.0)),
        'bottom_candidate_score': float(endpoint_choice.get('bottom_candidate_score', 0.0)),
        'top_candidate_score': float(endpoint_choice.get('top_candidate_score', 0.0)),
        'bottom_taper_score': float(endpoint_choice.get('bottom_taper_score', 0.0)),
        'top_taper_score': float(endpoint_choice.get('top_taper_score', 0.0)),
        'bottom_monotonic_score': float(endpoint_choice.get('bottom_monotonic_score', 0.0)),
        'top_monotonic_score': float(endpoint_choice.get('top_monotonic_score', 0.0)),
    }




def maybe_refine_candidate_axis_endpoint(contour, centroid, candidate, sharpness_profiles, config):
    """Correct one method axis toward nearby sharp/concave bottom endpoints."""
    if candidate is None or not bool(config.get('candidate_endpoint_refine_enabled', True)):
        return candidate

    pts = get_open_contour(contour)
    initial_bottom = np.asarray(candidate['bottom'], dtype=float)
    initial_top = np.asarray(candidate['top'], dtype=float)
    _, initial_length, initial_axis_dir, _ = build_axis_frame(initial_bottom, initial_top)
    if initial_length < 1.0:
        return candidate

    max_distance_px = max(
        float(config.get('candidate_endpoint_refine_max_distance_px', 0.0) or 0.0),
        initial_length * float(config.get('candidate_endpoint_refine_max_distance_fraction', 0.12)),
        1.0,
    )
    max_angle_gap = float(config.get('candidate_endpoint_refine_max_angle_gap_deg', 10.0))
    neighbor_window = max(1, int(config.get('candidate_endpoint_refine_neighbor_window', 14)))
    angle_scan_deg = max(0.0, float(config.get('candidate_endpoint_refine_angle_scan_deg', max_angle_gap)))
    angle_step = max(0.5, float(config.get('candidate_endpoint_refine_angle_step_deg', 1.0)))
    min_gain = float(config.get('candidate_endpoint_refine_min_score_gain', 0.005))

    hint_pts = get_bottom_refinement_hint_contour(pts, config)
    if hint_pts is None or len(hint_pts) != len(pts):
        hint_pts = pts

    baseline = score_method_axis_endpoint_refinement(
        pts,
        initial_bottom,
        initial_top,
        initial_bottom,
        initial_top,
        centroid,
        sharpness_profiles,
        config,
        max_distance_px=max_distance_px,
        max_angle_gap=max_angle_gap,
    )
    if baseline is None:
        return candidate

    best = baseline
    best_source = 'baseline'

    for anchor in (initial_bottom, initial_top):
        initial_idx = nearest_contour_index(hint_pts, anchor)
        for offset in range(-neighbor_window, neighbor_window + 1):
            idx = (initial_idx + offset) % len(pts)
            candidate_bottom = np.asarray(hint_pts[idx], dtype=float)
            candidate_axis = build_axis_from_bottom_and_centroid(pts, candidate_bottom, centroid)
            if candidate_axis is None:
                continue
            scored = score_method_axis_endpoint_refinement(
                pts,
                candidate_axis['bottom'],
                candidate_axis['top'],
                initial_bottom,
                initial_top,
                centroid,
                sharpness_profiles,
                config,
                max_distance_px=max_distance_px,
                max_angle_gap=max_angle_gap,
            )
            if scored is not None and scored['score'] > best['score']:
                best = scored
                best_source = 'contour_neighbor'

    base_angle = float(candidate.get('axis_angle_deg', candidate.get('base_angle_deg', 0.0)))
    if angle_scan_deg > 0.0:
        offsets = np.arange(-angle_scan_deg, angle_scan_deg + 0.5 * angle_step, angle_step)
        for offset in offsets:
            angle = normalize_axis_angle_deg(base_angle + float(offset))
            endpoint_info = centroid_axis_endpoints(pts, centroid, angle)
            if endpoint_info is None:
                continue
            point_a, point_b, _, _ = endpoint_info
            endpoint_choice = choose_bottom_top_from_endpoints(
                pts,
                point_a,
                point_b,
                sharpness_profiles,
                config,
            )
            scored = score_method_axis_endpoint_refinement(
                pts,
                endpoint_choice['bottom'],
                endpoint_choice['top'],
                initial_bottom,
                initial_top,
                centroid,
                sharpness_profiles,
                config,
                max_distance_px=max_distance_px,
                max_angle_gap=max_angle_gap,
            )
            if scored is not None and scored['score'] > best['score']:
                best = scored
                best_source = 'angle_scan'

    if best is baseline or best['score'] <= baseline['score'] + min_gain:
        return candidate

    bottom = np.asarray(best['bottom'], dtype=float)
    top = np.asarray(best['top'], dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    refined = {**candidate}
    refined.update({
        'bottom': bottom,
        'top': top,
        'bottom_idx': int(best.get('bottom_idx', nearest_contour_index(pts, bottom))),
        'top_idx': int(best.get('top_idx', nearest_contour_index(pts, top))),
        'length_px': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'axis_angle_deg': angle_from_direction(axis_dir),
        'endpoint_refined': True,
        'endpoint_refine_source': best_source,
        'endpoint_refine_distance_px': float(best.get('endpoint_refine_distance_px', best.get('distance_px', 0.0))),
        'endpoint_refine_angle_gap_deg': best.get('endpoint_refine_angle_gap_deg', best.get('axis_angle_gap_deg')),
        'bottom_sharpness': float(best.get('bottom_sharpness', candidate.get('bottom_sharpness', 0.0))),
        'top_sharpness': float(best.get('top_sharpness', candidate.get('top_sharpness', 0.0))),
        'sharpness_gap': float(best.get('sharpness_gap', candidate.get('sharpness_gap', 0.0))),
        'bottom_candidate_score': float(best.get('bottom_candidate_score', candidate.get('bottom_candidate_score', 0.0))),
        'bottom_taper_score': float(best.get('bottom_taper_score', candidate.get('bottom_taper_score', 0.0))),
        'bottom_monotonic_score': float(best.get('bottom_monotonic_score', candidate.get('bottom_monotonic_score', 0.0))),
    })
    return refined




def compute_mask_area_balance(mask_binary, centroid, perp_dir):
    if mask_binary is None or np.count_nonzero(mask_binary) == 0:
        return {
            'area_balance_penalty': 1.0,
            'area_balance_ratio': 0.0,
            'area_left_px': 0,
            'area_right_px': 0,
        }
    ys, xs = np.where(mask_binary > 0)
    if xs.size == 0:
        return {
            'area_balance_penalty': 1.0,
            'area_balance_ratio': 0.0,
            'area_left_px': 0,
            'area_right_px': 0,
        }
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    signed = np.dot(coords - np.asarray(centroid, dtype=float), np.asarray(perp_dir, dtype=float))
    eps = 0.25
    left = int(np.sum(signed < -eps))
    right = int(np.sum(signed > eps))
    total = left + right
    if total <= 0:
        penalty = 1.0
        ratio = 0.0
    else:
        penalty = abs(left - right) / float(total)
        ratio = min(left, right) / max(float(max(left, right)), 1.0)
    return {
        'area_balance_penalty': float(np.clip(penalty, 0.0, 1.0)),
        'area_balance_ratio': float(np.clip(ratio, 0.0, 1.0)),
        'area_left_px': left,
        'area_right_px': right,
    }




def compute_width_symmetry_penalty(mask_binary, bottom, top, config):
    bottom = np.asarray(bottom, dtype=float)
    top = np.asarray(top, dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if mask_binary is None or axis_length < 1.0:
        return {'width_symmetry_penalty': 1.0, 'width_valid_samples': 0}

    sample_count = max(8, int(config.get('candidate_width_samples', config.get('num_width_samples', 100))))
    low = float(config.get('candidate_width_fraction_min', 0.02))
    high = float(config.get('candidate_width_fraction_max', 0.98))
    low = float(np.clip(low, 0.0, 0.45))
    high = float(np.clip(high, low + 0.05, 1.0))
    max_dist = max(float(np.linalg.norm(np.ptp(np.column_stack(np.where(mask_binary > 0)), axis=0))) if np.count_nonzero(mask_binary) else 1.0, axis_length) + 10.0
    step = float(config.get('candidate_width_ray_step_px', 1.0))
    asymmetries = []
    widths = []

    for fraction in np.linspace(low, high, sample_count):
        pos = bottom + float(fraction) * axis_length * axis_dir
        if not mask_contains_point(mask_binary, pos):
            continue
        left = ray_distance_to_mask(mask_binary, pos, -perp_dir, max_dist, step=step)
        right = ray_distance_to_mask(mask_binary, pos, perp_dir, max_dist, step=step)
        width = left + right
        if width <= 1e-6:
            continue
        asymmetries.append(abs(left - right) / width)
        widths.append(width)

    stat = robust_penalty_stat(
        asymmetries,
        trim_fraction=float(config.get('candidate_width_trim_fraction', 0.15)),
        mode=config.get('candidate_width_stat', 'trimmed_mean'),
    )
    if stat is None:
        return {
            'width_symmetry_penalty': 1.0,
            'width_valid_samples': 0,
            'width_symmetry_median_asymmetry': None,
            'width_symmetry_mean_width_px': None,
        }
    return {
        'width_symmetry_penalty': float(np.clip(stat, 0.0, 1.0)),
        'width_valid_samples': int(len(asymmetries)),
        'width_symmetry_median_asymmetry': float(np.median(asymmetries)),
        'width_symmetry_mean_width_px': float(np.mean(widths)) if widths else None,
    }




def build_color_symmetry_context(image_bgr, mask_binary, config):
    if image_bgr is None or mask_binary is None:
        return {'available': False, 'lab': None, 'anomaly_mask': None}
    if image_bgr.shape[:2] != mask_binary.shape[:2]:
        image_bgr = cv2.resize(
            image_bgr,
            (mask_binary.shape[1], mask_binary.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fg = mask_binary > 0
    if not np.any(fg):
        return {'available': False, 'lab': lab, 'anomaly_mask': np.ones(mask_binary.shape, dtype=bool)}

    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    dark = L < float(config.get('color_anomaly_dark_l', 35.0))
    bright = L > float(config.get('color_anomaly_bright_l', 242.0))
    chroma_threshold = max(
        float(config.get('color_anomaly_extreme_chroma', 90.0)),
        float(np.percentile(chroma[fg], 99.0)) + float(config.get('color_anomaly_chroma_percentile_pad', 8.0)),
    )
    extreme_color = chroma > chroma_threshold

    median_kernel = max(3, int(config.get('color_anomaly_local_median_kernel', 7)))
    if median_kernel % 2 == 0:
        median_kernel += 1
    median_lab = cv2.medianBlur(lab.astype(np.uint8), median_kernel).astype(np.float32)
    local_delta = np.linalg.norm(lab - median_lab, axis=2)
    local_threshold = max(
        float(config.get('color_anomaly_local_delta_e', 28.0)),
        float(np.percentile(local_delta[fg], 98.0)),
    )
    local_outlier = local_delta > local_threshold

    anomaly = (dark | bright | extreme_color | local_outlier) & fg
    dilate_kernel = int(config.get('color_anomaly_dilate_kernel', 3))
    if dilate_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
        anomaly = cv2.dilate(anomaly.astype(np.uint8), kernel, iterations=1).astype(bool)
    anomaly = anomaly | (~fg)

    return {
        'available': True,
        'lab': lab,
        'anomaly_mask': anomaly,
        'anomaly_fraction': float(np.mean(anomaly[fg])) if np.any(fg) else 1.0,
    }




def sample_lab_at(lab, point):
    h, w = lab.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    return lab[y, x].astype(float)




def anomaly_at(anomaly_mask, point):
    h, w = anomaly_mask.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return True
    return bool(anomaly_mask[y, x])




def compute_color_symmetry_penalty(color_context, mask_binary, bottom, top, config):
    if not color_context.get('available', False):
        neutral = float(config.get('candidate_color_missing_penalty', 0.50))
        return {
            'color_symmetry_penalty': float(np.clip(neutral, 0.0, 1.0)),
            'color_valid_pairs': 0,
            'color_delta_median': None,
            'color_anomaly_fraction': None,
        }

    lab = color_context['lab']
    anomaly_mask = color_context['anomaly_mask']
    bottom = np.asarray(bottom, dtype=float)
    top = np.asarray(top, dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1.0:
        return {'color_symmetry_penalty': 1.0, 'color_valid_pairs': 0}

    sample_count = max(8, int(config.get('candidate_color_axis_samples', 80)))
    offsets_per_sample = max(2, int(config.get('candidate_color_offsets_per_sample', 5)))
    max_dist = max(float(np.linalg.norm(np.ptp(np.column_stack(np.where(mask_binary > 0)), axis=0))) if np.count_nonzero(mask_binary) else 1.0, axis_length) + 10.0
    ray_step = float(config.get('candidate_color_ray_step_px', 1.0))
    min_offset = float(config.get('candidate_color_min_offset_px', 1.5))
    deltas = []

    for fraction in np.linspace(0.04, 0.96, sample_count):
        pos = bottom + float(fraction) * axis_length * axis_dir
        if not mask_contains_point(mask_binary, pos):
            continue
        left = ray_distance_to_mask(mask_binary, pos, -perp_dir, max_dist, step=ray_step)
        right = ray_distance_to_mask(mask_binary, pos, perp_dir, max_dist, step=ray_step)
        pair_extent = min(left, right)
        if pair_extent <= min_offset:
            continue
        offsets = np.linspace(min_offset, pair_extent, offsets_per_sample)
        for offset in offsets:
            p_left = pos - float(offset) * perp_dir
            p_right = pos + float(offset) * perp_dir
            if anomaly_at(anomaly_mask, p_left) or anomaly_at(anomaly_mask, p_right):
                continue
            lab_left = sample_lab_at(lab, p_left)
            lab_right = sample_lab_at(lab, p_right)
            if lab_left is None or lab_right is None:
                continue
            deltas.append(float(np.linalg.norm(lab_left - lab_right)))

    stat = robust_penalty_stat(
        deltas,
        trim_fraction=float(config.get('candidate_color_trim_fraction', 0.15)),
        mode=config.get('candidate_color_stat', 'median'),
    )
    if stat is None:
        neutral = float(config.get('candidate_color_missing_penalty', 0.50))
        return {
            'color_symmetry_penalty': float(np.clip(neutral, 0.0, 1.0)),
            'color_valid_pairs': 0,
            'color_delta_median': None,
            'color_anomaly_fraction': color_context.get('anomaly_fraction'),
        }

    norm = max(1.0, float(config.get('candidate_color_delta_norm', 35.0)))
    return {
        'color_symmetry_penalty': float(np.clip(stat / norm, 0.0, 1.0)),
        'color_valid_pairs': int(len(deltas)),
        'color_delta_median': float(np.median(deltas)),
        'color_delta_trimmed': float(stat),
        'color_anomaly_fraction': color_context.get('anomaly_fraction'),
    }




def endpoint_local_shape_metrics(contour, mask_binary, endpoint, opposite, centroid,
                                 sharpness_profiles, config):
    pts = get_open_contour(contour)
    endpoint = np.asarray(endpoint, dtype=float)
    opposite = np.asarray(opposite, dtype=float)
    centroid = np.asarray(centroid, dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(endpoint, opposite)
    if axis_length < 1.0 or pts is None or len(pts) < 8:
        return {
            'roughness_score': 1.0,
            'sharpness': 0.0,
            'radial_ratio': 0.0,
            'support_ratio': 0.0,
        }

    base_sharpness = sharpness_profiles[0][1] if sharpness_profiles else np.zeros(len(pts), dtype=float)
    sharpness, idx = sharpness_around_point(
        pts,
        base_sharpness,
        endpoint,
        neighbor_radius=int(config.get('endpoint_sharpness_neighbor_radius', 2)),
        multiscale_profiles=sharpness_profiles,
    )
    smooth_pts = smooth_closed_contour(
        pts,
        window=int(config.get('bottom_smooth_contour_window', 11)),
        iterations=int(config.get('bottom_smooth_contour_iterations', 2)),
    )
    smooth_ref = smooth_pts[idx] if len(smooth_pts) == len(pts) else smooth_pts[nearest_contour_index(smooth_pts, endpoint)]
    smooth_offset_fraction = float(np.linalg.norm(pts[idx] - smooth_ref) / max(axis_length, 1e-6))

    radial = np.linalg.norm(pts - centroid, axis=1)
    radial_endpoint = float(np.linalg.norm(endpoint - centroid))
    max_radial = max(float(np.max(radial)), 1e-6)
    radial_ratio = float(radial_endpoint / max_radial)
    neighbor_window = max(3, int(config.get('bottom_roughness_neighbor_window', 14)))
    skip_center = max(1, int(config.get('bottom_roughness_skip_center_points', 2)))
    neighbors = [
        float(radial[(idx + offset) % len(radial)])
        for offset in range(-neighbor_window, neighbor_window + 1)
        if abs(offset) > skip_center
    ]
    radial_spike_fraction = 0.0
    if neighbors:
        radial_spike_fraction = max(0.0, radial_endpoint - float(np.median(neighbors))) / max(axis_length, 1e-6)

    support_fraction = float(config.get('candidate_endpoint_support_fraction', 0.08))
    mid_fraction = float(config.get('candidate_endpoint_mid_fraction', 0.50))
    max_dist = max(axis_length, max_radial * 2.0) + 10.0
    support_pos = endpoint + support_fraction * axis_length * axis_dir
    mid_pos = endpoint + mid_fraction * axis_length * axis_dir
    support_width = (
        ray_distance_to_mask(mask_binary, support_pos, -perp_dir, max_dist)
        + ray_distance_to_mask(mask_binary, support_pos, perp_dir, max_dist)
    )
    mid_width = (
        ray_distance_to_mask(mask_binary, mid_pos, -perp_dir, max_dist)
        + ray_distance_to_mask(mask_binary, mid_pos, perp_dir, max_dist)
    )
    support_ratio = float(support_width / max(mid_width, 1e-6)) if mid_width > 0 else 1.0

    smooth_threshold = max(1e-6, float(config.get('bottom_roughness_smooth_offset_fraction', 0.035)))
    spike_threshold = max(1e-6, float(config.get('bottom_roughness_radial_spike_fraction', 0.030)))
    min_support_ratio = max(1e-6, float(config.get('candidate_endpoint_min_support_ratio', 0.10)))
    smooth_score = max(0.0, smooth_offset_fraction / smooth_threshold - 1.0)
    spike_score = max(0.0, radial_spike_fraction / spike_threshold - 1.0)
    neck_score = max(0.0, (min_support_ratio - support_ratio) / min_support_ratio)
    roughness_score = float(np.clip(0.45 * smooth_score + 0.35 * spike_score + 0.20 * neck_score, 0.0, 1.0))

    return {
        'roughness_score': roughness_score,
        'sharpness': float(sharpness),
        'radial_ratio': radial_ratio,
        'support_ratio': support_ratio,
        'smooth_offset_fraction': float(smooth_offset_fraction),
        'radial_spike_fraction': float(radial_spike_fraction),
        'support_width_px': float(support_width),
        'mid_width_px': float(mid_width),
    }




def compute_endpoint_penalty(contour, mask_binary, bottom, top, centroid, sharpness_profiles, config):
    bottom_metrics = endpoint_local_shape_metrics(
        contour, mask_binary, bottom, top, centroid, sharpness_profiles, config
    )
    top_metrics = endpoint_local_shape_metrics(
        contour, mask_binary, top, bottom, centroid, sharpness_profiles, config
    )
    target_radial = float(config.get('candidate_endpoint_target_radial_ratio', 0.82))
    radial_penalty = 0.5 * (
        max(0.0, target_radial - bottom_metrics['radial_ratio']) / max(target_radial, 1e-6)
        + max(0.0, target_radial - top_metrics['radial_ratio']) / max(target_radial, 1e-6)
    )
    min_sharpness = float(config.get('candidate_endpoint_min_sharpness', 0.015))
    best_sharpness = max(bottom_metrics['sharpness'], top_metrics['sharpness'])
    flatness_penalty = max(0.0, min_sharpness - best_sharpness) / max(min_sharpness, 1e-6)
    roughness_penalty = 0.5 * (bottom_metrics['roughness_score'] + top_metrics['roughness_score'])
    endpoint_penalty = float(np.clip(
        0.45 * roughness_penalty + 0.35 * radial_penalty + 0.20 * flatness_penalty,
        0.0,
        1.0,
    ))
    return {
        'endpoint_penalty': endpoint_penalty,
        'endpoint_radial_penalty': float(np.clip(radial_penalty, 0.0, 1.0)),
        'endpoint_flatness_penalty': float(np.clip(flatness_penalty, 0.0, 1.0)),
        'bottom_endpoint_roughness': bottom_metrics['roughness_score'],
        'top_endpoint_roughness': top_metrics['roughness_score'],
        'bottom_endpoint_radial_ratio': bottom_metrics['radial_ratio'],
        'top_endpoint_radial_ratio': top_metrics['radial_ratio'],
        'bottom_endpoint_support_ratio': bottom_metrics['support_ratio'],
        'top_endpoint_support_ratio': top_metrics['support_ratio'],
    }




def scan_centroid_axis_summaries(contour, centroid, config):
    """Scan undirected centroid chords and keep their lengths."""
    step = max(0.5, float(config.get(
        'candidate_axis_global_scan_step_deg',
        config.get('centroid_chord_angle_step_deg', 1.0),
    )))
    summaries = []
    for angle in np.arange(0.0, 360.0, step):
        endpoint_info = centroid_axis_endpoints(contour, centroid, angle)
        if endpoint_info is None:
            continue
        point_a, point_b, direction, length = endpoint_info
        if length < float(config.get('candidate_axis_min_abs_length_px', 3.0)):
            continue
        summaries.append({
            'angle_deg': normalize_axis_angle_deg(angle),
            'length_px': float(length),
            'point_a': point_a,
            'point_b': point_b,
        })
    return summaries




def get_enabled_axis_methods(config):
    default_methods = [
        'moments_axis',
        'longest_centroid_chord',
        'shortest_centroid_chord',
    ]
    supported_methods = set(default_methods)
    methods = config.get('candidate_axis_methods') or default_methods
    aliases = {
        'moments_pca': 'moments_axis',
    }
    enabled = []
    for method in methods:
        method_name = aliases.get(str(method).strip(), str(method).strip())
        if method_name in supported_methods and method_name not in enabled:
            enabled.append(method_name)
    return enabled or default_methods




def collect_base_axis_angles(contour, mask_binary, centroid, config):
    base_axes = []
    enabled_methods = set(get_enabled_axis_methods(config))

    def add(source, angle, detail=None):
        if angle is None:
            return
        if str(source) not in enabled_methods:
            return
        base_axes.append({
            'source': str(source),
            'angle_deg': normalize_axis_angle_deg(angle),
            'detail': detail or {},
        })

    moment_axis = calculate_principal_axis_from_moments(mask_binary=mask_binary, contour=contour)
    if moment_axis.get('valid', False):
        add('moments_axis', moment_axis.get('axis_angle_deg'), {
            'minor_major_ratio': moment_axis.get('minor_major_ratio'),
            'source': moment_axis.get('source'),
        })

    needs_centroid_scan = bool(
        enabled_methods.intersection({
            'longest_centroid_chord',
            'shortest_centroid_chord',
        })
    )
    summaries = scan_centroid_axis_summaries(contour, centroid, config) if needs_centroid_scan else []
    if summaries:
        longest = max(summaries, key=lambda item: item['length_px'])
        shortest = min(summaries, key=lambda item: item['length_px'])
        add('longest_centroid_chord', longest['angle_deg'], {'length_px': longest['length_px']})
        add('shortest_centroid_chord', shortest['angle_deg'], {'length_px': shortest['length_px']})
    return base_axes




def score_axis_candidates(contour, mask_binary, centroid, candidates, sharpness_profiles,
                          config, image_bgr=None):
    if not candidates:
        return []

    color_context = build_color_symmetry_context(image_bgr, mask_binary, config)
    lengths = np.array([float(c.get('length_px', 0.0)) for c in candidates], dtype=float)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    if lengths.size:
        ref_length = float(np.percentile(lengths, float(config.get('candidate_length_reference_percentile', 90.0))))
    else:
        ref_length = 1.0
    min_length = max(
        float(config.get('candidate_axis_min_abs_length_px', 3.0)),
        ref_length * float(config.get('candidate_axis_min_length_fraction', 0.72)),
    )

    weights = {
        'area': float(config.get('axis_score_area_weight', 0.25)),
        'width': float(config.get('axis_score_width_weight', 0.30)),
        'color': float(config.get('axis_score_color_weight', 0.25)),
        'length': float(config.get('axis_score_length_weight', 0.10)),
        'endpoint': float(config.get('axis_score_endpoint_weight', 0.10)),
    }
    weight_sum = max(sum(weights.values()), 1e-6)

    scored = []
    for candidate_id, candidate in enumerate(candidates, start=1):
        bottom = np.asarray(candidate['bottom'], dtype=float)
        top = np.asarray(candidate['top'], dtype=float)
        _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
        if axis_length < 1.0:
            continue

        area_info = compute_mask_area_balance(mask_binary, centroid, perp_dir)
        width_info = compute_width_symmetry_penalty(mask_binary, bottom, top, config)
        color_info = compute_color_symmetry_penalty(color_context, mask_binary, bottom, top, config)
        endpoint_info = compute_endpoint_penalty(
            contour,
            mask_binary,
            bottom,
            top,
            centroid,
            sharpness_profiles,
            config,
        )
        if axis_length >= ref_length:
            length_penalty = 0.0
        elif axis_length >= min_length:
            length_penalty = float(np.clip(
                (ref_length - axis_length) / max(ref_length - min_length, 1e-6),
                0.0,
                1.0,
            ))
        else:
            length_penalty = 1.0

        total_score = (
            weights['area'] * area_info['area_balance_penalty']
            + weights['width'] * width_info['width_symmetry_penalty']
            + weights['color'] * color_info['color_symmetry_penalty']
            + weights['length'] * length_penalty
            + weights['endpoint'] * endpoint_info['endpoint_penalty']
        ) / weight_sum

        row = {**candidate}
        row.update(area_info)
        row.update(width_info)
        row.update(color_info)
        row.update(endpoint_info)
        row.update({
            'candidate_id': int(candidate_id),
            'length_px': float(axis_length),
            'axis_dir': axis_dir,
            'perp_dir': perp_dir,
            'axis_angle_deg': angle_from_direction(axis_dir),
            'length_reference_px': float(ref_length),
            'length_min_expected_px': float(min_length),
            'length_penalty': float(length_penalty),
            'total_score': float(np.clip(total_score, 0.0, 1.0)),
            'score': float(np.clip(total_score, 0.0, 1.0)),
            'axis_score_area_weight': weights['area'],
            'axis_score_width_weight': weights['width'],
            'axis_score_color_weight': weights['color'],
            'axis_score_length_weight': weights['length'],
            'axis_score_endpoint_weight': weights['endpoint'],
        })
        scored.append(row)

    scored.sort(key=lambda item: (float(item.get('total_score', 1.0)), -float(item.get('length_px', 0.0))))
    for rank, item in enumerate(scored, start=1):
        item['rank'] = int(rank)
        item['selected'] = bool(rank == 1)
        item['score_margin_to_second'] = None
    if len(scored) >= 2:
        margin = float(scored[1]['total_score'] - scored[0]['total_score'])
        scored[0]['score_margin_to_second'] = margin
    return scored




def generate_scored_axis_candidates(contour, mask_binary, centroid, sharpness_profiles,
                                    config, image_bgr=None):
    base_axes = collect_base_axis_angles(contour, mask_binary, centroid, config)
    if not base_axes:
        return [], [], []

    method_candidates = []
    for base_rank, base in enumerate(base_axes, start=1):
        base_angle = normalize_axis_angle_deg(base['angle_deg'])
        candidate = build_axis_candidate_from_angle(
            contour,
            centroid,
            base_angle,
            sharpness_profiles,
            config,
            base_source=base['source'],
            base_angle_deg=base_angle,
            scan_offset_deg=0.0,
            base_rank=base_rank,
        )
        candidate = maybe_refine_candidate_axis_endpoint(
            contour,
            centroid,
            candidate,
            sharpness_profiles,
            config,
        )
        if candidate is not None:
            candidate['method_representative'] = True
            method_candidates.append(candidate)

    method_candidates = merge_near_duplicate_method_axes(
        method_candidates,
        angle_threshold_deg=float(config.get('candidate_axis_duplicate_angle_deg', 10.0)),
    )

    scored_method_candidates = score_axis_candidates(
        contour,
        mask_binary,
        centroid,
        method_candidates,
        sharpness_profiles,
        config,
        image_bgr=image_bgr,
    )
    for candidate in scored_method_candidates:
        candidate['method_representative'] = True
        candidate['representative_rank'] = int(candidate.get('rank', 0) or 0)

    return scored_method_candidates, base_axes, scored_method_candidates




def merge_near_duplicate_method_axes(candidates, angle_threshold_deg=10.0):
    """If method axes nearly overlap, keep the longest corrected axis."""
    if not candidates:
        return []

    threshold = max(0.0, float(angle_threshold_deg))
    ordered = sorted(candidates, key=lambda item: float(item.get('length_px', 0.0)), reverse=True)
    kept = []

    for candidate in ordered:
        source = str(candidate.get('base_source', ''))
        merged = False
        for kept_candidate in kept:
            angle_gap = axis_angle_distance_deg(
                candidate.get('axis_angle_deg', 0.0),
                kept_candidate.get('axis_angle_deg', 0.0),
            )
            if angle_gap < threshold:
                sources = list(kept_candidate.get('merged_method_sources', []))
                if not sources:
                    sources = [str(kept_candidate.get('base_source', ''))]
                if source and source not in sources:
                    sources.append(source)
                kept_candidate['merged_method_sources'] = sources
                kept_candidate['merged_axis_count'] = int(len(sources))
                merged = True
                break
        if not merged:
            candidate['merged_method_sources'] = [source] if source else []
            candidate['merged_axis_count'] = 1
            kept.append(candidate)

    kept.sort(key=lambda item: int(item.get('base_rank', 9999) or 9999))
    return kept




def select_best_scored_axis(candidates, config):
    if not candidates:
        return None, True, 'no_candidates'
    candidates = sorted(candidates, key=lambda item: (float(item.get('total_score', 1.0)), item.get('rank', 9999)))
    best = candidates[0]
    reasons = []
    if len(candidates) >= 2:
        margin = float(candidates[1].get('total_score', 1.0) - best.get('total_score', 1.0))
        best['score_margin_to_second'] = margin
        if margin <= float(config.get('low_confidence_score_margin', 0.035)):
            reasons.append(f'top2_margin<={float(config.get("low_confidence_score_margin", 0.035)):.3f}')

    high_component = float(config.get('low_confidence_high_component_penalty', 0.70))
    for key in ['area_balance_penalty', 'width_symmetry_penalty', 'color_symmetry_penalty', 'endpoint_penalty']:
        value = best.get(key)
        if value is not None and float(value) >= high_component:
            reasons.append(f'{key}>={high_component:.2f}')

    conflict_gap = float(config.get('low_confidence_component_conflict_gap', 0.12))
    conflict_angle = float(config.get('low_confidence_component_conflict_angle_deg', 25.0))
    for key in ['area_balance_penalty', 'width_symmetry_penalty', 'color_symmetry_penalty']:
        valid = [c for c in candidates if c.get(key) is not None]
        if not valid:
            continue
        component_best = min(valid, key=lambda item: float(item.get(key, 1.0)))
        best_value = float(best.get(key, 1.0))
        component_value = float(component_best.get(key, 1.0))
        angle_gap = axis_angle_distance_deg(best.get('axis_angle_deg', 0.0), component_best.get('axis_angle_deg', 0.0))
        if best_value - component_value >= conflict_gap and angle_gap >= conflict_angle:
            reasons.append(f'{key}_winner_conflicts')

    return best, bool(reasons), '|'.join(reasons)




def measure_kernel_geometry(contour, mask_binary, config, num_width_samples,
                            kernel_name=None, subimage_path=None,
                            axis_predictor=None):
    """Shared per-kernel geometry used by both Stage 4 and Stage 5."""
    image_bgr = cv2.imread(subimage_path) if subimage_path and os.path.exists(subimage_path) else None

    chosen_bottom, axis_info = refine_bottom_point(
        contour,
        config,
        mask_binary=mask_binary,
        image_bgr=image_bgr,
        axis_predictor=axis_predictor,
    )
    centroid = coerce_point(axis_info.get('centroid'), fallback=calculate_centroid(contour))
    chosen_bottom = np.asarray(chosen_bottom, dtype=float)
    top = np.asarray(axis_info['top'], dtype=float)
    axis_angle_deg = axis_info.get('axis_angle_deg')
    if axis_angle_deg is None:
        axis_vec = top - chosen_bottom
        axis_angle_deg = (math.degrees(math.atan2(axis_vec[1], axis_vec[0])) + 360.0) % 180.0

    bottom_info = {
        **axis_info,
        'bottom': np.asarray(chosen_bottom, dtype=float),
        'top': top,
        'centroid': centroid,
        'centroid_source': axis_info.get('centroid_source', ''),
        'axis_angle_deg': float(axis_angle_deg),
        'bottom_source': axis_info.get('method', 'moments_axis'),
        'axis_bottom': np.asarray(chosen_bottom, dtype=float),
        'axis_top': np.asarray(top, dtype=float),
        'axis_method': axis_info.get('method', ''),
        'axis_score': axis_info.get('score'),
        'axis_bottom_candidate_score': axis_info.get('bottom_candidate_score'),
        'axis_top_candidate_score': axis_info.get('top_candidate_score'),
        'axis_model_pred_theta_deg': axis_info.get('axis_model_pred_theta_deg'),
        'axis_model_pred_theta_rad': axis_info.get('axis_model_pred_theta_rad'),
        'axis_model_pred_cos_theta': axis_info.get('axis_model_pred_cos_theta'),
        'axis_model_pred_sin_theta': axis_info.get('axis_model_pred_sin_theta'),
        'axis_model_pred_vector_norm': axis_info.get('axis_model_pred_vector_norm'),
        'axis_model_imgsz': axis_info.get('axis_model_imgsz'),
        'axis_model_preprocess': axis_info.get('axis_model_preprocess'),
    }
    measurement_bundle = compute_kernel_measurement_bundle(
        contour,
        chosen_bottom,
        top,
        bottom_info=bottom_info,
        num_width_samples=num_width_samples,
        mask_binary=mask_binary,
        config=config,
    )
    shape_info = classify_kernel_shape(
        contour,
        bottom_info,
        config,
        measurement_bundle=measurement_bundle,
        mask_binary=mask_binary,
        image_bgr=image_bgr,
    )
    bottom_info.update(shape_info)
    return chosen_bottom, bottom_info, measurement_bundle



