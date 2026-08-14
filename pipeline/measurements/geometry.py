"""geometry.py - pure contour/mask geometry and measurement primitives.

Extracted from kernel_metrics.py to keep that module focused on orchestration.
These functions depend only on numpy / cv2 / shapely / stdlib and each other,
NOT on the axis candidate/scoring logic, so they are independently unit-testable.
"""

import math
import os

import cv2
import numpy as np

from processing.contour_extraction import extract_largest_contour


def nearest_contour_index(contour, point):
    """Index of the contour point nearest to the given point."""
    distances = np.linalg.norm(contour - point, axis=1)
    return int(np.argmin(distances))




def compute_local_sharpness(contour, window=5):
    """
    Estimate how pointy each contour point is.

    Value range:
      0.0 -> smooth / round local boundary
      1.0 -> very sharp local boundary
    """
    n = len(contour)
    sharpness = np.zeros(n, dtype=float)
    window = max(1, min(window, max(1, n // 8)))

    for i in range(n):
        prev_pt = contour[(i - window) % n]
        curr_pt = contour[i]
        next_pt = contour[(i + window) % n]

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cosang = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.acos(cosang)
        sharpness[i] = 1.0 - (angle / math.pi)

    return sharpness



def calculate_centroid(contour):
    """Geometric centroid of the contour (mean of all points)."""
    return np.mean(contour, axis=0)




def point_is_valid(point):
    """Return True when a value can be interpreted as a finite 2D point."""
    try:
        arr = np.asarray(point, dtype=float).reshape(-1)
    except Exception:
        return False
    return arr.size == 2 and bool(np.all(np.isfinite(arr)))




def coerce_point(point, fallback=None):
    """Convert a value to a float xy point, or fall back when unavailable."""
    if point_is_valid(point):
        return np.asarray(point, dtype=float).reshape(2)
    if fallback is not None and point_is_valid(fallback):
        return np.asarray(fallback, dtype=float).reshape(2)
    return np.array([0.0, 0.0], dtype=float)




def load_binary_mask(mask_dir, kernel_name):
    """Load one cropped SAM binary mask and ensure it is 0/255 uint8."""
    mask_path = os.path.join(mask_dir, kernel_name)
    if not os.path.exists(mask_path):
        return None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask




def clean_kernel_mask(mask_binary, min_area=500, config=None):
    """
    Basic single-kernel mask cleanup: fill holes, remove tiny components,
    apply light morphology smoothing, and keep the largest valid body.
    """
    if mask_binary is None:
        return None, None

    cfg = config or {}
    _, binary = cv2.threshold(mask_binary.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)

    close_kernel = max(1, int(cfg.get('mask_cleanup_close_kernel', 5)))
    open_kernel = max(1, int(cfg.get('mask_cleanup_open_kernel', 3)))
    smooth_kernel = max(1, int(cfg.get('mask_cleanup_smooth_kernel', 3)))
    noise_area = max(1, int(cfg.get('mask_cleanup_min_noise_area', max(8, int(min_area * 0.05)))))

    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    flood = binary.copy()
    h, w = flood.shape[:2]
    # Hole-fill: flood from a background corner. Probe all four corners in case
    # the kernel touches (0,0); if every corner is foreground, skip hole-fill.
    seed = None
    for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if flood[corner[0], corner[1]] == 0:
            seed = corner
            break
    if seed is not None:
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, seed, 255)
        holes = cv2.bitwise_not(flood)
        binary = cv2.bitwise_or(binary, holes)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary > 0).astype(np.uint8),
        connectivity=8,
    )
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= noise_area:
            cleaned[labels == label] = 255
    if np.count_nonzero(cleaned) == 0:
        cleaned = binary

    if smooth_kernel > 1:
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        blurred = cv2.GaussianBlur(cleaned, (smooth_kernel, smooth_kernel), 0)
        _, cleaned = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contour = extract_largest_contour(cleaned, min_area=min_area)
    if contour is None:
        return cleaned, None
    cleaned = contour_to_filled_mask(contour, cleaned.shape)
    return cleaned, np.asarray(contour, dtype=float)




def contour_to_filled_mask(contour, mask_shape):
    """Rasterize one contour into a clean single-component binary mask."""
    mask = np.zeros(mask_shape, dtype=np.uint8)
    pts = np.round(np.asarray(contour, dtype=float)).astype(np.int32).reshape((-1, 1, 2))
    cv2.drawContours(mask, [pts], contourIdx=-1, color=255, thickness=-1)
    return mask




def load_kernels_from_mask_dir(mask_dir, min_area=500, config=None):
    """
    Build the active kernel list directly from SAM binary masks.

    Returns:
      [
        {
          'image_name': 'IMG_xxx_kernel_001.jpg',
          'mask_binary': uint8 mask,
          'contour': Nx2 contour array,
        },
        ...
      ]
    """
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(mask_dir)

    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    kernels = []
    skipped = 0

    for image_name in mask_files:
        mask_binary = load_binary_mask(mask_dir, image_name)
        if mask_binary is None:
            print(f'  WARNING: Could not read binary mask for {image_name}')
            skipped += 1
            continue

        mask_binary, contour = clean_kernel_mask(
            mask_binary,
            min_area=min_area,
            config=config,
        )
        if contour is None:
            print(f'  WARNING: No valid contour found in {image_name} — skipping')
            skipped += 1
            continue

        kernels.append({
            'image_name': image_name,
            'mask_binary': mask_binary,
            'contour': np.asarray(contour, dtype=float),
        })

    return kernels, skipped




def calculate_mask_centroid(mask_binary, contour=None):
    """
    Centroid from the SAM single-kernel mask using cv2.moments.

    Falls back to contour moments if the mask is missing or degenerate.
    """
    if mask_binary is not None:
        moments = cv2.moments(mask_binary, binaryImage=True)
        if abs(moments.get('m00', 0.0)) > 1e-6:
            return np.array([
                moments['m10'] / moments['m00'],
                moments['m01'] / moments['m00'],
            ], dtype=float), 'mask_moments'

    pts = get_open_contour(contour) if contour is not None else None
    if pts is not None and len(pts) >= 3:
        contour_for_cv = np.round(pts).astype(np.float32).reshape((-1, 1, 2))
        moments = cv2.moments(contour_for_cv)
        if abs(moments.get('m00', 0.0)) > 1e-6:
            return np.array([
                moments['m10'] / moments['m00'],
                moments['m01'] / moments['m00'],
            ], dtype=float), 'contour_moments'

    if pts is not None and len(pts) > 0:
        return calculate_centroid(pts), 'contour_mean'
    return np.array([0.0, 0.0], dtype=float), 'origin_fallback'




def calculate_principal_axis_from_moments(mask_binary=None, contour=None):
    """
    Estimate the major-axis direction from second-order central moments.

    This uses the filled mask moments when available.
    """
    pts = get_open_contour(contour) if contour is not None else None
    moments = None
    source = ''

    if mask_binary is not None:
        candidate = cv2.moments(mask_binary, binaryImage=True)
        if abs(candidate.get('m00', 0.0)) > 1e-6:
            moments = candidate
            source = 'mask_moments'

    if moments is None and pts is not None and len(pts) >= 3:
        contour_for_cv = np.round(pts).astype(np.float32).reshape((-1, 1, 2))
        candidate = cv2.moments(contour_for_cv)
        if abs(candidate.get('m00', 0.0)) > 1e-6:
            moments = candidate
            source = 'contour_moments'

    if moments is None:
        return {
            'valid': False,
            'major_axis': np.array([0.0, -1.0], dtype=float),
            'axis_angle_deg': 90.0,
            'minor_major_ratio': 1.0,
            'source': 'moments_unavailable',
        }

    m00 = float(moments['m00'])
    cov = np.array([
        [float(moments['mu20']) / m00, float(moments['mu11']) / m00],
        [float(moments['mu11']) / m00, float(moments['mu02']) / m00],
    ], dtype=float)
    if not np.all(np.isfinite(cov)):
        return {
            'valid': False,
            'major_axis': np.array([0.0, -1.0], dtype=float),
            'axis_angle_deg': 90.0,
            'minor_major_ratio': 1.0,
            'source': f'{source}_invalid_covariance',
        }

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_val = max(float(eigvals[order[0]]), 0.0)
    minor_val = max(float(eigvals[order[-1]]), 0.0)
    if major_val < 1e-12:
        return {
            'valid': False,
            'major_axis': np.array([0.0, -1.0], dtype=float),
            'axis_angle_deg': 90.0,
            'minor_major_ratio': 1.0,
            'source': f'{source}_degenerate',
        }

    major_axis = np.asarray(eigvecs[:, order[0]], dtype=float)
    major_axis = major_axis / max(np.linalg.norm(major_axis), 1e-6)
    axis_angle_deg = (math.degrees(math.atan2(major_axis[1], major_axis[0])) + 360.0) % 180.0
    minor_major_ratio = float(math.sqrt(minor_val / max(major_val, 1e-12)))

    return {
        'valid': True,
        'major_axis': major_axis,
        'axis_angle_deg': float(axis_angle_deg),
        'minor_major_ratio': minor_major_ratio,
        'source': source,
    }




def build_multiscale_sharpness_profiles(contour, base_window=5, scale_factors=None):
    """Precompute local sharpness at several contour scales for tip scoring."""
    pts = get_open_contour(contour)
    if pts is None or len(pts) == 0:
        return []

    if scale_factors is None:
        scale_factors = [1.0, 2.0, 3.5]

    max_window = max(1, len(pts) // 6)
    profiles = []
    used_windows = set()

    for factor in scale_factors:
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            continue
        window = int(round(max(1.0, float(base_window) * max(factor, 0.5))))
        window = max(1, min(window, max_window))
        if window in used_windows:
            continue
        used_windows.add(window)
        profiles.append((window, compute_local_sharpness(pts, window=window)))

    if not profiles:
        fallback_window = max(1, min(int(base_window), max_window))
        profiles.append((fallback_window, compute_local_sharpness(pts, window=fallback_window)))

    return profiles




def deduplicate_intersections(intersections, origin, direction, tol=1.0):
    """Merge near-duplicate line/contour intersections caused by shared vertices."""
    if not intersections:
        return []

    ordered = sorted(
        [(float(np.dot(np.asarray(pt, dtype=float) - origin, direction)), np.asarray(pt, dtype=float))
         for pt in intersections],
        key=lambda item: item[0],
    )

    deduped = [ordered[0][1]]
    last_proj = ordered[0][0]

    for proj, pt in ordered[1:]:
        if abs(proj - last_proj) <= tol:
            deduped[-1] = 0.5 * (deduped[-1] + pt)
            last_proj = 0.5 * (last_proj + proj)
        else:
            deduped.append(pt)
            last_proj = proj

    return deduped




def sharpness_around_point(contour, sharpness_values, point, neighbor_radius=2,
                           multiscale_profiles=None):
    """Estimate endpoint sharpness around the nearest contour index."""
    if len(contour) == 0 or len(sharpness_values) == 0:
        return 0.0, 0

    idx = nearest_contour_index(contour, point)

    def score_profile(values, radius):
        samples = np.array([
            float(values[(idx + offset) % len(values)])
            for offset in range(-radius, radius + 1)
        ], dtype=float)
        if samples.size == 0:
            return 0.0
        top_k = min(samples.size, max(2, radius + 1))
        strongest = np.sort(samples)[-top_k:]
        return float(0.65 * np.max(samples) + 0.35 * np.mean(strongest))

    radius = max(1, int(neighbor_radius))
    if not multiscale_profiles:
        return score_profile(sharpness_values, radius), int(idx)

    scores = []
    weights = []
    for level, (_, profile) in enumerate(multiscale_profiles):
        local_radius = max(radius, int(round(radius * (1.0 + 0.6 * level))))
        scores.append(score_profile(profile, local_radius))
        weights.append(1.0 / (1.0 + 0.5 * level))

    return float(np.average(scores, weights=weights)), int(idx)




def compute_major_axis_metrics(contour, centroid=None):
    """
    Principal-axis summary of the contour shape.

    Returns:
      - major_axis: unit vector of the first PCA axis
      - minor_major_ratio: sqrt(minor_variance / major_variance)
        1.0 -> round / isotropic
        smaller -> more elongated
    """
    pts = np.asarray(contour, dtype=float)
    if centroid is None:
        centroid = calculate_centroid(pts)
    centroid = np.asarray(centroid, dtype=float)

    if len(pts) < 3:
        return {
            'major_axis': np.array([0.0, -1.0]),
            'minor_major_ratio': 1.0,
        }

    centered = pts - centroid
    cov = np.cov(centered.T)
    if np.ndim(cov) != 2 or cov.shape != (2, 2):
        return {
            'major_axis': np.array([0.0, -1.0]),
            'minor_major_ratio': 1.0,
        }

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major_axis = np.asarray(eigvecs[:, order[0]], dtype=float)
    major_axis = major_axis / max(np.linalg.norm(major_axis), 1e-6)

    major_val = max(float(eigvals[order[0]]), 1e-12)
    minor_val = max(float(eigvals[order[-1]]), 0.0)
    minor_major_ratio = float(math.sqrt(minor_val / major_val))

    return {
        'major_axis': major_axis,
        'minor_major_ratio': minor_major_ratio,
    }




def compute_eccentricity(minor_major_ratio):
    """Convert a minor/major axis ratio into ellipse eccentricity."""
    try:
        ratio = float(minor_major_ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    ratio = min(1.0, max(0.0, ratio))
    return float(math.sqrt(max(0.0, 1.0 - ratio ** 2)))




def compute_kernel_measurement_bundle(contour, bottom, top, bottom_info=None,
                                      num_width_samples=100, mask_binary=None, config=None):
    """
    Compute the geometry bundle shared by CSV export, round filtering, and overlays.
    """
    bottom = np.asarray(bottom, dtype=float)
    top = np.asarray(top, dtype=float)
    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length >= 1.0:
        # Convention: crown = position 0, pedicel = position 1 (crown→pedicel).
        # bottom=pedicel, top=crown, so measure W25/W50/W75 from the crown.
        w25, w25_a, w25_b = width_at_fraction(
            contour, top, -axis_dir, perp_dir, 0.25, axis_length
        )
        w50, w50_a, w50_b = width_at_fraction(
            contour, top, -axis_dir, perp_dir, 0.50, axis_length
        )
        w75, w75_a, w75_b = width_at_fraction(
            contour, top, -axis_dir, perp_dir, 0.75, axis_length
        )
        max_width, mb_a, mb_b = max_width_from_contour(
            contour,
            bottom,
            axis_dir,
            perp_dir,
            axis_length,
            num_width_samples,
        )
    else:
        w25, w25_a, w25_b = 0.0, None, None
        w50, w50_a, w50_b = 0.0, None, None
        w75, w75_a, w75_b = 0.0, None, None
        max_width, mb_a, mb_b = 0.0, None, None

    area, perimeter = compute_area_perimeter(contour)
    circularity = compute_circularity(area, perimeter)
    moments_metrics = calculate_principal_axis_from_moments(mask_binary=mask_binary, contour=contour)
    if moments_metrics.get('valid', False):
        minor_major_ratio = float(moments_metrics.get('minor_major_ratio', 1.0))
    else:
        centroid = None if bottom_info is None else bottom_info.get('centroid')
        pca_metrics = compute_major_axis_metrics(
            contour,
            centroid=centroid if point_is_valid(centroid) else None,
        )
        minor_major_ratio = float(pca_metrics.get('minor_major_ratio', 1.0))
    eccentricity = compute_eccentricity(minor_major_ratio)
    length_width_ratio = None
    if axis_length >= 1.0 and w50 > 1e-6:
        length_width_ratio = float(axis_length / w50)

    return {
        'axis_length': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'width_25pct': float(w25),
        'width_25pct_pts': (w25_a, w25_b),
        'width_50pct': float(w50),
        'width_50pct_pts': (w50_a, w50_b),
        'width_75pct': float(w75),
        'width_75pct_pts': (w75_a, w75_b),
        'max_width': float(max_width),
        'max_width_pts': (mb_a, mb_b),
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'minor_major_ratio': minor_major_ratio,
        'eccentricity': float(eccentricity),
        'length_width_ratio': length_width_ratio,
    }




def get_open_contour(contour):
    """Return a non-duplicated contour array for circular indexing utilities."""
    pts = np.asarray(contour, dtype=float)
    if len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6:
        return pts[:-1]
    return pts




def resample_closed_contour(contour, n_points=360):
    """Resample a closed contour to equal-arc outline points."""
    pts = get_open_contour(contour)
    if pts is None or len(pts) < 3:
        return np.asarray(pts, dtype=float)

    try:
        n_points = int(n_points)
    except (TypeError, ValueError):
        n_points = 360
    n_points = max(24, int(n_points))

    closed = np.vstack([pts, pts[0]])
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(segments))
    if perimeter <= 1e-6:
        return pts.copy()

    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    sample_distances = np.linspace(0.0, perimeter, n_points, endpoint=False)
    resampled = np.zeros((n_points, 2), dtype=float)

    seg_idx = 0
    for out_idx, distance in enumerate(sample_distances):
        while seg_idx < len(segments) - 1 and cumulative[seg_idx + 1] <= distance:
            seg_idx += 1
        seg_len = max(float(segments[seg_idx]), 1e-6)
        t = float((distance - cumulative[seg_idx]) / seg_len)
        resampled[out_idx] = closed[seg_idx] + t * (closed[seg_idx + 1] - closed[seg_idx])

    return resampled




def smooth_closed_contour(contour, window=9, iterations=1):
    """Circular moving-average smoothing for equal-arc contours."""
    pts = get_open_contour(contour)
    if pts is None or len(pts) < 3:
        return np.asarray(pts, dtype=float)

    try:
        window = int(window)
    except (TypeError, ValueError):
        window = 9
    if window <= 1:
        return np.asarray(pts, dtype=float).copy()
    if window % 2 == 0:
        window += 1
    window = max(3, min(window, max(3, len(pts) // 3)))

    try:
        iterations = int(iterations)
    except (TypeError, ValueError):
        iterations = 1
    iterations = max(1, iterations)

    smoothed = np.asarray(pts, dtype=float).copy()
    radius = window // 2
    weights = np.ones(window, dtype=float) / float(window)
    for _ in range(iterations):
        updated = np.zeros_like(smoothed)
        for offset, weight in zip(range(-radius, radius + 1), weights):
            updated += weight * np.roll(smoothed, shift=offset, axis=0)
        smoothed = updated
    return smoothed




def get_bottom_refinement_hint_contour(contour, config):
    """Return the contour used for local bottom candidate hints."""
    pts = get_open_contour(contour)
    if not bool(config.get('bottom_refine_use_smoothed_contour', True)):
        return pts
    return smooth_closed_contour(
        pts,
        window=int(config.get('bottom_smooth_contour_window', 11)),
        iterations=int(config.get('bottom_smooth_contour_iterations', 2)),
    )




def sample_outline_count_points(contour, n_points=360):
    """
    Sample a closed outline into a fixed number of ordered count points.

    This is the first outline normalization step: preserve contour order and
    create exactly n points. The downstream analysis then resamples these count
    points by arc length so spacing becomes uniform.
    """
    pts = get_open_contour(contour)
    if pts is None or len(pts) < 3:
        return np.asarray(pts, dtype=float)

    try:
        n_points = int(n_points)
    except (TypeError, ValueError):
        n_points = 360
    n_points = max(24, int(n_points))

    count_positions = np.linspace(0.0, float(len(pts)), n_points, endpoint=False)
    sampled = np.zeros((n_points, 2), dtype=float)
    for out_idx, position in enumerate(count_positions):
        left_idx = int(math.floor(position)) % len(pts)
        right_idx = (left_idx + 1) % len(pts)
        t = float(position - math.floor(position))
        sampled[out_idx] = pts[left_idx] + t * (pts[right_idx] - pts[left_idx])

    return sampled




def get_analysis_contour(contour, config):
    """Return the equal-arc contour used by axis logic."""
    n_points = int(config.get('outline_resample_points', 360))
    count_points = sample_outline_count_points(contour, n_points=n_points)
    return resample_closed_contour(count_points, n_points=n_points)




def get_outline_count_points(contour, config):
    """Return the fixed-count outline points before equal-distance resampling."""
    n_points = int(config.get('outline_resample_points', 360))
    return sample_outline_count_points(contour, n_points=n_points)




def build_axis_frame(bottom, top):
    """Axis length and basis vectors for a bottom-to-top line."""
    axis_vec = top - bottom
    axis_length = np.linalg.norm(axis_vec)
    if axis_length > 0:
        axis_dir = axis_vec / axis_length
    else:
        axis_dir = np.array([0.0, -1.0])
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])
    return axis_vec, float(axis_length), axis_dir, perp_dir




def summarize_axis_widths(contour, bottom, top, config):
    """Support widths used for QC and top/bottom diagnostics."""
    _, axis_length, axis_dir, perp_dir = build_axis_frame(bottom, top)
    if axis_length < 1:
        return {
            'axis_length': float(axis_length),
            'axis_dir': axis_dir,
            'perp_dir': perp_dir,
            'near_width': 0.0,
            'mid_width': 0.0,
            'upper_width': 0.0,
        }

    near_fraction = float(config.get('bottom_near_fraction', 0.12))
    mid_fraction = float(config.get('bottom_mid_fraction', 0.50))
    upper_fraction = float(config.get('bottom_upper_fraction', 0.75))

    near_width, _, _ = width_at_fraction(
        contour, bottom, axis_dir, perp_dir, near_fraction, axis_length
    )
    mid_width, _, _ = width_at_fraction(
        contour, bottom, axis_dir, perp_dir, mid_fraction, axis_length
    )
    upper_width, _, _ = width_at_fraction(
        contour, bottom, axis_dir, perp_dir, upper_fraction, axis_length
    )

    return {
        'axis_length': float(axis_length),
        'axis_dir': axis_dir,
        'perp_dir': perp_dir,
        'near_width': float(near_width),
        'mid_width': float(mid_width),
        'upper_width': float(upper_width),
    }





def find_contour_intersections(contour, line):
    """
    Find where a Shapely Line crosses the kernel contour.

    Returns a list of (x, y) numpy arrays — the intersection points.
    We expect exactly 2 for any width measurement (left side and right side).
    """
    from shapely.geometry import LineString, Point, MultiPoint

    pts = get_open_contour(contour)
    if len(pts) < 2:
        return []
    closed_pts = np.vstack([pts, pts[0]])
    contour_line = LineString(closed_pts)
    intersection = contour_line.intersection(line)

    if intersection.is_empty:
        return []
    if isinstance(intersection, Point):
        return [np.array(intersection.coords[0])]
    if hasattr(intersection, 'geoms'):
        # Multiple intersection points (MultiPoint or GeometryCollection)
        points = []
        for geom in intersection.geoms:
            if hasattr(geom, 'coords'):
                points.append(np.array(geom.coords[0]))
        return points
    return [np.array(intersection.coords[0])]




def width_at_fraction(contour, bottom, axis_dir, perp_dir, fraction, axis_length):
    """
    Measure the kernel width at a given fraction along the main axis.

    Draws a line perpendicular to the axis at (bottom + fraction × axis),
    finds where it crosses the contour, and returns the distance between
    the two crossing points.

    Args:
        contour     : (N, 2) resampled contour
        bottom      : (x, y) bottom tip
        axis_dir    : unit vector along the main axis (bottom to top)
        perp_dir    : unit vector perpendicular to the axis
        fraction    : 0.0 = bottom end, 1.0 = top end
        axis_length : total length of the axis in pixels

    Returns:
        width   : distance between the two intersection points (pixels), or 0
        point_a : first intersection point (x, y) or None
        point_b : second intersection point (x, y) or None
    """
    pos = bottom + fraction * axis_length * axis_dir
    half_extent = max(float(np.linalg.norm(np.ptp(contour, axis=0))), 1.0) * 2.0 + 20.0
    from shapely.geometry import LineString
    line = LineString([
        pos - float(half_extent) * perp_dir,
        pos + float(half_extent) * perp_dir,
    ])
    intersections = find_contour_intersections(contour, line)
    intersections = deduplicate_intersections(intersections, pos, perp_dir)

    if len(intersections) < 2:
        return 0.0, None, None

    # Sort intersection points along the perpendicular direction
    projections = [np.dot(pt - pos, perp_dir) for pt in intersections]
    sorted_pts = [pt for _, pt in sorted(zip(projections, intersections))]

    width = np.linalg.norm(sorted_pts[-1] - sorted_pts[0])
    return float(width), sorted_pts[0], sorted_pts[-1]




def compute_width_profile(contour, bottom, top, axis_length, num_samples=100):
    """
    Compute the full width profile along the directed main axis.

    Samples ``num_samples`` evenly-spaced positions from bottom to top and
    measures the perpendicular width at each position.

    Args:
        contour     : (N, 2) resampled contour points
        bottom      : (x, y) bottom (pedicel) tip
        top         : (x, y) top (crown) tip
        axis_length : total axis length in pixels (or 1.0 for normalised)
        num_samples : number of sampling positions (default 100)

    Returns:
        fractions   : (num_samples,)  array of fraction values [0, 1]
        widths      : (num_samples,)  array of full-width values (px)
        half_widths : (num_samples,)  array of half-width values (px)
        left_pts    : list of (x, y)  left-side intersection points (or None)
        right_pts   : list of (x, y)  right-side intersection points (or None)
    """
    bottom = np.asarray(bottom, dtype=float)
    top = np.asarray(top, dtype=float)
    if axis_length < 1e-6:
        axis_length = 1.0
    axis_vec = top - bottom
    axis_dir = axis_vec / float(np.linalg.norm(axis_vec))
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])

    fractions = np.linspace(0.0, 1.0, num_samples)
    widths = np.zeros(num_samples, dtype=float)
    half_widths = np.zeros(num_samples, dtype=float)
    left_pts_list = []
    right_pts_list = []

    for i, frac in enumerate(fractions):
        w, pa, pb = width_at_fraction(
            contour, bottom, axis_dir, perp_dir, float(frac), float(axis_length),
        )
        widths[i] = w
        half_widths[i] = w / 2.0
        left_pts_list.append(pa)
        right_pts_list.append(pb)

    return fractions, widths, half_widths, left_pts_list, right_pts_list




def max_width_from_contour(contour, bottom, axis_dir, perp_dir, axis_length,
                              num_samples=100):
    """
    Find the maximum width across the entire kernel by scanning along the axis.

    Tries 'num_samples' evenly-spaced positions from bottom to top and
    returns the position with the widest perpendicular measurement.

    Returns:
        max_width : the maximum width found (pixels)
        point_a   : left endpoint of the max width line
        point_b   : right endpoint
    """
    max_width = 0.0
    best_pair = (None, None)

    for frac in np.linspace(0, 1, num_samples):
        w, pa, pb = width_at_fraction(contour, bottom, axis_dir, perp_dir,
                                       frac, axis_length)
        if w > max_width:
            max_width = w
            best_pair = (pa, pb)

    return max_width, best_pair[0], best_pair[1]




def compute_area_perimeter(contour):
    """
    Compute area and perimeter from the contour point array.

    Area uses the Shoelace formula (exact for polygons).
    Perimeter is the total arc length.
    """
    pts = get_open_contour(contour).astype(float)

    # Shoelace formula for polygon area
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i, 0] * pts[j, 1]
        area -= pts[j, 0] * pts[i, 1]
    area = abs(area) / 2.0

    # Perimeter = sum of segment lengths including the closing edge
    if len(pts) < 2:
        perimeter = 0.0
    else:
        closed_pts = np.vstack([pts, pts[0]])
        diffs = np.diff(closed_pts, axis=0)
        perimeter = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return area, perimeter




def compute_circularity(area, perimeter):
    """
    Circularity = 4π × Area / Perimeter²

    Value range:
      1.0 = perfect circle
      < 1.0 = elongated or irregular shape
    """
    if perimeter == 0:
        return 0.0
    return (4 * math.pi * area) / (perimeter ** 2)



