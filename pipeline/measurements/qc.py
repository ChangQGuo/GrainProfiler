"""qc.py - QC image rendering and overlay serialization (pure view layer).

Extracted from kernel_metrics.py.  Depends only on measurements.geometry,
utils.visualization and stdlib; it does not touch axis/orchestration logic.
"""

import json
import os

import cv2
import numpy as np

from measurements.geometry import (
    coerce_point,
    get_open_contour,
    get_outline_count_points,
    point_is_valid,
)
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


def save_outline_count_qc(subimage_path, contour, qc_path, config):
    """
    Save the initial fixed-count outline points before equal-distance resampling.
    """
    image = cv2.imread(subimage_path) if os.path.exists(subimage_path) else None
    contour = get_open_contour(contour)
    if image is None:
        h = int(contour[:, 1].max()) + 20
        w = int(contour[:, 0].max()) + 20
        image = np.zeros((h, w, 3), dtype=np.uint8)

    n_points = int(config.get('outline_resample_points', 360))
    count_points = get_outline_count_points(contour, config)

    result = draw_contour(image, contour, color=(0, 180, 0), thickness=1)
    for idx, point in enumerate(count_points):
        x, y = int(round(point[0])), int(round(point[1]))
        color = (0, 255, 255)
        radius = 1
        if idx == 0:
            color = (0, 0, 255)
            radius = 3
        cv2.circle(result, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    save_image(result, qc_path)




def save_measurement_qc(subimage_path, contour, bottom, top, qc_path,
                        refine_info=None, measurement_bundle=None):
    """
    Save one per-kernel measurement QC image with only final downstream geometry.
    """
    image = cv2.imread(subimage_path) if os.path.exists(subimage_path) else None
    if image is None:
        h = int(contour[:, 1].max()) + 20
        w = int(contour[:, 0].max()) + 20
        image = np.zeros((h, w, 3), dtype=np.uint8)

    centroid = coerce_point(
        refine_info.get('centroid') if refine_info is not None else None,
        fallback=0.5 * (np.asarray(bottom, dtype=float) + np.asarray(top, dtype=float)),
    )
    if measurement_bundle is not None:
        result = draw_contour(image, contour, color=(0, 180, 0), thickness=1)
        result = draw_all_measurements(
            result,
            bottom,
            top,
            measurement_bundle.get('width_25pct_pts', (None, None)),
            measurement_bundle.get('width_50pct_pts', (None, None)),
            measurement_bundle.get('width_75pct_pts', (None, None)),
            measurement_bundle.get('max_width_pts', (None, None)),
        )
        result = draw_centroid(result, centroid, radius=1, label='')
    else:
        result = draw_axis_points(image, contour, centroid, bottom, top)
    if refine_info is not None and refine_info.get('shape_label') == 'Round':
        orange = (0, 165, 255)
        cv2.rectangle(result, (1, 1), (result.shape[1] - 2, result.shape[0] - 2), orange, 2)
        result = draw_text_with_outline(
            result,
            'Round',
            (6, 18),
            color=orange,
            font_scale=0.42,
            thickness=1,
            outline_thickness=3,
        )
    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    save_image(result, qc_path)




def save_axis_candidate_image(subimage_path, contour, bottom_info, output_path):
    """Save a per-kernel final-axis review image with no score/candidate labels."""
    image = cv2.imread(subimage_path) if subimage_path and os.path.exists(subimage_path) else None
    contour = get_open_contour(contour)
    if image is None:
        h = int(contour[:, 1].max()) + 20
        w = int(contour[:, 0].max()) + 20
        image = np.zeros((h, w, 3), dtype=np.uint8)

    result = image.copy()

    bottom = bottom_info.get('bottom')
    top = bottom_info.get('top')
    centroid = bottom_info.get('centroid')
    if point_is_valid(bottom) and point_is_valid(top):
        result = draw_axis(result, bottom, top, color=(0, 0, 255), thickness=2)
        result = draw_bottom_point(result, bottom, radius=3, label='')
        result = draw_top_point(result, top, radius=3, label='')
    if point_is_valid(centroid):
        result = draw_centroid(result, centroid, radius=2, label='')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(result, output_path)


# ---------------------------------------------------------------------------
# Load combined data file


from utils.bbox import get_accepted_detections
get_accepted_detections_for_overlay = get_accepted_detections  # backward-compat alias
def load_detection_box_lookup(bbox_json_path):
    """Map {original_image_name: {kernel_id: [x1,y1,x2,y2]}}."""
    if not os.path.exists(bbox_json_path):
        return {}

    with open(bbox_json_path, 'r', encoding='utf-8') as f:
        bbox_data = json.load(f)

    lookup = {}
    for image_name, image_entry in bbox_data.items():
        accepted = get_accepted_detections_for_overlay(image_entry)
        if not accepted:
            continue
        lookup[image_name] = {
            int(item['kernel_id']): [int(v) for v in item['box']]
            for item in accepted
            if item.get('kernel_id') is not None and item.get('box') is not None
        }
    return lookup




def subimage_point_to_full_image(local_point, box, padding, image_shape):
    """Convert a point from subimage coordinates back to full-image coordinates."""
    x1, y1, x2, y2 = [int(v) for v in box]
    height, width = image_shape[:2]
    x1p = max(0, x1 - int(padding))
    y1p = max(0, y1 - int(padding))
    full_x = int(round(x1p + float(local_point[0])))
    full_y = int(round(y1p + float(local_point[1])))
    full_x = max(0, min(width - 1, full_x))
    full_y = max(0, min(height - 1, full_y))
    return full_x, full_y




def overlay_scale(image):
    """Compact but readable marker sizes for full-tray overlays."""
    min_dim = max(1, min(image.shape[:2]))
    point_radius = max(2, min(3, int(round(min_dim / 1000.0))))
    line_thickness = max(1, min(2, int(round(min_dim / 1400.0))))
    font_scale = max(0.34, min(0.46, min_dim / 3600.0))
    return point_radius, line_thickness, font_scale




def save_measurement_overlay_images(overlay_items_by_image, image_dir, overlay_dir, padding):
    """
    Save one full-tray overlay per original image.

    Per kernel:
      - main axis
      - centroid
      - bottom / top / centroid points
    """
    if not overlay_items_by_image:
        return 0

    os.makedirs(overlay_dir, exist_ok=True)
    saved_count = 0

    for image_name, items in overlay_items_by_image.items():
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f'  WARNING: Could not read original image for measurement overlay: {image_path}')
            continue
        point_radius, line_thickness, font_scale = overlay_scale(image)

        for item in items:
            box = item.get('box')
            if box is None:
                continue
            if item.get('shape_label') == 'Round' or bool(item.get('is_round', False)):
                x1, y1, x2, y2 = [int(v) for v in box]
                orange = (0, 165, 255)
                cv2.rectangle(image, (x1, y1), (x2, y2), orange, max(2, line_thickness + 1))
                image = draw_text_with_outline(
                    image,
                    'Round',
                    (x1 + 4, max(18, y1 - 6)),
                    color=orange,
                    font_scale=max(font_scale, 0.42),
                    thickness=1,
                    outline_thickness=3,
                )
                continue

            bottom_pt = subimage_point_to_full_image(item['bottom_corrected'], box, padding, image.shape)
            top_pt = subimage_point_to_full_image(item['top'], box, padding, image.shape)
            centroid_pt = subimage_point_to_full_image(item['centroid'], box, padding, image.shape)

            image = draw_axis(image, bottom_pt, top_pt, color=(170, 60, 190), thickness=line_thickness)
            cv2.circle(image, bottom_pt, point_radius, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, top_pt, point_radius, (255, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(image, centroid_pt, max(1, point_radius - 1), (0, 255, 255), -1, lineType=cv2.LINE_AA)

        image = draw_qc_legend(
            image,
            [
                ('axis', (170, 60, 190), 'line'),
                ('bottom', (0, 0, 255), 'bottom'),
                ('top', (255, 0, 255), 'top'),
                ('centroid', (0, 255, 255), 'centroid'),
                ('Round', (0, 165, 255), 'box'),
            ],
            origin=(18, 28),
            font_scale=font_scale,
        )

        overlay_path = os.path.join(overlay_dir, f'measurement_overlay_{image_name}')
        save_image(image, overlay_path)
        saved_count += 1

    return saved_count




def save_circularity_overlay_images(overlay_items_by_image, image_dir, overlay_dir):
    """
    Save one full-tray circularity overlay per original image.

    Per kernel:
      - draw the circularity value near the kernel
    """
    if not overlay_items_by_image:
        return 0

    os.makedirs(overlay_dir, exist_ok=True)
    saved_count = 0

    for image_name, items in overlay_items_by_image.items():
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f'  WARNING: Could not read original image for circularity overlay: {image_path}')
            continue
        _, _, font_scale = overlay_scale(image)

        for item in items:
            box = item.get('box')
            if box is None:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            circularity = float(item.get('circularity', 0.0))
            label = f'{circularity:.3f}'
            text_color = (255, 255, 255)

            text_x = x1
            text_y = max(20, y1 - 8)

            image = draw_text_with_outline(
                image,
                label,
                (text_x, text_y),
                color=text_color,
                font_scale=font_scale,
                thickness=1,
            )

        overlay_path = os.path.join(overlay_dir, f'circularity_overlay_{image_name}')
        save_image(image, overlay_path)
        saved_count += 1

    return saved_count


# ---------------------------------------------------------------------------
# Main stage runner
# ---------------------------------------------------------------------------


