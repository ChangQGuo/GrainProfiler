"""
detection/yolo_detect.py
========================
Stage 2 - Detect individual kernel bounding boxes in each full tray image.

READS:   Full tray images (config -> input.image_dir)
WRITES:  yolo_bounding_boxes.json
           Format:
             {
               "IMG_14.jpg": {
                 "accepted_boxes": [[x1, y1, x2, y2], ...],
                 "accepted_detections": [
                   {"candidate_id": 1, "kernel_id": 1, "box": [x1, y1, x2, y2]},
                   ...
                 ],
                 "detections": [
                   {
                     "candidate_id": 1,
                     "kernel_id": 1,
                     "box": [x1, y1, x2, y2],
                     "deleted": false,
                     "filter_reasons": []
                   }
                 ],
                 "summary": {...}
               }
             }
         QC images with accepted and filtered boxes drawn on the original image

The trained YOLO model detects every kernel in the image as a bounding box.
This stage intentionally keeps the YOLO result close to the raw prediction, then
removes only errors that are easy to justify from the tray context:
  1. boxes outside the detected blue tray
  2. boxes with impossible aspect ratio
  3. boxes that are extreme size outliers relative to tray-internal boxes
  4. contained/overlapping boxes that are less plausible than their neighbor

The final kernel_id numbering is always consistent with the subimage numbering
used in the segmentation stage.

RUN WITH: ultralytics conda environment
  conda activate ultralytics
  python detection/yolo_detect.py config.yaml
"""

import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calibration import detect_reference_tray
from utils.device import resolve_stage_device


from utils.config import load_config


def normalize_box(box, image_shape):
    """Clamp a predicted box to image bounds and convert to ints."""
    height, width = image_shape[:2]
    x1 = max(0, min(width - 1, int(round(float(box[0])))))
    y1 = max(0, min(height - 1, int(round(float(box[1])))))
    x2 = max(0, min(width, int(round(float(box[2])))))
    y2 = max(0, min(height, int(round(float(box[3])))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return [x1, y1, x2, y2]


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def box_overlap_fraction(box, region_box):
    area = box_area(box)
    if area <= 0:
        return 0.0
    return float(box_intersection_area(box, region_box) / float(area))


def box_contains_point(box, x, y):
    x1, y1, x2, y2 = box
    return bool(x1 <= x <= x2 and y1 <= y <= y2)


def padded_box(box, image_shape, padding_px):
    x1, y1, x2, y2 = box
    return normalize_box([x1 - padding_px, y1 - padding_px, x2 + padding_px, y2 + padding_px], image_shape)


def clipped_box(box, region_box):
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region_box
    return [max(x1, rx1), max(y1, ry1), min(x2, rx2), min(y2, ry2)]


def add_filter_reason(det, reason):
    if reason not in det["filter_reasons"]:
        det["filter_reasons"].append(reason)
    det["filtered"] = True
    det["deleted"] = True


def update_box_geometry(det, box):
    x1, y1, x2, y2 = [int(value) for value in box]
    det["box"] = [x1, y1, x2, y2]
    det["width"] = int(x2 - x1)
    det["height"] = int(y2 - y1)
    det["area"] = int(det["width"] * det["height"])
    det["long_side"] = int(max(det["width"], det["height"]))
    det["short_side"] = int(min(det["width"], det["height"]))
    det["aspect_ratio"] = float(det["long_side"] / max(det["short_side"], 1))
    det["center_x"] = float((x1 + x2) / 2.0)
    det["center_y"] = float((y1 + y2) / 2.0)


def build_detection_records(result, image_shape):
    """Convert YOLO outputs into plain dict records with geometry metadata."""
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None

    for idx, box in enumerate(xyxy):
        x1, y1, x2, y2 = normalize_box(box, image_shape)
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        long_side = max(box_width, box_height)
        short_side = min(box_width, box_height)
        detections.append(
            {
                "raw_index": idx,
                "candidate_id": None,
                "kernel_id": None,
                "box": [x1, y1, x2, y2],
                "confidence": float(confs[idx]) if confs is not None else None,
                "width": int(box_width),
                "height": int(box_height),
                "area": int(box_area),
                "long_side": int(long_side),
                "short_side": int(short_side),
                "aspect_ratio": float(long_side / max(short_side, 1)),
                "center_x": float((x1 + x2) / 2.0),
                "center_y": float((y1 + y2) / 2.0),
                "edge_distance_px": None,
                "tray_overlap_fraction": None,
                "center_in_tray": None,
                "outside_tray": False,
                "near_edge": False,
                "bad_aspect_ratio": False,
                "size_outlier": False,
                "overlap_suppressed": False,
                "overlap_suppressed_by_candidate_id": None,
                "filtered": False,
                "deleted": False,
                "filter_reasons": [],
            }
        )

    return detections


def sort_detections_reading_order(detections):
    """
    Sort boxes in a stable top-to-bottom, left-to-right order.

    A simple row grouping is used so IDs are easy to follow in QC images.
    """
    if not detections:
        return detections

    median_height = float(np.median([det["height"] for det in detections]))
    row_tolerance = max(10.0, median_height * 0.6)

    rows = []
    for det in sorted(detections, key=lambda item: (item["center_y"], item["center_x"])):
        placed = False
        for row in rows:
            if abs(det["center_y"] - row["anchor_y"]) <= row_tolerance:
                row["items"].append(det)
                row["anchor_y"] = float(
                    np.mean([item["center_y"] for item in row["items"]])
                )
                placed = True
                break
        if not placed:
            rows.append({"anchor_y": det["center_y"], "items": [det]})

    ordered = []
    for row in sorted(rows, key=lambda item: item["anchor_y"]):
        ordered.extend(sorted(row["items"], key=lambda item: item["center_x"]))

    for candidate_id, det in enumerate(ordered, start=1):
        det["candidate_id"] = candidate_id

    return ordered


def get_active_detections(detections):
    return [det for det in detections if not det["deleted"]]


def summarize_box_size(detections):
    """Median box geometry for currently plausible tray-internal detections."""
    if not detections:
        return None
    return {
        "median_area": float(np.median([det["area"] for det in detections])),
        "median_width": float(np.median([det["width"] for det in detections])),
        "median_height": float(np.median([det["height"] for det in detections])),
        "median_long_side": float(np.median([det["long_side"] for det in detections])),
        "median_short_side": float(np.median([det["short_side"] for det in detections])),
    }


def log_size_distance(value, reference):
    if value <= 0 or reference is None or reference <= 0:
        return 0.0
    return abs(float(np.log(float(value) / float(reference))))


def box_quality_distance(det, size_ref):
    """
    Smaller is better: close to tray median size, compact enough, and confident.
    Used only to choose between boxes that already strongly overlap.
    """
    if size_ref is None:
        size_score = 0.0
    else:
        size_score = (
            log_size_distance(det["area"], size_ref["median_area"])
            + 0.45 * log_size_distance(det["long_side"], size_ref["median_long_side"])
            + 0.45 * log_size_distance(det["short_side"], size_ref["median_short_side"])
        )
    conf = float(det["confidence"]) if det.get("confidence") is not None else 0.0
    return float(size_score - 0.25 * conf)


def choose_overlap_suppression_candidate(det_a, det_b, size_ref, detection_cfg):
    """
    Pick which strongly-overlapping box to suppress.

    The decision is deliberately conservative: if one box is a tiny partial box,
    suppress that one; if one box is a large fused box, suppress that one;
    otherwise keep the box closer to the tray median and with better confidence.
    """
    if det_a["area"] <= det_b["area"]:
        smaller, larger = det_a, det_b
    else:
        smaller, larger = det_b, det_a

    if size_ref is not None:
        min_area_ratio = float(detection_cfg.get("min_area_ratio", 0.35))
        max_area_ratio = float(detection_cfg.get("max_area_ratio", 3.00))
        median_area = size_ref["median_area"]
        if smaller["area"] < median_area * min_area_ratio:
            return smaller, larger
        if larger["area"] > median_area * max_area_ratio:
            return larger, smaller

    score_a = box_quality_distance(det_a, size_ref)
    score_b = box_quality_distance(det_b, size_ref)
    if abs(score_a - score_b) > 0.05:
        return (det_a, det_b) if score_a > score_b else (det_b, det_a)

    conf_a = float(det_a["confidence"]) if det_a.get("confidence") is not None else 0.0
    conf_b = float(det_b["confidence"]) if det_b.get("confidence") is not None else 0.0
    if abs(conf_a - conf_b) > 0.05:
        return (det_a, det_b) if conf_a < conf_b else (det_b, det_a)

    return larger, smaller


def suppress_overlapping_boxes(detections, size_ref, detection_cfg):
    """Suppress nested/fused boxes without touching normal neighboring kernels."""
    if len(detections) < 2:
        return 0

    suppressed_count = 0
    nested_overlap_min = float(detection_cfg.get("nested_overlap_min", 0.86))
    nested_area_ratio = float(detection_cfg.get("nested_area_ratio", 1.15))

    pairs = []
    active = get_active_detections(detections)
    for idx_a in range(len(active)):
        for idx_b in range(idx_a + 1, len(active)):
            det_a = active[idx_a]
            det_b = active[idx_b]
            inter_area = box_intersection_area(det_a["box"], det_b["box"])
            if inter_area <= 0:
                continue
            smaller_area = max(1, min(det_a["area"], det_b["area"]))
            overlap_min = inter_area / float(smaller_area)
            area_ratio = max(det_a["area"], det_b["area"]) / float(smaller_area)
            if overlap_min >= nested_overlap_min and area_ratio >= nested_area_ratio:
                pairs.append((overlap_min, area_ratio, det_a, det_b))

    for _, _, det_a, det_b in sorted(pairs, key=lambda item: (item[0], item[1]), reverse=True):
        if det_a["deleted"] or det_b["deleted"]:
            continue
        suppress_det, keep_det = choose_overlap_suppression_candidate(det_a, det_b, size_ref, detection_cfg)
        suppress_det["overlap_suppressed"] = True
        suppress_det["overlap_suppressed_by_candidate_id"] = int(keep_det["candidate_id"])
        add_filter_reason(suppress_det, "overlap_contained")
        suppressed_count += 1

    multi_center_min_count = int(detection_cfg.get("multi_center_min_count", 2))
    multi_center_area_ratio = float(detection_cfg.get("multi_center_area_ratio", 1.25))

    if size_ref is not None:
        active = sorted(get_active_detections(detections), key=lambda item: item["area"], reverse=True)
        for det in active:
            if det["deleted"] or det["area"] < size_ref["median_area"] * multi_center_area_ratio:
                continue
            inside_ids = []
            for other in active:
                if other is det or other["deleted"]:
                    continue
                if box_contains_point(det["box"], other["center_x"], other["center_y"]):
                    inside_ids.append(int(other["candidate_id"]))
            if len(inside_ids) >= multi_center_min_count:
                det["overlap_suppressed"] = True
                det["overlap_suppressed_by_candidate_id"] = min(inside_ids)
                add_filter_reason(det, "overlap_multi_center")
                suppressed_count += 1

    return suppressed_count


def evaluate_filter_rules(detections, image_shape, detection_cfg, tray_info=None):
    """Remove outside-tray boxes, impossible boxes, size outliers, and bad overlaps."""
    height, width = image_shape[:2]

    edge_margin_ratio = float(detection_cfg.get("edge_margin_ratio", 0.02))
    edge_margin_px_min = int(detection_cfg.get("edge_margin_px_min", 10))
    edge_margin_px = max(edge_margin_px_min, int(round(min(height, width) * edge_margin_ratio)))

    tray_filter_enabled = bool(detection_cfg.get("tray_filter_enabled", True))
    tray_box = tray_info.get("box") if tray_info else None
    tray_region = None
    tray_padding_px = None
    if tray_filter_enabled and tray_box:
        tx1, ty1, tx2, ty2 = tray_box
        tray_side = min(max(1, tx2 - tx1), max(1, ty2 - ty1))
        tray_padding_px = int(round(tray_side * float(detection_cfg.get("tray_box_padding_ratio", 0.005))))
        tray_region = padded_box(tray_box, image_shape, tray_padding_px)

    min_box_tray_overlap = float(detection_cfg.get("min_box_tray_overlap", 0.55))
    require_center_in_tray = bool(detection_cfg.get("require_box_center_in_tray", True))
    clip_boxes_to_tray = bool(detection_cfg.get("clip_boxes_to_tray", True))

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        det["edge_distance_px"] = int(min(x1, y1, width - x2, height - y2))

        if tray_region is not None:
            center_in_tray = box_contains_point(tray_region, det["center_x"], det["center_y"])
            tray_overlap = box_overlap_fraction(det["box"], tray_region)
            det["center_in_tray"] = bool(center_in_tray)
            det["tray_overlap_fraction"] = round(float(tray_overlap), 4)

            if (require_center_in_tray and not center_in_tray) or tray_overlap < min_box_tray_overlap:
                det["outside_tray"] = True
                add_filter_reason(det, "outside_tray")
                continue

            if clip_boxes_to_tray:
                clipped = clipped_box(det["box"], tray_region)
                if box_area(clipped) > 0 and clipped != det["box"]:
                    update_box_geometry(det, clipped)
                    det["tray_overlap_fraction"] = 1.0
        else:
            det["near_edge"] = bool(
                x1 <= edge_margin_px
                or y1 <= edge_margin_px
                or x2 >= width - edge_margin_px
                or y2 >= height - edge_margin_px
            )
            if det["near_edge"]:
                add_filter_reason(det, "near_image_edge")

    max_box_aspect_ratio = float(detection_cfg.get("max_box_aspect_ratio", 4.00))
    for det in get_active_detections(detections):
        if det["aspect_ratio"] > max_box_aspect_ratio:
            det["bad_aspect_ratio"] = True
            add_filter_reason(det, "bad_aspect_ratio")

    min_boxes_for_size_filter = int(detection_cfg.get("min_boxes_for_size_filter", 8))
    min_area_ratio = float(detection_cfg.get("min_area_ratio", 0.35))
    max_area_ratio = float(detection_cfg.get("max_area_ratio", 3.00))
    min_side_ratio = float(detection_cfg.get("min_side_ratio", 0.45))
    max_side_ratio = float(detection_cfg.get("max_side_ratio", 2.20))

    size_ref = summarize_box_size(get_active_detections(detections))
    size_filter_applied = (
        size_ref is not None
        and len(get_active_detections(detections)) >= min_boxes_for_size_filter
    )

    if size_filter_applied:
        for det in get_active_detections(detections):
            too_small = (
                det["area"] < size_ref["median_area"] * min_area_ratio
                or det["long_side"] < size_ref["median_long_side"] * min_side_ratio
                or det["short_side"] < size_ref["median_short_side"] * min_side_ratio
            )
            too_large = (
                det["area"] > size_ref["median_area"] * max_area_ratio
                or det["long_side"] > size_ref["median_long_side"] * max_side_ratio
                or det["short_side"] > size_ref["median_short_side"] * max_side_ratio
            )

            if too_small:
                add_filter_reason(det, "too_small")
            if too_large:
                add_filter_reason(det, "too_large")
            det["size_outlier"] = bool(too_small or too_large)

    overlap_filtered = suppress_overlapping_boxes(detections, size_ref, detection_cfg)

    return {
        "tray_filter_enabled": bool(tray_filter_enabled),
        "tray_found": bool(tray_region is not None),
        "tray_color_name": tray_info.get("color_name") if tray_info else None,
        "tray_box": [int(value) for value in tray_box] if tray_box else None,
        "tray_region": [int(value) for value in tray_region] if tray_region else None,
        "tray_padding_px": tray_padding_px,
        "edge_margin_px": int(edge_margin_px),
        "size_filter_applied": bool(size_filter_applied),
        "median_box_area": round(size_ref["median_area"], 2) if size_ref else None,
        "median_box_width": round(size_ref["median_width"], 2) if size_ref else None,
        "median_box_height": round(size_ref["median_height"], 2) if size_ref else None,
        "median_box_long_side": round(size_ref["median_long_side"], 2) if size_ref else None,
        "median_box_short_side": round(size_ref["median_short_side"], 2) if size_ref else None,
        "outside_tray": sum(1 for det in detections if det["outside_tray"]),
        "near_image_edge": sum(1 for det in detections if det["near_edge"]),
        "bad_aspect_ratio": sum(1 for det in detections if det["bad_aspect_ratio"]),
        "size_outliers": sum(1 for det in detections if det["size_outlier"]),
        "overlap_filtered": int(overlap_filtered),
    }


def draw_text(image, text, origin, color, scale=0.65, thickness=2):
    """Draw readable text with a dark outline for visibility."""
    x, y = origin
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def get_qc_text_style(
    image,
    text=None,
    base_scale=0.65,
    base_thickness=2,
    reference_dim=2200,
    max_boost=1.35,
    max_width=None,
    min_scale=0.55,
):
    """Keep QC annotations readable without letting long labels dominate the frame."""
    max_dim = float(max(image.shape[:2]))
    boost = 1.0
    if max_dim > reference_dim:
        boost = min(max_boost, (max_dim / float(reference_dim)) ** 0.5)

    scale = base_scale * boost
    if text and max_width is not None and max_width > 0:
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, base_thickness)
        if text_w > max_width:
            scale = max(min_scale, scale * (max_width / float(text_w)))

    scale_ratio = scale / max(base_scale, 1e-6)
    thickness = max(1, int(round(base_thickness * min(1.5, max(1.0, scale_ratio)))))
    return scale, thickness


def render_detection_qc(image_bgr, detections, tray_region=None):
    """
    Draw accepted detections and filtered boxes together.

    Red labels are final kernel IDs passed downstream. Blue labels are candidate
    IDs removed by the tray/size/overlap filters.
    """
    result = image_bgr.copy()
    label_scale, label_thickness = get_qc_text_style(result, base_scale=0.62, base_thickness=2)

    if tray_region:
        tx1, ty1, tx2, ty2 = tray_region
        cv2.rectangle(result, (tx1, ty1), (tx2, ty2), color=(255, 255, 0), thickness=3)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        if det.get("deleted", False):
            color = (255, 0, 0)
            label = f"{det['candidate_id']} X"
        else:
            color = (0, 0, 255)
            label = str(det["kernel_id"])
        cv2.rectangle(result, (x1, y1), (x2, y2), color=color, thickness=2)
        draw_text(result, label, (x1 + 3, y1 + 20), color=color, scale=label_scale, thickness=label_thickness)

    accepted_count = sum(1 for det in detections if not det["deleted"])
    filtered_count = sum(1 for det in detections if det["deleted"])
    header_1 = f"Detection QC | raw={len(detections)} | accepted={accepted_count} | filtered={filtered_count}"
    header_2 = "Red = accepted | Blue X = filtered | Cyan = tray ROI"
    header_width = max(240, result.shape[1] - 30)
    header_1_scale, header_1_thickness = get_qc_text_style(
        result,
        text=header_1,
        base_scale=0.95,
        base_thickness=3,
        max_width=header_width,
    )
    header_2_scale, header_2_thickness = get_qc_text_style(
        result,
        text=header_2,
        base_scale=0.86,
        base_thickness=3,
        max_width=header_width,
    )
    draw_text(result, header_1, (15, 32), color=(255, 255, 255), scale=header_1_scale, thickness=header_1_thickness)
    draw_text(result, header_2, (15, 58), color=(255, 255, 255), scale=header_2_scale, thickness=header_2_thickness)
    return result


def save_image(image_bgr, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image_bgr)


def assign_final_kernel_ids(detections):
    """Assign final sequential IDs only to detections that remain after filtering."""
    accepted_boxes = []
    next_kernel_id = 1

    for det in detections:
        if det["deleted"]:
            det["kernel_id"] = None
            continue
        det["kernel_id"] = next_kernel_id
        accepted_boxes.append(det["box"])
        next_kernel_id += 1

    return accepted_boxes


def get_final_status(det):
    return "filtered" if det["deleted"] else "accepted"


def serialize_detection(det):
    """Keep the JSON output clean and fully serializable."""
    return {
        "candidate_id": int(det["candidate_id"]),
        "kernel_id": int(det["kernel_id"]) if det["kernel_id"] is not None else None,
        "box": [int(value) for value in det["box"]],
        "confidence": round(float(det["confidence"]), 6) if det["confidence"] is not None else None,
        "width": int(det["width"]),
        "height": int(det["height"]),
        "area": int(det["area"]),
        "long_side": int(det["long_side"]),
        "short_side": int(det["short_side"]),
        "aspect_ratio": round(float(det["aspect_ratio"]), 4),
        "center_x": round(float(det["center_x"]), 2),
        "center_y": round(float(det["center_y"]), 2),
        "edge_distance_px": int(det["edge_distance_px"]) if det["edge_distance_px"] is not None else None,
        "tray_overlap_fraction": (
            round(float(det["tray_overlap_fraction"]), 4)
            if det.get("tray_overlap_fraction") is not None else None
        ),
        "center_in_tray": (
            bool(det["center_in_tray"]) if det.get("center_in_tray") is not None else None
        ),
        "outside_tray": bool(det["outside_tray"]),
        "near_edge": bool(det["near_edge"]),
        "bad_aspect_ratio": bool(det["bad_aspect_ratio"]),
        "size_outlier": bool(det["size_outlier"]),
        "overlap_suppressed": bool(det["overlap_suppressed"]),
        "overlap_suppressed_by_candidate_id": (
            int(det["overlap_suppressed_by_candidate_id"])
            if det.get("overlap_suppressed_by_candidate_id") is not None else None
        ),
        "filter_reasons": list(det["filter_reasons"]),
        "filtered": bool(det["filtered"]),
        "deleted": bool(det["deleted"]),
        "kept": bool(not det["deleted"]),
        "final_status": get_final_status(det),
    }


def build_accepted_detection_list(detections):
    """Explicit list of detections that downstream stages should keep."""
    accepted = []
    for det in detections:
        if det["deleted"]:
            continue
        accepted.append(
            {
                "candidate_id": int(det["candidate_id"]),
                "kernel_id": int(det["kernel_id"]),
                "box": [int(value) for value in det["box"]],
                "confidence": round(float(det["confidence"]), 6) if det["confidence"] is not None else None,
                "final_status": get_final_status(det),
            }
        )
    return accepted


def format_id_list(values):
    """Serialize a list of IDs for CSV storage."""
    return ",".join(str(value) for value in values)


def save_detection_results_csv(all_boxes, csv_path):
    """Write one summary row per image for fast QC review in spreadsheets."""
    fieldnames = [
        "image_name",
        "raw_detections",
        "filtered",
        "accepted",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for image_name in sorted(all_boxes.keys()):
            image_entry = all_boxes[image_name]
            summary = image_entry.get("summary") or {}

            writer.writerow(
                {
                    "image_name": image_name,
                    "raw_detections": summary.get("raw_detections", 0),
                    "filtered": summary.get("filtered", 0),
                    "accepted": summary.get("accepted", 0),
                }
            )


def empty_image_entry():
    """Return the JSON shape used when an image cannot be read."""
    return {
        "image_size": None,
        "accepted_boxes": [],
        "accepted_detections": [],
        "detections": [],
        "summary": {
            "raw_detections": 0,
            "filtered": 0,
            "accepted": 0,
            "tray_filter_enabled": False,
            "tray_found": False,
            "tray_color_name": None,
            "tray_box": None,
            "tray_region": None,
            "tray_padding_px": None,
            "outside_tray": 0,
            "near_image_edge": 0,
            "bad_aspect_ratio": 0,
            "size_outliers": 0,
            "overlap_filtered": 0,
            "size_filter_applied": False,
            "edge_margin_px": None,
            "median_box_area": None,
            "median_box_width": None,
            "median_box_height": None,
            "median_box_long_side": None,
            "median_box_short_side": None,
        },
        "final_qc_path": None,
    }


def run(config_path):
    config = load_config(config_path)
    detection_cfg = config["detection"]

    image_dir = config["input"]["image_dir"]
    ext = config["input"]["image_extension"]
    output_dir = config["output"]["base_dir"]
    bbox_json = os.path.join(output_dir, config["output"]["bounding_boxes_json"])
    detect_results_csv = os.path.join(output_dir, config["output"]["detect_results_csv"])
    qc_dir = os.path.join(output_dir, config["output"]["qc_dir"], "detection")
    save_qc = bool(config.get("visualization", {}).get("save_qc", True))

    model_path = config["models"]["yolo_detection"]
    conf = detection_cfg["confidence"]
    nms_iou_value = detection_cfg.get("nms_iou", None)
    nms_iou = None if nms_iou_value in (None, "") else float(nms_iou_value)
    inference_size_value = detection_cfg.get("inference_size", None)
    inference_size = None if inference_size_value in (None, "") else int(inference_size_value)
    max_det = detection_cfg["max_detections"]
    device = resolve_stage_device(config, "detection")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed in this environment.")
        print("Run this script with: conda activate ultralytics")
        sys.exit(1)

    print(
        f"Loading YOLO model: {model_path}"
        f" | conf={conf}"
        f" | iou={nms_iou if nms_iou is not None else 'ultralytics_default'}"
        f" | imgsz={inference_size if inference_size is not None else 'ultralytics_default'}"
        f" | device={device}"
    )
    model = YOLO(model_path)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    if not image_files:
        print(f"ERROR: No {ext} images found in {image_dir}")
        sys.exit(1)

    print(f"Detecting kernels in {len(image_files)} images...")
    all_boxes = {}

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            print(f"  {image_file} -> WARNING: Could not read image, saving empty result")
            all_boxes[image_file] = empty_image_entry()
            continue

        print(f"  {image_file}", end=" ")

        predict_kwargs = {
            "source": image_path,
            "conf": conf,
            "max_det": max_det,
            "device": device,
            "verbose": False,
        }
        if nms_iou is not None:
            predict_kwargs["iou"] = nms_iou
        if inference_size is not None:
            predict_kwargs["imgsz"] = inference_size

        try:
            results = model.predict(**predict_kwargs)
            result = results[0]
        except Exception as e:
            print(f"-> WARNING: detection inference failed: {e}")
            all_boxes[image_file] = empty_image_entry()
            continue

        detections = build_detection_records(result, image_bgr.shape)
        detections = sort_detections_reading_order(detections)
        tray_info = detect_reference_tray(image_bgr, config.get("calibration", {}))
        filter_summary = evaluate_filter_rules(detections, image_bgr.shape, detection_cfg, tray_info=tray_info)
        accepted_boxes = assign_final_kernel_ids(detections)

        final_qc_path = os.path.join(qc_dir, f"det_{image_file}")
        if save_qc:
            final_image = render_detection_qc(image_bgr, detections, tray_region=filter_summary.get("tray_region"))
            save_image(final_image, final_qc_path)

        raw_count = len(detections)
        filtered_count = sum(1 for det in detections if det["deleted"])
        accepted_count = len(accepted_boxes)
        outside_tray_count = sum(1 for det in detections if det["outside_tray"])
        near_image_edge_count = sum(1 for det in detections if det["near_edge"])
        bad_aspect_count = sum(1 for det in detections if det["bad_aspect_ratio"])
        size_outlier_count = sum(1 for det in detections if det["size_outlier"])
        overlap_filtered_count = sum(1 for det in detections if det["overlap_suppressed"])

        all_boxes[image_file] = {
            "image_size": {
                "width": int(image_bgr.shape[1]),
                "height": int(image_bgr.shape[0]),
            },
            "accepted_boxes": accepted_boxes,
            "accepted_detections": build_accepted_detection_list(detections),
            "detections": [serialize_detection(det) for det in detections],
            "summary": {
                "raw_detections": raw_count,
                "filtered": filtered_count,
                "accepted": accepted_count,
                "outside_tray": outside_tray_count,
                "near_image_edge": near_image_edge_count,
                "bad_aspect_ratio": bad_aspect_count,
                "size_outliers": size_outlier_count,
                "overlap_filtered": overlap_filtered_count,
                **filter_summary,
            },
            "final_qc_path": final_qc_path if save_qc else None,
        }

        print(
            f"-> raw={raw_count}, accepted={accepted_count}, "
            f"filtered={filtered_count}, tray/size/overlap cleanup"
        )

    with open(bbox_json, "w", encoding="utf-8") as f:
        json.dump(all_boxes, f, indent=2)

    save_detection_results_csv(all_boxes, detect_results_csv)

    total_raw = sum(item["summary"]["raw_detections"] for item in all_boxes.values())
    total_filtered = sum(item["summary"]["filtered"] for item in all_boxes.values())
    total_accepted = sum(item["summary"]["accepted"] for item in all_boxes.values())

    print(f"\nDetected {total_raw} raw boxes across {len(all_boxes)} images.")
    print(f"Filtered {total_filtered} outside-tray/size/overlap detections.")
    print(f"Accepted {total_accepted} detections for downstream segmentation.")
    print(f"Bounding boxes saved -> {bbox_json}")
    print(f"Detection summary CSV saved -> {detect_results_csv}")
    print("Check detection QC images in:", qc_dir)

    # --- QC cleanup ---
    _cleanup_qc(config, output_dir)


def _cleanup_qc(config, output_dir):
    import shutil
    vis = config.get('visualization', {})
    if not vis.get('save_detection_qc', True):
        qc_dir = os.path.join(output_dir, config['output'].get('qc_dir', 'qc_visualizations'), 'detection')
        if os.path.exists(qc_dir):
            shutil.rmtree(qc_dir)
            print(f'  Cleaned: qc_visualizations/detection')


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
