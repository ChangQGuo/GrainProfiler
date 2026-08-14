"""qc_render.py - OCR QC image rendering helpers (pure view layer).

Extracted from metadata_extraction.py.  Depends only on cv2 / numpy / os; it does
not perform OCR or text parsing.
"""

import os

import cv2
import numpy as np


def draw_text(image, text, origin, color, scale=0.7, thickness=2):
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
    base_scale=0.7,
    base_thickness=2,
    reference_dim=2200,
    max_boost=1.35,
    max_width=None,
    min_scale=0.55,
):
    """Keep QC annotations readable while preventing oversized text on large images."""
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




def draw_text_centered(image, text, center_x, baseline_y, color, scale=1.0, thickness=2):
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(10, int(round(center_x - text_w / 2.0)))
    y = max(text_h + 10, int(round(baseline_y)))
    draw_text(image, text, (x, y), color, scale=scale, thickness=thickness)




def draw_dashed_line(image, pt1, pt2, color, thickness=2, dash_length=18, gap_length=10):
    x1, y1 = pt1
    x2, y2 = pt2
    total_length = float(np.hypot(x2 - x1, y2 - y1))
    if total_length <= 1e-6:
        return

    dx = (x2 - x1) / total_length
    dy = (y2 - y1) / total_length
    step = dash_length + gap_length
    distance = 0.0
    while distance < total_length:
        start_x = int(round(x1 + dx * distance))
        start_y = int(round(y1 + dy * distance))
        end_distance = min(distance + dash_length, total_length)
        end_x = int(round(x1 + dx * end_distance))
        end_y = int(round(y1 + dy * end_distance))
        cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
        distance += step




def serialize_box(box):
    return ",".join(str(int(v)) for v in box) if box else ""





def save_ocr_qc_image(
    image_bgr,
    qc_path,
    split_y,
    plant_name,
    weight_g,
    label_candidate=None,
    weight_candidate=None,
    yolo_label_box=None,
    yolo_weight_box=None,
    yolo_used=False,
):
    """Render a clean OCR QC image with only the final selected boxes.

    When yolo_used is True, YOLO-detected label/weight_screen boxes are drawn
    instead of the dashed split line.
    """
    result = image_bgr.copy()
    section_width = max(240, result.shape[1] - 24)

    if yolo_used:
        # YOLO-guided mode: draw the detected ROIs
        if yolo_label_box is not None:
            x1, y1, x2, y2 = yolo_label_box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label_text = "YOLO: Label"
            label_scale, label_thickness = get_qc_text_style(
                result, text=label_text, base_scale=1.08, base_thickness=3,
                max_width=max(240, result.shape[1] - x1 - 12),
            )
            draw_text(result, label_text, (x1, max(26, y1 - 10)),
                      (0, 255, 0), label_scale, label_thickness)

        if yolo_weight_box is not None:
            x1, y1, x2, y2 = yolo_weight_box
            cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 3)
            weight_text = "YOLO: Weight Screen"
            weight_scale, weight_thickness = get_qc_text_style(
                result, text=weight_text, base_scale=1.08, base_thickness=3,
                max_width=max(240, result.shape[1] - x1 - 12),
            )
            draw_text(result, weight_text, (x1, max(26, y1 - 10)),
                      (255, 0, 0), weight_scale, weight_thickness)
    else:
        upper_label = "UPPER (Plant Name)"
        lower_label = "LOWER (Weight)"
        section_scale, section_thickness = get_qc_text_style(
            result, text=upper_label, base_scale=1.22, base_thickness=3,
            max_width=section_width,
        )
        draw_dashed_line(
            result,
            (0, split_y),
            (result.shape[1] - 1, split_y),
            (255, 255, 0),
            thickness=8,
            dash_length=18,
            gap_length=10,
        )
        draw_text(result, upper_label, (12, max(32, split_y - 10)),
                  (255, 255, 0), section_scale, section_thickness)
        draw_text(result, lower_label,
                  (12, min(result.shape[0] - 12, split_y + 24)),
                  (255, 255, 0), section_scale, section_thickness)

    if label_candidate is not None and label_candidate.get("item") is not None:
        item = label_candidate["item"]
        pts = item["bbox"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 0, 255), 2)
        anchor = tuple(pts[0, 0])
        label_text = item["text"]
        if label_candidate.get("score") is not None:
            label_text += f" ({item['confidence']:.2f})"
        label_scale, label_thickness = get_qc_text_style(
            result,
            text=label_text,
            base_scale=1.08,
            base_thickness=3,
            max_width=max(180, result.shape[1] - anchor[0] - 12),
        )
        draw_text(
            result,
            label_text,
            (anchor[0], max(26, anchor[1] - 10)),
            (0, 0, 255),
            label_scale,
            label_thickness,
        )

    if weight_candidate is not None and weight_candidate["item"].get("search_box"):
        item = weight_candidate["item"]
        pts = item["bbox"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], True, (0, 0, 255), 2)
        anchor = tuple(pts[0, 0])
        weight_text = f"{weight_candidate['value']} ({item['confidence']:.2f})"
        weight_scale, weight_thickness = get_qc_text_style(
            result,
            text=weight_text,
            base_scale=1.00,
            base_thickness=3,
            max_width=max(160, result.shape[1] - anchor[0] - 12),
        )
        draw_text(
            result,
            weight_text,
            (anchor[0], max(26, anchor[1] - 10)),
            (0, 0, 255),
            weight_scale,
            weight_thickness,
        )

    overlay = result.copy()
    banner_height = min(150, max(92, int(round(result.shape[0] * 0.100))))
    cv2.rectangle(overlay, (0, 0), (result.shape[1], banner_height), (0, 0, 0), thickness=-1)
    result = cv2.addWeighted(overlay, 0.38, result, 0.62, 0)

    summary_text = f"Plant: {plant_name or 'NOT FOUND'}  |  Weight: {(weight_g + ' g') if weight_g else 'NOT FOUND'}"
    summary_scale, summary_thickness = get_qc_text_style(
        result,
        text=summary_text,
        base_scale=1.45,
        base_thickness=3,
        max_width=max(260, result.shape[1] - 24),
    )
    draw_text_centered(
        result,
        summary_text,
        result.shape[1] / 2.0,
        max(56, int(round(banner_height * 0.72))),
        (255, 255, 255),
        scale=summary_scale,
        thickness=summary_thickness,
    )

    max_dim = 2200
    height, width = result.shape[:2]
    scale = min(1.0, max_dim / float(max(height, width)))
    if scale < 1.0:
        result = cv2.resize(result, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    cv2.imwrite(qc_path, result)
    print(f"      QC saved: {os.path.basename(qc_path)}")



