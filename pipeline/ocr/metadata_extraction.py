"""
ocr/metadata_extraction.py
==========================
Stage 1 - Extract plant name and weight from each image using split-region OCR - PaddleOCR v2
PaddleOCR https://github.com/PaddlePaddle/PaddleOCR

READS:   Original full tray images (config -> input.image_dir)
WRITES:  metadata.csv  (image_name, plant_name, weight_g, plus ROI debug fields)
         QC images showing the upper/lower OCR scan regions and OCR results

Each image has two pieces of text we care about:
  - Plant name / variety code  (e.g. '23-11-HN-CG-163-04')
  - Weight in grams            (e.g. '8.76')

The plant name is the "plant ID" that links all kernels from the same image together
for downstream analysis. The weight is used to compute per-kernel weight later on.
"""

import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.calibration import clamp_box, detect_reference_tray
from utils.device import paddleocr_device_kwargs, resolve_stage_device


from utils.config import load_config
from ocr.qc_render import (
    draw_text,
    get_qc_text_style,
    draw_text_centered,
    draw_dashed_line,
    serialize_box,
    save_ocr_qc_image,
)
from ocr.text_parse import (
    normalize_weight_text,
    normalize_label_text,
    choose_best_label_candidate,
    extract_weight_candidates,
    summarize_texts,
    choose_best_weight_candidate,
)




def extract_text_from_array(ocr, image_bgr, min_confidence=0.5, offset=(0, 0), local_scale=1.0, source="roi"):
    """
    Run PaddleOCR on one numpy image and return OCR lines with full-image bboxes.

    local_scale is used when OCR is run on an upscaled crop. A local bbox point
    is multiplied by local_scale before the global offset is added.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    try:
        result = ocr.ocr(image_bgr, cls=True)
    except Exception as e:
        print(f'  WARNING: OCR inference failed: {e}')
        return []
    texts = []
    if not result or not result[0]:
        return texts

    offset_x, offset_y = offset
    for line in result[0]:
        bbox = np.array(line[0], dtype=np.float32)
        text = str(line[1][0]).strip()
        confidence = float(line[1][1])
        if confidence < min_confidence:
            continue

        bbox[:, 0] = bbox[:, 0] * local_scale + offset_x
        bbox[:, 1] = bbox[:, 1] * local_scale + offset_y

        texts.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": bbox,
                "source": source,
            }
        )

    return texts


def preprocess_weight_screen_crop(weight_screen_crop, scale):
    """Create OCR-friendly variants of the weight screen crop."""
    scale = max(1.0, float(scale))
    resized = cv2.resize(weight_screen_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return resized, thresh_bgr


def resolve_split_line_ratio(ocr_cfg):
    ratio = ocr_cfg.get("split_line_ratio")
    if ratio is None:
        ratio = ocr_cfg.get(
            "weight_screen_search_top_ratio",
            ocr_cfg.get("label_search_height_ratio", 0.50),
        )

    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 0.50

    return min(0.80, max(0.25, ratio))


def build_split_regions(image_bgr, ocr_cfg):
    height, width = image_bgr.shape[:2]
    split_y = int(round(height * resolve_split_line_ratio(ocr_cfg)))
    split_y = max(1, min(height - 1, split_y))

    label_region = {
        "box": [0, 0, width, split_y],
        "source": "top_split_scan",
    }
    weight_region = {
        "box": [0, split_y, width, height],
        "source": "bottom_split_scan",
    }
    return label_region, weight_region, split_y


def build_weight_search_regions(image_bgr, ocr_cfg, tray_info=None):
    height, width = image_bgr.shape[:2]
    _, bottom_region, split_y = build_split_regions(image_bgr, ocr_cfg)

    regions = [
        {
            "box": bottom_region["box"],
            "source": "bottom_split_scan",
            "priority": 0.45,
        }
    ]

    center_box = clamp_box(
        [
            width * 0.22,
            max(split_y, height * 0.62),
            width * 0.78,
            height,
        ],
        width,
        height,
    )
    regions.append(
        {
            "box": center_box,
            "source": "bottom_center_scan",
            "priority": 1.00,
        }
    )

    if tray_info is not None and tray_info.get("box"):
        tx1, ty1, tx2, ty2 = tray_info["box"]
        tray_w = max(1, tx2 - tx1)
        tray_h = max(1, ty2 - ty1)

        tray_bottom_box = clamp_box(
            [
                tx1 - 0.08 * tray_w,
                max(split_y, ty2 - 0.06 * tray_h),
                tx2 + 0.08 * tray_w,
                min(height, ty2 + 0.42 * tray_h),
            ],
            width,
            height,
        )
        tray_weight_screen_band = clamp_box(
            [
                tx1 + 0.18 * tray_w,
                max(split_y, ty2 - 0.02 * tray_h),
                tx2 - 0.18 * tray_w,
                min(height, ty2 + 0.30 * tray_h),
            ],
            width,
            height,
        )

        regions.append(
            {
                "box": tray_bottom_box,
                "source": "tray_bottom_scan",
                "priority": 1.20,
            }
        )
        regions.append(
            {
                "box": tray_weight_screen_band,
                "source": "tray_weight_screen_band",
                "priority": 1.35,
            }
        )

    deduped = []
    seen_boxes = set()
    for region in regions:
        box_key = tuple(int(v) for v in region["box"])
        if box_key in seen_boxes:
            continue
        seen_boxes.add(box_key)
        deduped.append(region)

    return deduped


def compute_effective_ocr_scale(crop_bgr, requested_scale, max_dim):
    requested_scale = max(1.0, float(requested_scale))
    height, width = crop_bgr.shape[:2]
    largest_dim = max(1, height, width)
    max_scale = max(1.0, float(max_dim) / float(largest_dim))
    return min(requested_scale, max_scale)


def extract_texts_from_search_box(
    ocr,
    image_bgr,
    box,
    min_confidence,
    requested_scale,
    source_prefix,
    max_dim,
    include_binary=True,
):
    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return []

    texts = []
    texts.extend(
        extract_text_from_array(
            ocr,
            crop,
            min_confidence=min_confidence,
            offset=(x1, y1),
            source=f"{source_prefix}_raw",
        )
    )

    effective_scale = compute_effective_ocr_scale(crop, requested_scale, max_dim)
    scaled_crop, binary_crop = preprocess_weight_screen_crop(crop, effective_scale)
    texts.extend(
        extract_text_from_array(
            ocr,
            scaled_crop,
            min_confidence=min_confidence,
            offset=(x1, y1),
            local_scale=1.0 / effective_scale,
            source=f"{source_prefix}_scaled",
        )
    )

    if include_binary:
        texts.extend(
            extract_text_from_array(
                ocr,
                binary_crop,
                min_confidence=min_confidence,
                offset=(x1, y1),
                local_scale=1.0 / effective_scale,
                source=f"{source_prefix}_binary",
            )
        )

    serialized_box = [int(x1), int(y1), int(x2), int(y2)]
    for item in texts:
        item["search_box"] = serialized_box
        item["region_source"] = source_prefix

    return texts


def prompt_manual_weight(image_file, current_value):
    """
    Ask the user to type a weight when OCR could not find one.

    Pressing Enter keeps the current value unchanged.
    """
    print(f"    Manual weight correction requested for image: {image_file}")
    while True:
        try:
            user_text = input("    Enter weight like 11.48, or press Enter to keep NOT FOUND: ").strip()
        except EOFError:
            print("    WARNING: stdin is not interactive. Keeping NOT FOUND.")
            return current_value

        if not user_text:
            return current_value

        normalized = normalize_weight_text(user_text)
        matches = re.findall(r"\d{1,3}\.\d{1,2}", normalized)
        if not matches:
            print("    Invalid input. Please enter a decimal number like 11.48.")
            continue

        try:
            value = float(matches[0])
        except ValueError:
            print("    Invalid input. Please enter a decimal number like 11.48.")
            continue

        return f"{value:.2f}"


def collect_label_texts(ocr, image_bgr, label_region, ocr_cfg):
    """Run OCR on the upper split region and extract the plant label."""
    min_confidence = float(ocr_cfg.get("min_confidence", 0.5))
    label_scale = float(ocr_cfg.get("label_ocr_scale", 1.8))
    max_dim = int(ocr_cfg.get("scan_max_dim", 2600))

    texts = extract_texts_from_search_box(
        ocr,
        image_bgr,
        label_region["box"],
        min_confidence=min_confidence,
        requested_scale=label_scale,
        source_prefix=label_region["source"],
        max_dim=max_dim,
        include_binary=False,
    )
    for item in texts:
        item["region_priority"] = 1.0

    label_candidate = choose_best_label_candidate(texts)
    plant_name = label_candidate["value"] if label_candidate is not None else None
    if plant_name is not None or not bool(ocr_cfg.get("global_fallback", True)):
        return texts, plant_name, label_candidate

    height, width = image_bgr.shape[:2]
    fallback_texts = extract_texts_from_search_box(
        ocr,
        image_bgr,
        [0, 0, width, height],
        min_confidence=min_confidence,
        requested_scale=1.0,
        source_prefix="label_full_fallback",
        max_dim=max_dim,
        include_binary=False,
    )
    for item in fallback_texts:
        item["region_priority"] = 0.25

    texts.extend(fallback_texts)
    label_candidate = choose_best_label_candidate(texts)
    plant_name = label_candidate["value"] if label_candidate is not None else None
    return texts, plant_name, label_candidate


def collect_weight_texts(ocr, image_bgr, weight_regions, ocr_cfg):
    min_confidence = float(ocr_cfg.get("min_confidence", 0.5))
    scale = float(ocr_cfg.get("weight_screen_ocr_scale", 3.0))
    max_dim = int(ocr_cfg.get("scan_max_dim", 2600))

    texts = []
    for region in weight_regions:
        region_texts = extract_texts_from_search_box(
            ocr,
            image_bgr,
            region["box"],
            min_confidence=min_confidence,
            requested_scale=scale,
            source_prefix=region["source"],
            max_dim=max_dim,
            include_binary=True,
        )
        for item in region_texts:
            item["region_priority"] = float(region.get("priority", 0.0))
        texts.extend(region_texts)

    best_candidate = choose_best_weight_candidate(texts, ocr_cfg, image_shape=image_bgr.shape)
    weight_g = best_candidate["value"] if best_candidate is not None else None
    return texts, weight_g, best_candidate


def run(config_path):
    config = load_config(config_path)

    image_dir = config["input"]["image_dir"]
    ext = config["input"]["image_extension"]
    output_dir = config["output"]["base_dir"]
    metadata_csv = os.path.join(output_dir, config["output"]["metadata_csv"])
    qc_dir = os.path.join(output_dir, config["output"]["qc_dir"], "ocr")
    save_qc = bool(config.get("visualization", {}).get("save_qc", True))
    ocr_cfg = config.get("ocr", {})
    calibration_cfg = config.get("calibration", {})
    correct_weight = bool(ocr_cfg.get("correct_weight", False))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    ocr_device = resolve_stage_device(config, "ocr")
    ocr_device_kwargs = paddleocr_device_kwargs(ocr_device)

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: paddleocr is not installed in this environment.")
        print("Run this script with: conda activate paddle")
        sys.exit(1)

    print(f"Loading PaddleOCR model on {ocr_device}...")
    try:
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            **ocr_device_kwargs,
        )
    except TypeError:
        if "gpu_id" not in ocr_device_kwargs:
            raise
        fallback_kwargs = dict(ocr_device_kwargs)
        fallback_kwargs.pop("gpu_id", None)
        print("WARNING: This PaddleOCR version does not accept gpu_id; retrying with use_gpu only.")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            **fallback_kwargs,
        )

    # --- Load pre-computed YOLO boxes + weight values (from pre_ocr stage) ---
    yolo_boxes = {}
    yolo_json_path = os.path.join(output_dir, config["output"].get(
        "yolo_label_weight_json", "yolo_label_weight_boxes.json"
    ))
    if os.path.exists(yolo_json_path):
        with open(yolo_json_path, "r") as f:
            raw_boxes = json.load(f)
        for entry in raw_boxes:
            img_name = entry.get("image_name", "")
            yolo_boxes[img_name] = {
                "label_box": entry.get("label_box"),
                "weight_screen_box": entry.get("weight_screen_box"),
                "weight_g": entry.get("weight_g"),
            }
        found_label = sum(1 for v in yolo_boxes.values() if v["label_box"])
        found_ws = sum(1 for v in yolo_boxes.values() if v["weight_screen_box"])
        found_wg = sum(1 for v in yolo_boxes.values() if v["weight_g"])
        print(f"Loaded YOLO boxes for {len(yolo_boxes)} images from {os.path.basename(yolo_json_path)}")
        print(f"  Label boxes: {found_label}  Weight-screen: {found_ws}  "
              f"Weight values: {found_wg}")
    else:
        print(f"WARNING: {os.path.basename(yolo_json_path)} not found. "
              "Using split-region OCR for all images.")
        print("  Run the pre_ocr stage first: python main.py config.yaml --stage pre_ocr")

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    if not image_files:
        print(f"ERROR: No {ext} images found in {image_dir}")
        sys.exit(1)

    print(f"Processing {len(image_files)} images...")
    records = []
    fail_label = 0
    fail_weight = 0

    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print("    WARNING: Could not read image, skipping")
            continue

        tray_info = detect_reference_tray(image_bgr, calibration_cfg)
        yolo_used = False
        yolo_label_box = None
        yolo_weight_box = None

        # --- YOLO-guided ROI detection (boxes + weight pre-computed by pre_ocr stage) ---
        yolo_entry = yolo_boxes.get(image_file, {})
        precomputed_weight = None
        if yolo_entry:
            label_entry = yolo_entry.get("label_box")
            weight_entry = yolo_entry.get("weight_screen_box")
            precomputed_weight = yolo_entry.get("weight_g")
            if label_entry and label_entry.get("box"):
                yolo_label_box = label_entry["box"]
            if weight_entry and weight_entry.get("box"):
                yolo_weight_box = weight_entry["box"]

            yolo_used = bool(yolo_label_box or yolo_weight_box)

        # --- Build label region ---
        if yolo_label_box is not None:
            label_region = {
                "box": yolo_label_box,
                "source": "yolo_label",
            }
        else:
            label_region, _, split_y_fallback = build_split_regions(image_bgr, ocr_cfg)

        # --- Build weight search regions ---
        if yolo_weight_box is not None:
            # Expand slightly for safety margin
            height, width = image_bgr.shape[:2]
            pad_x = int(round(0.05 * (yolo_weight_box[2] - yolo_weight_box[0])))
            pad_y = int(round(0.05 * (yolo_weight_box[3] - yolo_weight_box[1])))
            expanded_box = clamp_box(
                [
                    yolo_weight_box[0] - pad_x,
                    yolo_weight_box[1] - pad_y,
                    yolo_weight_box[2] + pad_x,
                    yolo_weight_box[3] + pad_y,
                ],
                width,
                height,
            )
            weight_regions = [
                {
                    "box": expanded_box,
                    "source": "yolo_weight_screen",
                    "priority": 1.50,
                }
            ]
        else:
            weight_regions = build_weight_search_regions(image_bgr, ocr_cfg, tray_info=tray_info)

        # Determine split_y for QC (use YOLO boundary or fall back to computed split)
        if yolo_used and yolo_weight_box is not None and yolo_label_box is not None:
            split_y = int(round((yolo_label_box[3] + yolo_weight_box[1]) / 2.0))
        elif yolo_used and yolo_weight_box is not None:
            split_y = yolo_weight_box[1]
        elif yolo_used and yolo_label_box is not None:
            split_y = yolo_label_box[3]
        else:
            _, _, split_y = build_split_regions(image_bgr, ocr_cfg)

        # --- Label OCR (PaddleOCR always) ---
        label_texts, plant_name, label_candidate = collect_label_texts(ocr, image_bgr, label_region, ocr_cfg)
        weight_texts = []  # only populated when PaddleOCR fallback is used
        weight_candidate = None

        # --- Weight: use pre-computed YOLOv8 value when available ---
        if precomputed_weight is not None:
            weight_g = precomputed_weight
        else:
            # Fallback: PaddleOCR on weight region
            weight_texts, weight_g, weight_candidate = collect_weight_texts(ocr, image_bgr, weight_regions, ocr_cfg)

        # --- Fallback: if YOLO found a label box but OCR failed, try full split region ---
        if plant_name is None and yolo_label_box is not None and bool(config.get("label_weight_detection", {}).get("fallback_to_split", True)):
            print("    YOLO label region OCR failed, falling back to split-region label scan...")
            fallback_label_region, _, _ = build_split_regions(image_bgr, ocr_cfg)
            fallback_texts, fallback_name, fallback_candidate = collect_label_texts(
                ocr, image_bgr, fallback_label_region, ocr_cfg
            )
            if fallback_name is not None:
                label_texts.extend(fallback_texts)
                plant_name = fallback_name
                label_candidate = fallback_candidate
                label_region = fallback_label_region

        # --- Fallback: if YOLO found a weight box but no weight yet, try PaddleOCR ---
        if weight_g is None and yolo_weight_box is not None and bool(config.get("label_weight_detection", {}).get("fallback_to_split", True)):
            print("    No pre-computed weight, falling back to PaddleOCR weight scan...")
            fallback_regions = build_weight_search_regions(image_bgr, ocr_cfg, tray_info=tray_info)
            fallback_texts, fallback_weight, fallback_candidate = collect_weight_texts(
                ocr, image_bgr, fallback_regions, ocr_cfg
            )
            if fallback_weight is not None:
                weight_texts = fallback_texts
                weight_g = fallback_weight
                weight_candidate = fallback_candidate

        if plant_name is None:
            print("    WARNING: No plant label found - falling back to filename stem")
            plant_name = os.path.splitext(image_file)[0]

        if weight_g is None:
            print("    WARNING: No weight found in scan regions")
            if correct_weight:
                weight_g = prompt_manual_weight(image_file, weight_g)

        weight_source = (
            weight_candidate["item"].get("region_source")
            if weight_candidate is not None
            else weight_regions[0]["source"] if weight_regions else "unknown"
        )
        if plant_name is None:
            fail_label += 1
        if weight_g is None:
            fail_weight += 1

        print(f"  [{idx}/{len(image_files)}] Plant: {plant_name} | Weight: {weight_g}")
        if tray_info is not None:
            print(
                f"    Tray: {tray_info['tray_side_px']:.1f}px = {tray_info['tray_size_mm']:.1f}mm"
                f" | Scale: {tray_info['mm_per_px']:.4f} mm/px"
            )
        elif bool(calibration_cfg.get("enabled", False)):
            print("    WARNING: Blue tray calibration failed for this image")

        records.append(
            {
                "image_name": image_file,
                "plant_name": plant_name,
                "weight_g": weight_g if weight_g else "UNKNOWN",
                "label_roi_source": label_region["source"],
                "weight_screen_roi_source": weight_source,
                "label_roi_box": serialize_box(label_region["box"]),
                "weight_screen_roi_box": serialize_box(
                    weight_candidate["item"]["search_box"]
                    if weight_candidate is not None and weight_candidate["item"].get("search_box")
                    else (weight_regions[0]["box"] if weight_regions else [0, 0, 0, 0])
                ),
                "yolo_label_box": serialize_box(yolo_label_box) if yolo_label_box else "",
                "yolo_weight_screen_box": serialize_box(yolo_weight_box) if yolo_weight_box else "",
                "yolo_used": int(yolo_used),
                "label_ocr_texts": summarize_texts(label_texts),
                "tray_box": serialize_box(tray_info["box"]) if tray_info is not None else "",
                "tray_side_px": round(float(tray_info["tray_side_px"]), 3) if tray_info is not None else "",
                "tray_long_side_px": round(float(tray_info["tray_long_side_px"]), 3) if tray_info is not None else "",
                "tray_short_side_px": round(float(tray_info["tray_short_side_px"]), 3) if tray_info is not None else "",
                "tray_size_mm": round(float(tray_info["tray_size_mm"]), 3) if tray_info is not None else "",
                "mm_per_px": round(float(tray_info["mm_per_px"]), 6) if tray_info is not None else "",
            }
        )

        if save_qc:
            qc_path = os.path.join(qc_dir, f"ocr_{image_file}")
            save_ocr_qc_image(
                image_bgr,
                qc_path,
                split_y,
                plant_name,
                weight_g,
                label_candidate=label_candidate,
                weight_candidate=weight_candidate,
                yolo_label_box=yolo_label_box,
                yolo_weight_box=yolo_weight_box,
                yolo_used=yolo_used,
            )

    df = pd.DataFrame(records)
    df.to_csv(metadata_csv, index=False)
    print(f"\nSaved metadata for {len(records)} images -> {metadata_csv}")
    print("Check QC images in:", qc_dir)
    if yolo_boxes:
        print("YOLO-guided OCR was used (boxes from pre_ocr stage).")
    else:
        print("Split-region OCR was used (no YOLO box JSON available).")

    # --- QC cleanup ---
    _cleanup_qc(config, output_dir)


def _cleanup_qc(config, output_dir):
    import shutil
    vis = config.get('visualization', {})
    if not vis.get('save_ocr_qc', True):
        qc_path = os.path.join(output_dir, config['output'].get('qc_dir', 'qc_visualizations'), 'ocr')
        if os.path.exists(qc_path):
            shutil.rmtree(qc_path)
            print(f'  QC cleanup: removed qc_visualizations/ocr')


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
