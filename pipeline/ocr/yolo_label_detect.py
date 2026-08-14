"""
pipeline/ocr/yolo_label_detect.py
=========================
Pre-OCR step — runs TWO YOLO models on tray images:

  1. YOLO11   — detects label & weight_screen ROIs
  2. YOLOv8n   — detects individual seven-segment digits (0-9) on the weight_screen crop

Outputs:
  - yolo_label_weight_boxes.json   — label boxes + weight_screen boxes + weight_g
  - weight_digit_result/           — annotated digit-detection QC images per image

RUN WITH:
          conda activate yoloenv
          cd /path/to/pipeline/ocr/
          python yolo_label_detect.py ../config.yaml
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is not installed in this environment.")
    print("Run this script with: conda activate yoloenv")
    sys.exit(1)


from utils.config import load_config


def deduplicate_overlapping_digits(detections, overlap_threshold=0.80):
    """Remove duplicate digit detections where one box heavily overlaps another.

    When two boxes overlap by more than ``overlap_threshold`` (80%) of the
    *smaller* box's area, keep only the one with higher confidence.
    This handles the case where YOLOv8 detects the same digit twice.
    """
    if len(detections) <= 1:
        return detections

    n = len(detections)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        x1_i, y1_i = detections[i]["x1"], detections[i].get("y1", 0)
        x2_i = x1_i + detections[i].get("w", 10)
        y2_i = y1_i + detections[i].get("h", 10)
        area_i = max(1.0, (x2_i - x1_i) * (y2_i - y1_i))

        for j in range(i + 1, n):
            if not keep[j]:
                continue
            x1_j, y1_j = detections[j]["x1"], detections[j].get("y1", 0)
            x2_j = x1_j + detections[j].get("w", 10)
            y2_j = y1_j + detections[j].get("h", 10)
            area_j = max(1.0, (x2_j - x1_j) * (y2_j - y1_j))

            # Intersection area
            ix1 = max(x1_i, x1_j); iy1 = max(y1_i, y1_j)
            ix2 = min(x2_i, x2_j); iy2 = min(y2_i, y2_j)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)

            # Overlap as fraction of the smaller box
            overlap = inter / min(area_i, area_j)
            if overlap > overlap_threshold:
                # Keep the higher-confidence detection
                if detections[i]["conf"] >= detections[j]["conf"]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i eliminated, move to next outer loop iteration

    return [d for d, k in zip(detections, keep) if k]


def read_weight_from_detections(detections, class_names):
    """Sort seven-segment digit detections left→right, assemble weight string with auto decimal.

    Args:
        detections: list of {"label": str, "x1": float, "conf": float}
        class_names: dict {cls_id: cls_name} from the seven-segment digit model

    Returns:
        weight_str: e.g. "12.54" or None
    """
    # Keep only digit detections (0-9), sort by x1
    digits = [d for d in detections if str(d["label"]).isdigit()]
    if not digits:
        return None

    digits.sort(key=lambda d: d["x1"])
    digit_chars = [str(d["label"]) for d in digits]

    # Scale convention: last 2 digits are always decimals ("1254" → "12.54").
    # Always insert the decimal, even for short reads (scales often suppress
    # leading zeros, e.g. "0.54" reads as "54", "0.05" reads as "5").
    if len(digit_chars) == 1:
        weight_str = f"0.0{digit_chars[0]}"
    else:
        weight_str = "".join(digit_chars[:-2]) + "." + "".join(digit_chars[-2:])

    try:
        float(weight_str)
        return weight_str
    except ValueError:
        return None


def run(config_path):
    config = load_config(config_path)

    image_dir = config["input"]["image_dir"]
    ext = config["input"]["image_extension"]
    output_dir = config["output"]["base_dir"]

    # Output paths
    output_json = os.path.join(
        output_dir,
        config["output"].get("yolo_label_weight_json", "yolo_label_weight_boxes.json"),
    )
    weight_digit_result_dir = os.path.join(
        output_dir,
        config["output"].get("weight_digit_result_dir", "qc_visualizations/weight_digit_result/"),
    )
    save_qc = bool(config.get('visualization', {}).get('save_qc', True))

    # YOLO11 — label / weight_screen ROI detector
    yolo11_path = config.get("models", {}).get("label_weight_yolo", "")
    lw_cfg = config.get("label_weight_detection", {})

    # YOLOv8 — digit detector (on weight_screen crops)
    yolo8_path = config.get("models", {}).get("weight_digit_yolo", "")
    wd_cfg = config.get("weight_digit_detection", {})

    # --- Resolve device ---
    runtime_device = config.get("runtime", {}).get("gpu_device", "cuda:0")
    device_str = str(runtime_device).strip()
    print(f"Device: {device_str}")

    # --- Validate YOLO11 ---
    if not yolo11_path or not os.path.exists(yolo11_path):
        print(f"ERROR: YOLO11 label/weight model not found at {yolo11_path}")
        sys.exit(1)

    conf_stage1 = float(lw_cfg.get("confidence", 0.25))
    label_class = str(lw_cfg.get("label_class", "label"))
    weight_class = str(lw_cfg.get("weight_class", "weight_screen"))

    print(f"Loading YOLO11 label/weight detector: {yolo11_path}")
    model_stage1 = YOLO(yolo11_path)

    # --- Validate / load YOLOv8 ---
    if yolo8_path and os.path.exists(yolo8_path):
        conf_stage2 = float(wd_cfg.get("confidence", 0.30))
        print(f"Loading YOLOv8 digit detector: {yolo8_path}")
        model_stage2 = YOLO(yolo8_path)
    else:
        print(f"WARNING: YOLOv8 digit model not found at {yolo8_path}")
        print("  Weight digit detection will be SKIPPED — weight_g = UNKNOWN")
        model_stage2 = None

    # --- Image list ---
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    if not image_files:
        print(f"ERROR: No {ext} images found in {image_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(weight_digit_result_dir, exist_ok=True)  # needed for intermediate crops (not just QC)

    print(f"\nProcessing {len(image_files)} images...")
    print(f"  Label class: '{label_class}'  Weight class: '{weight_class}'  Conf: {conf_stage1}")
    print(f"  Digit detection: {'YOLOv8' if model_stage2 else 'DISABLED'}")
    print()

    results_list = []
    fail_count = 0
    for idx, image_file in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, image_file)
        stem = os.path.splitext(image_file)[0]
        img = cv2.imread(image_path)

        if img is None:
            print(f"  WARNING: Cannot read {image_file}, skipping")
            results_list.append({
                "image_name": image_file,
                "label_box": None,
                "weight_screen_box": None,
                "weight_g": None,
                "error": "image_read_failed",
            })
            continue

        h, w = img.shape[:2]

        # ==================================================================
        # Step 1 — YOLO11: detect label + weight_screen
        # ==================================================================
        try:
            results_s1 = model_stage1(img, conf=conf_stage1, device=device_str,
                                      verbose=False)
        except Exception as e:
            print(f"  WARNING: {image_file} label/weight detection failed: {e}")
            results_list.append({
                "image_name": image_file,
                "label_box": None,
                "weight_screen_box": None,
                "weight_g": None,
                "error": "detection_failed",
            })
            continue

        best_label = None
        best_weight_roi = None

        for r in results_s1:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_name = r.names.get(int(box.cls[0]), "")
                box_conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                bx1, by1, bx2, by2 = [int(round(v)) for v in xyxy]

                box_info = {
                    "box": [bx1, by1, bx2, by2],
                    "confidence": round(box_conf, 4),
                }

                if cls_name == label_class:
                    if best_label is None or box_conf > best_label["confidence"]:
                        best_label = box_info
                elif cls_name == weight_class:
                    if best_weight_roi is None or box_conf > best_weight_roi["confidence"]:
                        best_weight_roi = box_info

        # ==================================================================
        # Step 2 — Crop weight_screen + YOLOv8 digit detection
        # ==================================================================
        weight_g = None

        if best_weight_roi is not None and model_stage2 is not None:
            sx1, sy1, sx2, sy2 = best_weight_roi["box"]
            sx1 = max(0, sx1); sy1 = max(0, sy1)
            sx2 = min(w, sx2); sy2 = min(h, sy2)

            screen_crop = img[sy1:sy2, sx1:sx2]
            if screen_crop.size > 0:
                # Letterbox into YOLOv8 training canvas (512h × 1280w), preserving
                # the ROI aspect ratio so digits are not stretched/distorted.
                sc_h, sc_w = screen_crop.shape[:2]
                scale = min(1280.0 / max(sc_w, 1), 512.0 / max(sc_h, 1))
                new_w = max(1, int(round(sc_w * scale)))
                new_h = max(1, int(round(sc_h * scale)))
                resized = cv2.resize(
                    screen_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                screen_crop_resized = np.full((512, 1280, 3), 128, dtype=np.uint8)
                x_off = (1280 - new_w) // 2
                y_off = (512 - new_h) // 2
                screen_crop_resized[y_off:y_off + new_h, x_off:x_off + new_w] = resized

                # Save intermediate crop so YOLOv8 processes the ROI, not the original image
                temp_crop_path = os.path.join(
                    weight_digit_result_dir, f"{stem}_weight_screen_crop.png"
                )
                try:
                    cv2.imwrite(temp_crop_path, screen_crop_resized)
                    # YOLOv8 digit detection on the saved crop file
                    results_s2 = model_stage2(
                        temp_crop_path, conf=conf_stage2,
                        device=device_str, verbose=False,
                    )
                except Exception as e:
                    print(f"  WARNING: {image_file} digit detection failed: {e}")
                    results_s2 = None
                finally:
                    # Always clean up the temp crop, even if inference raises
                    try:
                        os.remove(temp_crop_path)
                    except OSError:
                        pass

                digit_dets = []
                for r in (results_s2 or []):
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model_stage2.names.get(cls_id, str(cls_id))
                        xyxy = box.xyxy[0].cpu().numpy()
                        digit_dets.append({
                            "label": cls_name,
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "w":  float(xyxy[2] - xyxy[0]),
                            "h":  float(xyxy[3] - xyxy[1]),
                            "conf": float(box.conf[0]),
                        })

                # Remove overlapping duplicate detections
                digit_dets = deduplicate_overlapping_digits(digit_dets)

                weight_g = read_weight_from_detections(digit_dets, model_stage2.names)

                # Draw digit detection visualization on the weight_screen crop
                color_map = {
                    0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0),
                    3: (0, 255, 255), 4: (255, 0, 255), 5: (255, 255, 0),
                    6: (0, 165, 255), 7: (203, 192, 255), 8: (50, 255, 50),
                    9: (255, 255, 255),
                }
                digit_viz = screen_crop_resized.copy()
                for r in (results_s2 or []):
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model_stage2.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].cpu().numpy()
                        dx1, dy1, dx2, dy2 = [int(round(v)) for v in xyxy]
                        color = color_map.get(cls_id, (255, 255, 255))
                        cv2.rectangle(digit_viz, (dx1, dy1), (dx2, dy2), color, 2)
                        label = f"{cls_name} ({conf:.2f})"
                        cv2.putText(digit_viz, label, (dx1, max(dy1 - 5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if save_qc:  # master switch gates the QC image (intermediate crop is transient)
                    digit_result_path = os.path.join(
                        weight_digit_result_dir, f"{stem}_weight_digit.jpg"
                    )
                    cv2.imwrite(digit_result_path, digit_viz)
        entry = {
            "image_name": image_file,
            "label_box": best_label,
            "weight_screen_box": best_weight_roi,
            "weight_g": weight_g,
        }
        results_list.append(entry)

        wg_s = f"{weight_g}g" if weight_g else "FAIL"
        if weight_g is None:
            fail_count += 1
            print(f"  [{idx}/{len(image_files)}] {image_file}  →  {wg_s}")

        # Progress tick every 50 images
        if idx % 50 == 0:
            print(f"  ... {idx}/{len(image_files)}  ({fail_count} failed so far)")

    # --- Save JSON ---
    with open(output_json, "w") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    found_label = sum(1 for r in results_list if r["label_box"])
    found_ws = sum(1 for r in results_list if r["weight_screen_box"])
    found_wg = sum(1 for r in results_list if r["weight_g"])

    print(f"\n{'='*60}")
    print(f"Results saved -> {output_json}")
    print(f"  Label boxes:      {found_label}/{len(results_list)}")
    print(f"  Weight-screen:    {found_ws}/{len(results_list)}")
    print(f"  Weight values:    {found_wg}/{len(results_list)}")
    print(f"  Digit QC images:  {weight_digit_result_dir}/")
    print(f"{'='*60}")

    # --- QC cleanup ---
    _cleanup_qc(config, output_dir)


def _cleanup_qc(config, output_dir):
    import shutil
    vis = config.get('visualization', {})
    if not vis.get('save_weight_digit_qc', True):
        qc_dir = os.path.join(output_dir, config['output'].get('weight_digit_result_dir', 'qc_visualizations/weight_digit_result'))
        if os.path.exists(qc_dir):
            shutil.rmtree(qc_dir)
            print(f'  Cleaned: qc_visualizations/weight_digit_result')


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
