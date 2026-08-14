import cv2
import numpy as np

from utils.kernel_id import get_original_image_name  # re-exported for callers


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return [x1, y1, x2, y2]


def ratios_to_box(width, height, ratios):
    x1r, y1r, x2r, y2r = ratios
    return clamp_box([x1r * width, y1r * height, x2r * width, y2r * height], width, height)


def safe_positive_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def convert_px_to_mm(value_px, mm_per_px):
    if mm_per_px is None:
        return ""
    return round(float(value_px) * float(mm_per_px), 4)


def convert_area_px2_to_mm2(value_px2, mm_per_px):
    if mm_per_px is None:
        return ""
    return round(float(value_px2) * float(mm_per_px) * float(mm_per_px), 4)


def build_image_scale_lookup(meta_df):
    lookup = {}
    if meta_df is None:
        return lookup

    for _, row in meta_df.iterrows():
        image_name = str(row.get("image_name", "")).strip()
        if not image_name:
            continue

        mm_per_px = safe_positive_float(row.get("mm_per_px"))
        tray_side_px = safe_positive_float(row.get("tray_side_px"))
        tray_size_mm = safe_positive_float(row.get("tray_size_mm"))
        if mm_per_px is None:
            continue

        lookup[image_name] = {
            "mm_per_px": mm_per_px,
            "tray_side_px": tray_side_px,
            "tray_size_mm": tray_size_mm,
            "tray_source": row.get("tray_source", ""),
        }

    return lookup


def get_tray_color_ranges(calibration_cfg):
    """
    Return HSV ranges used to find the reference tray/frame.

    Backward compatible:
      tray_hsv_lower/tray_hsv_upper still means "blue" when tray_color_ranges
      is not configured.
    """
    ranges = calibration_cfg.get("tray_color_ranges")
    if ranges:
        parsed = []
        for idx, item in enumerate(ranges):
            if not isinstance(item, dict):
                continue
            lower = item.get("hsv_lower", item.get("lower"))
            upper = item.get("hsv_upper", item.get("upper"))
            if lower is None or upper is None:
                continue
            parsed.append(
                {
                    "name": str(item.get("name", f"range_{idx + 1}")),
                    "lower": np.array(lower, dtype=np.uint8),
                    "upper": np.array(upper, dtype=np.uint8),
                }
            )
        if parsed:
            return parsed

    return [
        {
            "name": "blue",
            "lower": np.array(calibration_cfg.get("tray_hsv_lower", [95, 60, 40]), dtype=np.uint8),
            "upper": np.array(calibration_cfg.get("tray_hsv_upper", [140, 255, 255]), dtype=np.uint8),
        }
    ]


def detect_reference_tray(image_bgr, calibration_cfg):
    """
    Detect the colored calibration tray/frame and derive a mm-per-pixel factor.

    The tray is expected to be a large square/near-square object near the center
    of the image. Detection is based on one or more HSV masks plus contour
    scoring. This supports both the original blue frame and green frame images.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    if not bool(calibration_cfg.get("enabled", False)):
        return None

    tray_size_mm = safe_positive_float(calibration_cfg.get("tray_size_mm"))
    if tray_size_mm is None:
        return None

    height, width = image_bgr.shape[:2]
    search_ratios = calibration_cfg.get("tray_search_box_ratios", [0.18, 0.14, 0.82, 0.84])
    search_box = ratios_to_box(width, height, search_ratios)
    x1, y1, x2, y2 = search_box
    search = image_bgr[y1:y2, x1:x2]
    if search.size == 0:
        return None

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    color_ranges = get_tray_color_ranges(calibration_cfg)

    kernel_size = max(3, int(calibration_cfg.get("tray_mask_kernel", 11)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    min_area_ratio = float(calibration_cfg.get("tray_min_area_ratio", 0.03))
    max_area_ratio = float(calibration_cfg.get("tray_max_area_ratio", 0.45))
    max_aspect_ratio = float(calibration_cfg.get("tray_max_aspect_ratio", 1.45))
    min_fill_ratio = float(calibration_cfg.get("tray_min_fill_ratio", 0.45))

    best = None
    best_score = -1e9
    for color_range in color_ranges:
        mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area <= 0:
                continue

            area_ratio = contour_area / float(width * height)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue

            rect = cv2.minAreaRect(contour)
            (cx, cy), (rect_w, rect_h), angle = rect
            rect_w = float(rect_w)
            rect_h = float(rect_h)
            if rect_w < 1 or rect_h < 1:
                continue

            long_side = max(rect_w, rect_h)
            short_side = min(rect_w, rect_h)
            aspect_ratio = long_side / max(short_side, 1e-6)
            if aspect_ratio > max_aspect_ratio:
                continue

            rect_area = max(rect_w * rect_h, 1.0)
            fill_ratio = contour_area / rect_area
            if fill_ratio < min_fill_ratio:
                continue

            cx_full = cx + x1
            cy_full = cy + y1
            center_x_score = 1.0 - abs(cx_full - width / 2.0) / max(width / 2.0, 1.0)
            center_y_score = 1.0 - abs(cy_full - height * 0.52) / max(height * 0.35, 1.0)
            square_score = 1.0 - min(1.0, abs(aspect_ratio - 1.0) / max(max_aspect_ratio - 1.0, 1e-6))
            score = (
                120.0 * area_ratio
                + 2.4 * max(center_x_score, -1.0)
                + 1.8 * max(center_y_score, -1.0)
                + 1.6 * fill_ratio
                + 1.3 * square_score
            )

            if score > best_score:
                bx, by, bw, bh = cv2.boundingRect(contour)
                best_score = score
                best = {
                    "bounding_box": clamp_box([bx + x1, by + y1, bx + x1 + bw, by + y1 + bh], width, height),
                    "long_side_px": long_side,
                    "short_side_px": short_side,
                    "contour_area_px2": float(contour_area),
                    "score": float(score),
                    "angle_deg": float(angle),
                    "color_name": color_range["name"],
                }

    if best is None:
        return None

    side_mode = str(calibration_cfg.get("tray_side_mode", "mean_side")).strip().lower()
    if side_mode == "long_side":
        tray_side_px = best["long_side_px"]
    elif side_mode == "short_side":
        tray_side_px = best["short_side_px"]
    else:
        tray_side_px = 0.5 * (best["long_side_px"] + best["short_side_px"])

    tray_side_px = safe_positive_float(tray_side_px)
    if tray_side_px is None:
        return None

    return {
        "source": "detected",
        "box": best["bounding_box"],
        "tray_side_px": tray_side_px,
        "tray_size_mm": tray_size_mm,
        "mm_per_px": tray_size_mm / tray_side_px,
        "tray_long_side_px": best["long_side_px"],
        "tray_short_side_px": best["short_side_px"],
        "tray_contour_area_px2": best["contour_area_px2"],
        "score": best["score"],
        "angle_deg": best["angle_deg"],
        "color_name": best.get("color_name", ""),
    }


def calibrate_with_fallback(image_bgr, calibration_cfg):
    """Detect tray for mm/px calibration, fall back to fixed value on failure.

    Returns dict with keys: source, tray_side_px, tray_size_mm, mm_per_px,
    tray_long_side_px, tray_short_side_px, warning (optional).
    """
    tray_info = detect_reference_tray(image_bgr, calibration_cfg)
    fixed_mm_per_px = calibration_cfg.get("fixed_mm_per_px")

    if tray_info is not None:
        # Verify against fixed value if available
        if fixed_mm_per_px is not None:
            measured = tray_info["mm_per_px"]
            deviation = abs(measured - fixed_mm_per_px) / max(fixed_mm_per_px, 1e-6)
            if deviation > 0.03:
                tray_info["warning"] = (
                    f"mm_per_px deviation {deviation:.1%}: measured={measured:.5f}, "
                    f"fixed={fixed_mm_per_px:.5f}"
                )
        return tray_info

    # Detection failed — use fixed calibration if available
    if fixed_mm_per_px is not None:
        tray_size_mm = float(calibration_cfg.get("tray_size_mm", 100.0))
        return {
            "source": "fixed",
            "tray_side_px": None,
            "tray_size_mm": tray_size_mm,
            "mm_per_px": float(fixed_mm_per_px),
            "tray_long_side_px": None,
            "tray_short_side_px": None,
            "warning": "tray detection failed, using fixed calibration",
        }

    return None
