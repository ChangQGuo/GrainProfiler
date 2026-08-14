"""
processing/contour_extraction.py
================================
Compatibility contour helpers used by downstream measurement code.

The active pipeline no longer runs a standalone contour stage, but
`measurements/kernel_metrics.py` still imports `extract_largest_contour` from
this module. Keeping the helper here ensures fresh environments work without
depending on stale `__pycache__` files.
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import load_config


def extract_largest_contour(binary_mask, min_area=500):
    """
    Return the largest external contour in a binary kernel mask.

    Args:
        binary_mask: uint8 mask with foreground=255, background=0
        min_area: minimum contour area in pixels

    Returns:
        Nx2 contour array or None when no valid contour is found.
    """
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None

    valid = [c for c in contours if cv2.contourArea(c) >= float(min_area)]
    if not valid:
        return None

    largest = max(valid, key=cv2.contourArea)
    return largest.reshape(-1, 2)


def save_contour_qc(subimage_path, contour, qc_path):
    image = cv2.imread(subimage_path)
    if image is None:
        return

    result = image.copy()
    pts = contour.astype("int32").reshape((-1, 1, 2))
    cv2.polylines(result, [pts], isClosed=True, color=(0, 220, 0), thickness=2)

    start = (int(contour[0, 0]), int(contour[0, 1]))
    cv2.circle(result, start, 4, (0, 0, 255), -1)

    os.makedirs(os.path.dirname(qc_path), exist_ok=True)
    cv2.imwrite(qc_path, result)


def run(config_path):
    config = load_config(config_path)

    output_dir = config["output"]["base_dir"]
    binary_dir = os.path.join(output_dir, config["output"]["masks_binary_dir"])
    subimages_dir = os.path.join(output_dir, config["output"]["subimages_dir"])
    contour_file = os.path.join(
        output_dir,
        config["output"].get("contour_data_file", "contour_data.txt"),
    )
    qc_dir = os.path.join(output_dir, config["output"]["qc_dir"], "contours")
    save_qc = bool(config.get("visualization", {}).get("save_qc", True))
    min_area = int(config.get("contour_filtering", {}).get("min_contour_area", 500))

    os.makedirs(qc_dir, exist_ok=True)

    if not os.path.isdir(binary_dir):
        print(f"ERROR: Binary masks directory not found: {binary_dir}")
        sys.exit(1)

    mask_files = sorted(
        f
        for f in os.listdir(binary_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not mask_files:
        print(f"ERROR: No mask files found in {binary_dir}")
        sys.exit(1)

    print(f"Extracting contours from {len(mask_files)} binary masks...")

    output_lines = []
    skipped = 0
    for mask_file in mask_files:
        mask_path = os.path.join(binary_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: Could not read {mask_path}")
            skipped += 1
            continue

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contour = extract_largest_contour(mask, min_area=min_area)
        if contour is None:
            print(f"  WARNING: No valid contour found in {mask_file} - skipping")
            skipped += 1
            continue

        output_lines.append(f"Subimage: {mask_file}")
        output_lines.append(f"Contour Points: {contour.flatten().tolist()}")

        if save_qc:
            subimage_path = os.path.join(subimages_dir, mask_file)
            qc_path = os.path.join(qc_dir, f"contour_{mask_file}")
            save_contour_qc(subimage_path, contour, qc_path)

    with open(contour_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + ("\n" if output_lines else ""))

    print(f"\nContours extracted for {(len(output_lines) // 2)} kernels -> {contour_file}")
    if skipped > 0:
        print(f"WARNING: {skipped} kernels skipped (no valid contour found).")
    print("Check QC images in:", qc_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run(config_path)
