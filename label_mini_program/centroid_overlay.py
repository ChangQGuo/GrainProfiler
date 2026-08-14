"""
Draw mask-moment centroids back onto RGB kernel crops.

Usage:
  python centroid_overlay.py --images path/to/subimages --masks path/to/masks_binary \
      --out-csv centroids.csv --overlay-dir centroid_overlay
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(image_dir):
    return sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def find_matching_file(folder, name):
    folder = Path(folder)
    exact = folder / name
    if exact.exists():
        return exact
    stem = Path(name).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_mask(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return keep_largest_component(mask)


def keep_largest_component(mask):
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def centroid_from_mask(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments.get("m00", 0.0)) < 1e-9:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def draw_overlay(image, mask, center):
    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(result, contours, -1, (0, 255, 255), 1, cv2.LINE_AA)
    cx, cy = center
    cv2.circle(result, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(result, (int(round(cx)), int(round(cy))), 9, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(
        result,
        f"center=({cx:.1f},{cy:.1f})",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Draw moments centroids onto kernel crops.")
    parser.add_argument("--images", required=True, help="RGB kernel crop folder.")
    parser.add_argument("--masks", required=True, help="Binary mask folder with matching filenames.")
    parser.add_argument("--out-csv", required=True, help="Output centroid CSV.")
    parser.add_argument("--overlay-dir", required=True, help="Folder for centroid overlay images.")
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    overlay_dir = Path(args.overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in iter_images(image_dir):
        image = cv2.imread(str(image_path))
        mask_path = find_matching_file(mask_dir, image_path.name)
        status = "ok"
        cx = cy = ""

        if image is None:
            status = "image_read_failed"
        elif mask_path is None:
            status = "mask_missing"
        else:
            mask = read_mask(mask_path)
            if mask is None:
                status = "mask_read_or_component_failed"
            else:
                center = centroid_from_mask(mask)
                if center is None:
                    status = "moments_failed"
                else:
                    cx, cy = center
                    if image.shape[:2] != mask.shape[:2]:
                        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
                    overlay = draw_overlay(image, mask, center)
                    cv2.imwrite(str(overlay_dir / image_path.name), overlay)

        rows.append(
            {
                "image_path": str(image_path),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "status": status,
                "center_x": f"{cx:.8f}" if isinstance(cx, float) else "",
                "center_y": f"{cy:.8f}" if isinstance(cy, float) else "",
            }
        )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "status", "center_x", "center_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved CSV: {args.out_csv}")
    print(f"saved overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
