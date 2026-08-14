"""OpenCV <-> Qt image conversion utilities."""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def cv2_to_qpixmap(cv_image: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR (H, W, 3) uint8 array to a QPixmap."""
    if cv_image is None or cv_image.size == 0:
        return QPixmap()
    h, w = cv_image.shape[:2]
    if cv_image.ndim == 2:
        # Grayscale
        bytes_per_line = w
        qimg = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    bytes_per_line = 3 * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def load_image(path: str) -> np.ndarray | None:
    """cv2.imread with error tolerance.  Returns None on failure."""
    if not Path(path).exists():
        return None
    img = cv2.imread(path)
    if img is None or img.size == 0:
        return None
    return img


def crop_to_tray(image: np.ndarray, tray_box_str: str) -> np.ndarray:
    """Crop image to the tray bounding box.

    tray_box_str is "x1,y1,x2,y2".
    Returns the cropped region (or the full image if parsing fails).
    """
    try:
        parts = [int(x.strip()) for x in tray_box_str.split(",")]
        x1, y1, x2, y2 = parts
        h, w = image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2].copy()
    except (ValueError, IndexError):
        pass
    return image
