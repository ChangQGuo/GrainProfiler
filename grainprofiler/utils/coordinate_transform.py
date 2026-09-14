"""Subimage <-> full-image coordinate transforms.

Replicates the logic from pipeline/measurements/kernel_metrics.py:3768
(subimage_point_to_full_image) so the GUI has zero pipeline dependencies.
"""

from grainprofiler.app.settings import PADDING_PX


def subimage_point_to_full(
    local_x: float,
    local_y: float,
    yolo_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    """Convert a point from subimage-local to full-image coordinates.

    The subimage is the YOLO bounding box expanded by PADDING_PX on all sides
    before being fed into SAM2.  Contour / axis points are returned in the
    coordinate system of that padded crop.

    Args:
        local_x, local_y: point in subimage space.
        yolo_box: [x1, y1, x2, y2] in full-image space.
        image_shape: (height, width) of the full tray photo.

    Returns:
        (full_x, full_y) clamped to [0, width-1] / [0, height-1].
    """
    x1, y1, _x2, _y2 = (int(v) for v in yolo_box)
    height, width = image_shape

    x1p = max(0, x1 - PADDING_PX)
    y1p = max(0, y1 - PADDING_PX)

    full_x = int(round(x1p + float(local_x)))
    full_y = int(round(y1p + float(local_y)))

    full_x = max(0, min(width - 1, full_x))
    full_y = max(0, min(height - 1, full_y))
    return full_x, full_y


def transform_contour(
    contour: list[list[float]],
    yolo_box: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    """Transform an entire contour from subimage to full-image coords."""
    return [
        subimage_point_to_full(pt[0], pt[1], yolo_box, image_shape)
        for pt in contour
    ]


def overlay_scale(image_shape: tuple[int, int]) -> tuple[int, int, float]:
    """Adaptive marker sizes based on image dimension.

    Replicates overlay_scale() from kernel_metrics.py:3781.
    """
    min_dim = max(1, min(image_shape[:2]))
    point_radius = max(2, min(3, int(round(min_dim / 1000.0))))
    line_thickness = max(1, min(2, int(round(min_dim / 1400.0))))
    font_scale = max(0.34, min(0.46, min_dim / 3600.0))
    return point_radius, line_thickness, font_scale
