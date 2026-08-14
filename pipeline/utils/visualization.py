"""
utils/visualization.py
======================
Shared drawing functions used by multiple pipeline stages.

All functions work on OpenCV images (numpy arrays, BGR color format).
They draw annotations directly on a copy of the image and return it,
or save it to a file — they never call plt.show() so they work on a server.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for server use (no display)
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------------------------
# Core drawing functions
# ---------------------------------------------------------------------------

def draw_milk_yellow_overlay(image, binary_mask, alpha=0.5, color_bgr=(100, 200, 230)):
    """
    Blend a milk-yellow color over the kernel region in the image.

    The color (100, 200, 230) in BGR gives a warm yellow that resembles
    real maize kernel color — much more informative visually than a plain
    green or red overlay.

    Args:
        image      : original BGR image (numpy array)
        binary_mask: single-channel uint8 mask (255=kernel, 0=background)
        alpha      : blending weight — 0.0=fully transparent, 1.0=solid color
        color_bgr  : BGR tuple for the overlay color

    Returns:
        result: BGR image with colored overlay blended in
    """
    result = image.copy()
    # Create a solid color image the same size as the original
    color_layer = np.full_like(image, color_bgr, dtype=np.uint8)
    # Only blend where the mask is white (kernel region)
    mask_bool = binary_mask > 0
    result[mask_bool] = cv2.addWeighted(
        image, 1 - alpha, color_layer, alpha, 0
    )[mask_bool]
    return result


def draw_contour(image, contour_points, color=(0, 255, 0), thickness=2):
    """
    Draw the kernel outline contour on the image.

    Args:
        image         : BGR image to draw on (modified in place)
        contour_points: numpy array of shape (N, 2) with (x, y) coordinates
        color         : BGR color for the contour line
        thickness     : line thickness in pixels

    Returns:
        image with contour drawn
    """
    result = image.copy()
    pts = contour_points.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(result, [pts], isClosed=True, color=color, thickness=thickness)
    return result


def clamp_text_origin(image, text, origin, font_scale=0.5, thickness=1, margin=4):
    """Keep a cv2 text origin inside the visible image area."""
    h, w = image.shape[:2]
    text_size, baseline = cv2.getTextSize(
        str(text),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(font_scale),
        int(thickness),
    )
    text_w, text_h = text_size
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    x = max(int(margin), min(max(int(margin), w - text_w - int(margin)), x))
    y = max(text_h + int(margin), min(max(text_h + int(margin), h - baseline - int(margin)), y))
    return x, y


def draw_bottom_point(image, point, radius=3, color=(0, 0, 255), label='',
                      font_scale=0.38, thickness=1, text_offset=(6, -4)):
    """
    Draw a filled circle at the kernel's bottom tip (base) location.

    The bottom point is the pointy base of the maize kernel —

    Args:
        point: (x, y) coordinates as integers or floats
    """
    result = image.copy()
    cx, cy = int(round(point[0])), int(round(point[1]))
    cv2.circle(result, (cx, cy), max(1, radius + 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(result, (cx, cy), max(1, radius), color, -1, lineType=cv2.LINE_AA)
    if label:
        result = draw_text_with_outline(
            result,
            label,
            (cx + text_offset[0], cy + text_offset[1]),
            color=color,
            font_scale=font_scale,
            thickness=thickness,
            outline_thickness=max(2, thickness + 2),
        )
    return result


def draw_top_point(image, point, radius=3, color=(255, 0, 255), label='',
                   font_scale=0.38, thickness=1, text_offset=(6, -4)):
    """Draw a filled circle at the kernel's top (crown) point."""
    result = image.copy()
    cx, cy = int(round(point[0])), int(round(point[1]))
    cv2.circle(result, (cx, cy), max(1, radius + 1), (255, 255, 255), 1, lineType=cv2.LINE_AA)
    cv2.circle(result, (cx, cy), max(1, radius), color, -1, lineType=cv2.LINE_AA)
    if label:
        result = draw_text_with_outline(
            result,
            label,
            (cx + text_offset[0], cy + text_offset[1]),
            color=color,
            font_scale=font_scale,
            thickness=thickness,
            outline_thickness=max(2, thickness + 2),
        )
    return result


def draw_centroid(image, point, radius=2, color=(0, 255, 255), label='',
                  font_scale=0.36, thickness=1, text_offset=(6, -4)):
    """Draw the centroid estimated from the SAM binary mask."""
    result = image.copy()
    cx, cy = int(round(point[0])), int(round(point[1]))
    cv2.circle(result, (cx, cy), max(1, radius + 1), (0, 0, 0), 1, lineType=cv2.LINE_AA)
    cv2.circle(result, (cx, cy), max(1, radius), color, -1, lineType=cv2.LINE_AA)
    if label:
        result = draw_text_with_outline(
            result,
            label,
            (cx + text_offset[0], cy + text_offset[1]),
            color=color,
            font_scale=font_scale,
            thickness=thickness,
            outline_thickness=max(2, thickness + 2),
        )
    return result


def draw_text_with_outline(image, text, origin, color, font_scale=0.5, thickness=1,
                           outline_color=(0, 0, 0), outline_thickness=3):
    """Draw legible text with a dark outline for busy tray backgrounds."""
    result = image.copy()
    pos = clamp_text_origin(result, text, origin, font_scale=font_scale, thickness=thickness)
    cv2.putText(
        result,
        str(text),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        str(text),
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return result


def draw_axis(image, bottom_point, top_point, color=(128, 0, 128), thickness=1, label=None):
    """Draw the main axis line from bottom to top of the kernel."""
    result = image.copy()
    pt1 = (int(round(bottom_point[0])), int(round(bottom_point[1])))
    pt2 = (int(round(top_point[0])), int(round(top_point[1])))
    cv2.line(result, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)
    if label:
        mid = ((pt1[0] + pt2[0]) // 2 + 6, (pt1[1] + pt2[1]) // 2 - 8)
        result = draw_text_with_outline(
            result,
            label,
            mid,
            color=color,
            font_scale=0.4,
            thickness=1,
            outline_thickness=3,
        )
    return result


def draw_qc_legend(image, entries, origin=(12, 20), font_scale=0.36):
    """
    Draw a compact legend for OpenCV QC overlays.

    entries: list of (label, color_bgr, style), where style is one of
    line, bottom, top, centroid, solid, ring, or box.
    """
    result = image.copy()
    if not entries:
        return result

    x, y = int(origin[0]), int(origin[1])
    line_gap = max(13, int(round(34 * float(font_scale))))
    seen = set()

    for label, color, style in entries:
        if label in seen:
            continue
        seen.add(label)

        marker_center = (x + 8, y - 5)
        color = tuple(int(v) for v in color)
        if style == 'line':
            cv2.line(
                result,
                (x + 1, y - 5),
                (x + 17, y - 5),
                color,
                2,
                lineType=cv2.LINE_AA,
            )
        elif style in {'bottom', 'top', 'centroid'}:
            cv2.circle(result, marker_center, 4, color, -1, lineType=cv2.LINE_AA)
        elif style == 'ring':
            cv2.circle(result, marker_center, 5, color, 1, lineType=cv2.LINE_AA)
        elif style == 'box':
            cv2.rectangle(result, (x + 2, y - 11), (x + 16, y + 1), color, 2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(result, marker_center, 4, color, -1, lineType=cv2.LINE_AA)

        result = draw_text_with_outline(
            result,
            label,
            (x + 24, y),
            color=color,
            font_scale=font_scale,
            thickness=1,
        )
        y += line_gap

    return result


def draw_width_line(image, point_a, point_b, color=(255, 180, 0), thickness=2, label=None,
                    endpoint_radius=2):
    """
    Draw a single width measurement line between two contour intersection points.

    Args:
        point_a, point_b : the two endpoints of the width line (x, y) floats
        label            : optional text label (e.g. 'W50') drawn at midpoint
    """
    result = image.copy()
    if point_a is None or point_b is None:
        return result
    pa = (int(round(point_a[0])), int(round(point_a[1])))
    pb = (int(round(point_b[0])), int(round(point_b[1])))
    cv2.line(result, pa, pb, color, thickness, lineType=cv2.LINE_AA)
    # Small endpoint dots keep the width line readable in per-kernel QC panels.
    cv2.circle(result, pa, max(1, int(endpoint_radius)), color, -1, lineType=cv2.LINE_AA)
    cv2.circle(result, pb, max(1, int(endpoint_radius)), color, -1, lineType=cv2.LINE_AA)
    if label:
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2 - 6)
        result = draw_text_with_outline(
            result,
            label,
            mid,
            color=color,
            font_scale=0.4,
            thickness=1,
            outline_thickness=3,
        )
    return result


def draw_all_measurements(image, bottom, top, w25_pts, w50_pts, w75_pts, mb_pts):
    """
    Draw all measurement lines on the image in one call.

    Args:
        bottom   : (x, y) bottom tip point
        top      : (x, y) top point
        w25_pts  : (point_a, point_b) width line at 25%
        w50_pts  : (point_a, point_b) width line at 50%
        w75_pts  : (point_a, point_b) width line at 75%
        mb_pts   : (point_a, point_b) max width line

    Returns:
        annotated image
    """
    result = image.copy()
    result = draw_axis(result, bottom, top, color=(170, 60, 190), thickness=1, label=None)
    result = draw_bottom_point(result, bottom, radius=2, label='')
    result = draw_top_point(result, top, radius=2, label='')
    # Width lines in blue shades
    result = draw_width_line(result, *w25_pts, color=(255, 100, 0), label=None, thickness=1)
    result = draw_width_line(result, *w50_pts, color=(255, 180, 0), label=None, thickness=1)
    result = draw_width_line(result, *w75_pts, color=(200, 220, 0), label=None, thickness=1)
    # Max width in orange
    result = draw_width_line(result, *mb_pts, color=(0, 165, 255), label=None, thickness=2)
    return result


def draw_axis_points(image, contour, centroid, bottom, top,
                     contour_color=(0, 180, 0), axis_color=(128, 0, 128)):
    """
    Draw the contour, centroid, main axis, and top/bottom points.

    This is the simplified QC view requested for the updated pipeline.
    """
    result = draw_contour(image, contour, color=contour_color, thickness=1)
    result = draw_axis(result, bottom, top, color=axis_color, thickness=1)
    result = draw_bottom_point(result, bottom, radius=2, label='')
    result = draw_top_point(result, top, radius=2, label='')
    result = draw_centroid(result, centroid, radius=1, label='')
    return result


# ---------------------------------------------------------------------------
# Matplotlib-based plots (saved as PNG, not displayed)
# ---------------------------------------------------------------------------

def save_area_chart(contour, bottom_point, top_point, axis_length,
                    w25, w50, w75, max_width,
                    w25_pts, w50_pts, w75_pts, mb_pts,
                    image_name, output_path):
    """
    Save the normalized area chart (kernel width profile) as a PNG.

    The x-axis is kernel length as a percentage (0% = bottom, 100% = top).
    The y-axis is signed perpendicular distance from the axis (left/right of centerline).
    This is the stacked area chart approach from the notebooks.

    Args:
        contour      : (N, 2) array of contour points
        bottom_point : (x, y) bottom tip
        top_point    : (x, y) top
        axis_length  : float, length of the main axis in pixels
        w25..w75     : scalar width values at each fraction
        *_pts        : (point_a, point_b) endpoint pairs for each width line
        image_name   : used in plot title
        output_path  : where to save the PNG
    """
    axis_dir = (top_point - bottom_point) / axis_length
    perp_dir = np.array([-axis_dir[1], axis_dir[0]])  # 90° rotation = perpendicular

    # Project each contour point onto the axis (x) and perpendicular (y)
    positions = []      # fraction along axis (0 to 1)
    signed_dists = []   # signed distance perpendicular to axis

    for pt in contour:
        vec = pt - bottom_point
        pos = np.dot(vec, axis_dir) / axis_length   # 0 = bottom, 1 = top
        dist = np.dot(vec, perp_dir)
        positions.append(pos * 100)   # convert to percentage
        signed_dists.append(dist)

    positions = np.array(positions)
    signed_dists = np.array(signed_dists)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(positions, signed_dists, alpha=0.4, color='goldenrod', label='Kernel profile')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    # Mark bottom and top
    ax.axvline(0,   color='red',    linestyle=':', linewidth=1.2, label='Bottom (0%)')
    ax.axvline(100, color='purple', linestyle=':', linewidth=1.2, label='Top (100%)')

    # Mark width lines
    _draw_width_on_axis(ax, 25,  w25,  w25_pts,  bottom_point, axis_dir, perp_dir, axis_length, 'W25', 'blue')
    _draw_width_on_axis(ax, 50,  w50,  w50_pts,  bottom_point, axis_dir, perp_dir, axis_length, 'W50', 'steelblue')
    _draw_width_on_axis(ax, 75,  w75,  w75_pts,  bottom_point, axis_dir, perp_dir, axis_length, 'W75', 'teal')

    # Max width — find its percentage position
    if mb_pts[0] is not None:
        mb_mid = (np.array(mb_pts[0]) + np.array(mb_pts[1])) / 2
        mb_pos = np.dot(mb_mid - bottom_point, axis_dir) / axis_length * 100
        ax.axvline(mb_pos, color='orange', linewidth=2, label=f'Max Width ({max_width:.1f}px)')

    ax.set_title(f'Kernel Width Profile: {image_name}')
    ax.set_xlabel('Kernel Length (%)')
    ax.set_ylabel('Width (pixels, signed)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def _draw_width_on_axis(ax, pct, width_val, pts, bottom, axis_dir, perp_dir, axis_len, label, color):
    """Helper: draw a vertical width marker on the area chart axes."""
    if pts[0] is None or pts[1] is None:
        return
    p1_perp = np.dot(np.array(pts[0]) - bottom, perp_dir)
    p2_perp = np.dot(np.array(pts[1]) - bottom, perp_dir)
    ax.plot([pct, pct], [p1_perp, p2_perp], color=color,
            linestyle='--', linewidth=1.5, label=f'{label} = {width_val:.1f}px')


def save_circle_deviation_plot(contour, centroid, bottom_point, top_point,
                                image_name, output_path):
    """
    Save the circular deviation plot for one kernel.

    Shows:
    - Left panel: kernel outline with enclosing circle
    - Right panel: polar plot of how much the outline deviates from the circle

    The contour is normalized before computing deviations so that kernels
    of different sizes can be compared on the same scale.

    Args:
        contour      : (N, 2) contour points
        centroid     : (x, y) centroid of contour
        bottom_point : (x, y) used to set the angular reference (0°)
        top_point    : (x, y) used for orientation
        image_name   : used in plot title
        output_path  : where to save the PNG
    """
    # Translate so centroid is at origin
    centered = contour - centroid

    # Normalize: scale so the mean radial distance is 0.5
    radial_dists = np.linalg.norm(centered, axis=1)
    scale = 0.5 / np.mean(radial_dists)
    normalized = centered * scale

    # Compute polar angles and radial distances after normalization
    thetas = np.arctan2(normalized[:, 1], normalized[:, 0])
    radii = np.linalg.norm(normalized, axis=1)
    deviations = radii - 0.5   # deviation from the reference circle of radius 0.5

    # Sort by angle for a smooth polar plot
    sort_idx = np.argsort(thetas)
    thetas_sorted = thetas[sort_idx]
    devs_sorted = deviations[sort_idx]

    fig = plt.figure(figsize=(11.2, 5.2), facecolor='#f7f7f4')
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1, 1.05],
        wspace=0.32,
        top=0.74,
        bottom=0.08,
    )

    # --- Left panel: kernel outline + reference circle ---
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_facecolor('#fcfbf7')
    ax1.plot(normalized[:, 0], normalized[:, 1], color='#d8a03d', linewidth=2, label='Kernel outline')
    ax1.fill(normalized[:, 0], normalized[:, 1], alpha=0.32, color='#d8a03d')
    circle = plt.Circle((0, 0), 0.5, color='#3c78a8', fill=False,
                         linestyle='--', linewidth=1.4, label='Reference circle')
    ax1.add_artist(circle)
    ax1.scatter(0, 0, color='#c23b22', s=18, zorder=5, label='Centroid')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.24)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title(f'Outline vs Circle\n{image_name}', pad=8)

    # --- Right panel: polar deviation plot ---
    ax2 = fig.add_subplot(grid[0, 1], polar=True)
    ax2.set_facecolor('#fcfbf7')
    ax2.plot(thetas_sorted, devs_sorted, color='#1f7a6d', linewidth=1.4, label='Deviation')
    ax2.fill(thetas_sorted, np.maximum(devs_sorted, 0), alpha=0.28, color='#1f7a6d')
    ax2.fill(thetas_sorted, np.minimum(devs_sorted, 0), alpha=0.24, color='#c23b22')
    ax2.axhline(0, color='#333333', linestyle='--', linewidth=0.8)
    ax2.set_title('Radial Deviation\noutside vs inside', pad=14)

    fig.suptitle(f'Circular Deviation - {image_name}', fontsize=12, y=0.97)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# File I/O helper
# ---------------------------------------------------------------------------

def save_image(image, output_path):
    """Save a BGR numpy array as an image file. Creates parent directories."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
