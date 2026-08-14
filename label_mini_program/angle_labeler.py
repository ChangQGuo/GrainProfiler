"""
Semi-automatic local labeler for kernel main-axis angles.

The primary label is a directed axis encoded as cos(theta), sin(theta), so it
matches the red arrow shown in the UI. The saved endpoints are computed from
the current angle, the mask centroid, and intersections with the original mask
outline. cos(2 theta), sin(2 theta) are still saved as auxiliary undirected
axis labels.
"""

import argparse
import csv
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LEFT_KEYS = {81, 2424832}
RIGHT_KEYS = {83, 2555904}
UP_KEYS = {82, 2490368}
DOWN_KEYS = {84, 2621440}
DELETE_KEYS = {8, 127, 3014656}
HUD_HEIGHT = 56
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_FONT_SCALE = 0.42
HUD_THICKNESS = 1
DEFAULT_DISPLAY_SIZE = 640
CSV_FIELDS = [
    "image_label",
    "status",
    "theta_deg",
    "theta_rad",
    "cos_theta",
    "sin_theta",
    "cos2theta",
    "sin2theta",
    "center_x",
    "center_y",
    "axis_x1",
    "axis_y1",
    "axis_x2",
    "axis_y2",
]
LEGACY_CSV_FIELDS = [
    "image_path",
    "mask_path",
    "status",
    "theta_deg",
    "theta_rad",
    "cos_theta",
    "sin_theta",
    "cos2theta",
    "sin2theta",
    "center_x",
    "center_y",
    "axis_x1",
    "axis_y1",
    "axis_x2",
    "axis_y2",
    "label_source",
]


def normalize_theta(theta):
    theta = float(theta) % (2.0 * math.pi)
    if theta < 0:
        theta += 2.0 * math.pi
    return theta


def iter_images(image_dir):
    return sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES])


def find_matching_file(folder, image_name):
    folder = Path(folder)
    exact = folder / image_name
    if exact.exists():
        return exact
    stem = Path(image_name).stem
    for suffix in IMAGE_SUFFIXES:
        candidate = folder / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def keep_largest_component(mask):
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def read_mask(mask_path, target_shape=None):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    if target_shape is not None and mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return keep_largest_component(mask)


def fallback_mask_from_black_background(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Black-background crops have almost-zero background. Keep non-black pixels.
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    return keep_largest_component(mask)


def load_main_mask(mask_dir, image_path, image_bgr):
    mask_path = find_matching_file(mask_dir, image_path.name)
    if mask_path is not None:
        mask = read_mask(mask_path, target_shape=image_bgr.shape[:2])
        if mask is not None:
            return mask, mask_path, "mask_file"

    mask = fallback_mask_from_black_background(image_bgr)
    if mask is not None:
        return mask, None, "fallback_black_background_threshold"
    return None, mask_path, "mask_missing_or_invalid"


def centroid_from_mask(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments.get("m00", 0.0)) < 1e-9:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=float)


def initial_theta_from_moments(mask):
    moments = cv2.moments(mask, binaryImage=True)
    if abs(moments.get("m00", 0.0)) < 1e-9:
        return 0.0
    angle = 0.5 * math.atan2(2.0 * moments["mu11"], moments["mu20"] - moments["mu02"])
    return normalize_theta(angle)


def choose_initial_theta(mask, mode):
    mode = str(mode or "moments").strip().lower()
    if mode == "random":
        return random.random() * 2.0 * math.pi
    if mode == "horizontal":
        return 0.0
    if mode == "vertical":
        return math.pi / 2.0
    return initial_theta_from_moments(mask)


def extract_external_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(float)


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def line_contour_intersections(contour, origin, direction, projection_tol=1.0):
    if contour is None or len(contour) < 2:
        return []

    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-9:
        return []
    direction = direction / direction_norm

    pts = np.asarray(contour, dtype=float).reshape(-1, 2)
    closed = np.vstack([pts, pts[0]])
    candidates = []

    for p, q in zip(closed[:-1], closed[1:]):
        edge = q - p
        denom = cross2(direction, edge)
        offset = p - origin

        if abs(denom) < 1e-9:
            if abs(cross2(offset, direction)) <= 0.75:
                candidates.append((float(np.dot(p - origin, direction)), p.copy()))
                candidates.append((float(np.dot(q - origin, direction)), q.copy()))
            continue

        t = cross2(offset, edge) / denom
        u = cross2(offset, direction) / denom
        if -1e-7 <= u <= 1.0 + 1e-7:
            point = origin + t * direction
            candidates.append((float(t), point))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0])
    merged = []
    for proj, point in candidates:
        if merged and abs(proj - merged[-1][0]) <= projection_tol:
            old_proj, old_point, count = merged[-1]
            new_count = count + 1
            merged[-1] = (
                (old_proj * count + proj) / new_count,
                (old_point * count + point) / new_count,
                new_count,
            )
        else:
            merged.append((proj, point, 1))

    return [point for _, point, _ in merged]


def axis_endpoints(contour, center, theta):
    direction = np.array([math.cos(theta), math.sin(theta)], dtype=float)
    intersections = line_contour_intersections(contour, center, direction)
    if len(intersections) >= 2:
        ordered = sorted(intersections, key=lambda pt: float(np.dot(pt - center, direction)))
        return ordered[0], ordered[-1]

    if contour is None or len(contour) < 2:
        length = 200.0
        return center - length * direction, center + length * direction

    projections = np.dot(contour - center, direction)
    return contour[int(np.argmin(projections))], contour[int(np.argmax(projections))]


def square_display_transform(image_shape, display_size):
    height, width = image_shape[:2]
    side = max(int(height), int(width), 1)
    pad_top = (side - int(height)) // 2
    pad_bottom = side - int(height) - pad_top
    pad_left = (side - int(width)) // 2
    pad_right = side - int(width) - pad_left
    scale = float(display_size) / float(side)
    return pad_top, pad_bottom, pad_left, pad_right, scale


def square_resize_for_display(image, display_size, color=(0, 0, 0)):
    display_size = max(1, int(display_size))
    pad_top, pad_bottom, pad_left, pad_right, _ = square_display_transform(image.shape, display_size)
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    if padded.shape[0] == display_size and padded.shape[1] == display_size:
        return padded
    return cv2.resize(padded, (display_size, display_size), interpolation=cv2.INTER_AREA)


def display_to_image_coords(x, y, image_shape, display_size):
    pad_top, _, pad_left, _, scale = square_display_transform(image_shape, display_size)
    if scale <= 0:
        return float(x), float(y)
    return float(x) / scale - float(pad_left), float(y) / scale - float(pad_top)


def fit_text_to_width(text, max_width, font, scale, thickness):
    text = str(text)
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text

    ellipsis = "..."
    if cv2.getTextSize(ellipsis, font, scale, thickness)[0][0] > max_width:
        return ""

    best = ellipsis
    low, high = 0, max(0, len(text) - len(ellipsis))
    while low <= high:
        keep = (low + high) // 2
        left = (keep + 1) // 2
        right = keep // 2
        candidate = text[:left] + ellipsis + (text[-right:] if right else "")
        if cv2.getTextSize(candidate, font, scale, thickness)[0][0] <= max_width:
            best = candidate
            low = keep + 1
        else:
            high = keep - 1
    return best


def draw_overlay(image_bgr, mask, contour, center, theta, sample_text, status, display_size=DEFAULT_DISPLAY_SIZE):
    display = image_bgr.copy()

    if contour is not None and len(contour) > 1:
        cv2.drawContours(display, [np.round(contour).astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 255), 1, cv2.LINE_AA)

    p1, p2 = axis_endpoints(contour, center, theta)
    p1_i = tuple(np.round(p1).astype(int))
    p2_i = tuple(np.round(p2).astype(int))
    center_i = tuple(np.round(center).astype(int))

    # The arrow points along the current positive theta direction. The primary
    # training label uses cos(theta), sin(theta), so the arrow direction matters.
    cv2.arrowedLine(display, p1_i, p2_i, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.08)
    cv2.circle(display, center_i, 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(display, center_i, 10, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(display, p1_i, 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(display, p2_i, 5, (0, 255, 0), -1, cv2.LINE_AA)

    display = square_resize_for_display(display, display_size, color=(0, 0, 0))

    canvas = cv2.copyMakeBorder(
        display,
        HUD_HEIGHT,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )

    theta_deg = math.degrees(theta)
    max_text_width = max(20, canvas.shape[1] - 16)
    lines = [
        f"{sample_text}  status={status}",
        f"theta={theta_deg:.2f} deg  cos={math.cos(theta):.5f}  sin={math.sin(theta):.5f}",
    ]
    y = 20
    for text in lines:
        text = fit_text_to_width(text, max_text_width, HUD_FONT, HUD_FONT_SCALE, HUD_THICKNESS)
        cv2.putText(canvas, text, (8, y), HUD_FONT, HUD_FONT_SCALE, (245, 245, 245), HUD_THICKNESS, cv2.LINE_AA)
        y += 22

    return canvas, p1, p2


def image_label_from_path(image_path):
    return Path(str(image_path).replace("\\", "/")).name


def read_done_image_paths(csv_path):
    done = set()
    if not Path(csv_path).exists():
        return done
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_label = row.get("image_label") or image_label_from_path(row.get("image_path", ""))
            if image_label:
                done.add(image_label)
    return done


def ensure_csv_header(csv_path):
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header not in (CSV_FIELDS, LEGACY_CSV_FIELDS):
            raise SystemExit(
                "Existing CSV header does not match a supported directed-angle format. "
                "Use --no-resume, or choose a new --out CSV path."
            )


def csv_uses_legacy_header(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return False
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader, []) == LEGACY_CSV_FIELDS


def load_existing_rows(csv_path):
    rows = {}
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return rows
    ensure_csv_header(csv_path)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_label = row.get("image_label") or image_label_from_path(row.get("image_path", ""))
            if image_label:
                converted = {field: row.get(field, "") for field in CSV_FIELDS}
                converted["image_label"] = image_label
                rows[image_label] = converted
    return rows


def write_rows(csv_path, rows_by_image, image_order):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        image_order_set = set(image_order)
        for image_path in image_order:
            row = rows_by_image.get(image_path)
            if row:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        for image_path, row in rows_by_image.items():
            if image_path not in image_order_set:
                writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def sync_rows_to_overlay(rows_by_image, overlay_dir, image_order=None):
    """
    If a user deletes overlay images manually, remove the corresponding CSV rows.

    To avoid wiping a CSV because of a wrong/empty overlay path, syncing is only
    applied when the overlay directory exists and contains at least one image.
    Rows from other image folders are preserved so a shared CSV can accumulate
    labels across multiple datasets.
    """
    if overlay_dir is None:
        return 0
    overlay_dir = Path(overlay_dir)
    if not overlay_dir.exists():
        return 0
    overlay_names = {p.name for p in overlay_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    if not overlay_names:
        return 0
    image_order_set = set(image_order) if image_order is not None else None
    to_delete = [
        image_path
        for image_path in rows_by_image
        if (image_order_set is None or image_path in image_order_set)
        if Path(image_path).name not in overlay_names
    ]
    for image_path in to_delete:
        rows_by_image.pop(image_path, None)
    return len(to_delete)


def delete_overlay_for_image(overlay_dir, image_path):
    if overlay_dir is None:
        return
    overlay_path = Path(overlay_dir) / Path(image_path).name
    if overlay_path.exists():
        overlay_path.unlink()


def blank_row(image_path, status, label_source=""):
    row = {field: "" for field in CSV_FIELDS}
    row["image_label"] = image_label_from_path(image_path)
    row["status"] = status
    return row


def build_skip_row(image_path, mask_path, center=None, status="skipped", label_source="manual_skip"):
    row = blank_row(image_path, status, label_source=label_source)
    if center is not None:
        row["center_x"] = f"{float(center[0]):.8f}"
        row["center_y"] = f"{float(center[1]):.8f}"
    return row


class AngleLabeler:
    def __init__(self, overlay_dir=None, display_size=DEFAULT_DISPLAY_SIZE):
        self.window = "kernel_angle_labeler"
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        self.display_size = max(1, int(display_size))
        self.dragging = False
        self.theta = 0.0
        self.center = None
        self.current_display = None

    def set_sample(self, image_bgr, mask, contour, center, theta, sample_text, status):
        self.image_bgr = image_bgr
        self.mask = mask
        self.contour = contour
        self.center = np.asarray(center, dtype=float)
        self.theta = normalize_theta(theta)
        self.sample_text = sample_text
        self.status = status
        self.current_display = None

    def update_theta_from_mouse(self, x, y):
        vec = np.array([float(x) - self.center[0], float(y) - self.center[1]], dtype=float)
        if np.linalg.norm(vec) < 1.0:
            return
        self.theta = normalize_theta(math.atan2(vec[1], vec[0]))

    def mouse_callback(self, event, x, y, flags, param):
        display_y = y - HUD_HEIGHT
        if event == cv2.EVENT_LBUTTONDOWN:
            if display_y < 0:
                return
            self.dragging = True
            image_x, image_y = display_to_image_coords(x, display_y, self.image_bgr.shape, self.display_size)
            self.update_theta_from_mouse(image_x, image_y)
            self.current_display = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if display_y >= 0:
                image_x, image_y = display_to_image_coords(x, display_y, self.image_bgr.shape, self.display_size)
                self.update_theta_from_mouse(image_x, image_y)
                self.current_display = None
        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging and display_y >= 0:
                image_x, image_y = display_to_image_coords(x, display_y, self.image_bgr.shape, self.display_size)
                self.update_theta_from_mouse(image_x, image_y)
                self.current_display = None
            self.dragging = False

    def rotate_deg(self, degrees):
        self.theta = normalize_theta(self.theta + math.radians(degrees))
        self.current_display = None

    def render(self):
        if self.current_display is None:
            self.current_display, self.p1, self.p2 = draw_overlay(
                self.image_bgr,
                self.mask,
                self.contour,
                self.center,
                self.theta,
                self.sample_text,
                self.status,
                self.display_size,
            )
        return self.current_display

    def save_overlay(self, image_name):
        if self.overlay_dir is None:
            return
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay, _, _ = draw_overlay(
            self.image_bgr,
            self.mask,
            self.contour,
            self.center,
            self.theta,
            self.sample_text,
            self.status,
            self.display_size,
        )
        cv2.imwrite(str(self.overlay_dir / image_name), overlay)

    def label_one(self):
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        while True:
            cv2.imshow(self.window, self.render())
            key = cv2.waitKeyEx(20)

            if key in (-1, 255, 0xFF):
                continue
            key_ascii = key & 0xFF
            if key == 27 or key_ascii == 27:
                return "quit"
            if key in LEFT_KEYS or key in UP_KEYS:
                return "prev"
            if key in RIGHT_KEYS or key in DOWN_KEYS:
                return "next"
            if key in DELETE_KEYS:
                return "delete"
            if key_ascii in (ord("q"), ord("Q")):
                return "quit"
            if key_ascii in (13, 10, 32):
                return "save"
            if key_ascii in (ord("s"), ord("S")):
                return "skip"
            if key_ascii in (ord("x"), ord("X")):
                return "delete"
            if key_ascii in (ord("a"), ord("A")):
                self.rotate_deg(-1.0)
            elif key_ascii in (ord("d"), ord("D")):
                self.rotate_deg(1.0)
            elif key_ascii in (ord("z"), ord("Z")):
                self.rotate_deg(-5.0)
            elif key_ascii in (ord("c"), ord("C")):
                self.rotate_deg(5.0)
            elif key_ascii in (ord("f"), ord("F")):
                self.rotate_deg(180.0)


def build_label_row(image_path, mask_path, theta, center, p1, p2, status, label_source):
    theta = normalize_theta(theta)
    return {
        "image_label": image_label_from_path(image_path),
        "status": status,
        "theta_deg": f"{math.degrees(theta):.8f}",
        "theta_rad": f"{theta:.10f}",
        "cos_theta": f"{math.cos(theta):.10f}",
        "sin_theta": f"{math.sin(theta):.10f}",
        "cos2theta": f"{math.cos(2.0 * theta):.10f}",
        "sin2theta": f"{math.sin(2.0 * theta):.10f}",
        "center_x": f"{float(center[0]):.8f}",
        "center_y": f"{float(center[1]):.8f}",
        "axis_x1": f"{float(p1[0]):.8f}",
        "axis_y1": f"{float(p1[1]):.8f}",
        "axis_x2": f"{float(p2[0]):.8f}",
        "axis_y2": f"{float(p2[1]):.8f}",
    }


def prepare_output_csv(csv_path, no_resume):
    csv_path = Path(csv_path)
    if no_resume and csv_path.exists():
        csv_path.unlink()
    ensure_csv_header(csv_path)
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()


def backup_existing_csv(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = csv_path.with_name(f"{csv_path.name}.bak_{timestamp}")
    shutil.copy2(csv_path, backup_path)
    return backup_path


def row_has_theta(row):
    try:
        return bool(row) and row.get("theta_rad", "") != "" and math.isfinite(float(row["theta_rad"]))
    except (TypeError, ValueError):
        return False


def theta_from_existing_or_initial(row, mask, init_mode):
    if row_has_theta(row):
        return normalize_theta(float(row["theta_rad"]))
    return choose_initial_theta(mask, init_mode)


def find_first_unlabeled(images, rows_by_image):
    for idx, image_path in enumerate(images):
        if image_label_from_path(image_path) not in rows_by_image:
            return idx
    return max(0, len(images) - 1)


def clamp_index(index, images):
    if not images:
        return 0
    return max(0, min(len(images) - 1, index))


def next_index_after_record(images, rows_by_image, current_idx):
    if not images:
        return None
    for offset in range(1, len(images) + 1):
        idx = (current_idx + offset) % len(images)
        image_path = images[idx]
        if image_label_from_path(image_path) not in rows_by_image:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(description="Local semi-automatic kernel axis angle labeler.")
    parser.add_argument("--images", required=True, help="Folder of RGB kernel crops.")
    parser.add_argument("--masks", required=True, help="Folder of matching binary masks.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--overlay-dir", default=None, help="Optional folder for saved overlay visualizations.")
    parser.add_argument("--display-size", type=int, default=DEFAULT_DISPLAY_SIZE, help="Square display size after longest-side padding. Default: 320.")
    parser.add_argument("--no-resume", action="store_true", help="Relabel all images instead of skipping existing CSV rows.")
    parser.add_argument(
        "--init",
        choices=["moments", "random", "horizontal", "vertical"],
        default="moments",
        help="Initial red-axis angle for each image. Default: moments.",
    )
    args = parser.parse_args()

    image_dir = Path(args.images)
    mask_dir = Path(args.masks)
    out_csv = Path(args.out)
    had_existing_csv = out_csv.exists() and out_csv.stat().st_size > 0
    had_legacy_csv = had_existing_csv and csv_uses_legacy_header(out_csv)
    prepare_output_csv(out_csv, args.no_resume)
    backup_path = backup_existing_csv(out_csv) if (had_existing_csv and not args.no_resume) else None
    if backup_path is not None:
        print(f"backup CSV: {backup_path}")

    images = iter_images(image_dir)
    image_order = [image_label_from_path(p) for p in images]
    rows_by_image = {} if args.no_resume else load_existing_rows(out_csv)
    removed_by_overlay_sync = sync_rows_to_overlay(rows_by_image, args.overlay_dir, image_order)
    if removed_by_overlay_sync:
        write_rows(out_csv, rows_by_image, image_order)
        print(f"overlay sync: removed {removed_by_overlay_sync} CSV rows whose overlay image was deleted.")
    elif had_legacy_csv and not args.no_resume:
        write_rows(out_csv, rows_by_image, image_order)
        print("converted CSV to image_label format.")

    labeled_count = sum(1 for p in images if image_label_from_path(p) in rows_by_image)
    print(f"images: {len(images)} | already labeled/skipped: {labeled_count} | remaining: {len(images) - labeled_count}")
    print("Controls: mouse drag/click=set arrow angle | a/d=-/+1 deg | z/c=-/+5 deg | f=flip 180 deg")
    print("          Enter/Space=save ok | s=skip bad image | Left/Up=previous | Right/Down=next | x/Delete=delete current label | q/Esc=quit")

    labeler = AngleLabeler(overlay_dir=args.overlay_dir, display_size=args.display_size)
    if not images:
        print("No images found.")
        return

    idx = find_first_unlabeled(images, rows_by_image)
    while 0 <= idx < len(images):
        image_path = images[idx]
        image_label = image_label_from_path(image_path)
        existing_row = rows_by_image.get(image_label)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            row = blank_row(image_path, "image_read_failed")
            rows_by_image[image_label] = row
            write_rows(out_csv, rows_by_image, image_order)
            print(f"[{idx + 1}/{len(images)}] {image_path.name}: image_read_failed")
            next_idx = next_index_after_record(images, rows_by_image, idx)
            if next_idx is None:
                break
            idx = next_idx
            continue

        mask, mask_path, mask_source = load_main_mask(mask_dir, image_path, image_bgr)
        if mask is None:
            row = blank_row(image_path, "mask_missing_or_invalid", label_source=mask_source)
            rows_by_image[image_label] = row
            write_rows(out_csv, rows_by_image, image_order)
            print(f"[{idx + 1}/{len(images)}] {image_path.name}: mask_missing_or_invalid")
            next_idx = next_index_after_record(images, rows_by_image, idx)
            if next_idx is None:
                break
            idx = next_idx
            continue

        center = centroid_from_mask(mask)
        contour = extract_external_contour(mask)
        if center is None or contour is None:
            row = blank_row(image_path, "moments_or_contour_failed", label_source=mask_source)
            rows_by_image[image_label] = row
            write_rows(out_csv, rows_by_image, image_order)
            print(f"[{idx + 1}/{len(images)}] {image_path.name}: moments_or_contour_failed")
            next_idx = next_index_after_record(images, rows_by_image, idx)
            if next_idx is None:
                break
            idx = next_idx
            continue

        theta = theta_from_existing_or_initial(existing_row, mask, args.init)
        current_status = existing_row.get("status", "editing") if existing_row else "editing"
        sample_text = f"{idx + 1}/{len(images)} {image_path.name}"
        labeler.set_sample(
            image_bgr=image_bgr,
            mask=mask,
            contour=contour,
            center=center,
            theta=theta,
            sample_text=sample_text,
            status=current_status,
        )

        action = labeler.label_one()
        if action == "quit":
            print("quit requested; progress saved.")
            break
        if action == "prev":
            idx = clamp_index(idx - 1, images)
            continue
        if action == "next":
            idx = clamp_index(idx + 1, images)
            continue
        if action == "delete":
            rows_by_image.pop(image_label, None)
            delete_overlay_for_image(args.overlay_dir, image_path)
            write_rows(out_csv, rows_by_image, image_order)
            print(f"[{idx + 1}/{len(images)}] deleted label for {image_path.name}")
            continue
        if action == "skip":
            row = build_skip_row(
                image_path=image_path,
                mask_path=mask_path,
                center=center,
                status="skipped",
                label_source=f"manual_skip:{mask_source}",
            )
            rows_by_image[image_label] = row
            write_rows(out_csv, rows_by_image, image_order)
            labeler.status = "skipped"
            labeler.current_display = None
            labeler.save_overlay(image_path.name)
            print(f"[{idx + 1}/{len(images)}] skipped {image_path.name}")
            next_idx = next_index_after_record(images, rows_by_image, idx)
            if next_idx is None:
                print("All images now have a CSV row.")
                break
            idx = next_idx
            continue

        p1, p2 = axis_endpoints(contour, center, labeler.theta)
        row = build_label_row(
            image_path=image_path,
            mask_path=mask_path,
            theta=labeler.theta,
            center=center,
            p1=p1,
            p2=p2,
            status="ok",
            label_source=f"manual_directed_axis_angle:{mask_source}",
        )
        rows_by_image[image_label] = row
        write_rows(out_csv, rows_by_image, image_order)
        labeler.status = "ok"
        labeler.current_display = None
        labeler.save_overlay(image_path.name)
        print(f"[{idx + 1}/{len(images)}] saved {image_path.name}: theta={row['theta_deg']} deg")
        next_idx = next_index_after_record(images, rows_by_image, idx)
        if next_idx is None:
            print("All images now have a CSV row.")
            break
        idx = next_idx

    cv2.destroyAllWindows()
    print(f"labels saved: {out_csv}")


if __name__ == "__main__":
    main()
