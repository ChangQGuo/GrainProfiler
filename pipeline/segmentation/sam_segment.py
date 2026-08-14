"""
segmentation/sam_segment.py
============================
Stage 3 — Segment each kernel precisely using SAM2, then save subimages and masks.

READS:   Full tray images (config → input.image_dir)
         yolo_bounding_boxes.json  (from Stage 2 — detection)

WRITES:  For each kernel in each image:
           subimages/             — cropped kernel image (masked, neutral-gray background)
           masks_binary/          — binary mask of that kernel in subimage space
         For each full image:
           masks_combined/        — all kernels merged into one binary mask (full image)
           masks_overlay/         — milk-yellow overlay of all kernels (QC visualization)

Subimage naming convention: {original_image_stem}_kernel_{index:03d}.jpg
  e.g. IMG_14_kernel_001.jpg, IMG_14_kernel_002.jpg, ...
This naming links each subimage back to the original image (and thus to OCR metadata).

How SAM2 is used here:
  - We load one SAM2ImagePredictor and set the full image once per image.
  - Then for each YOLO bounding box we call predictor.predict() with that box.
  - This gives one binary mask per kernel.
  - We then crop the mask and the image to the bbox region (+ padding).
  - The cropped binary mask is applied to the cropped image → neutral-gray-background
    subimage (background pixels filled with (128, 128, 128), see crop_kernel).

NOTE: SAM2 must be run from its own repo directory (set in config → models.sam2_repo_dir)
      because it loads config files relative to that directory.

RUN WITH: SAM2 conda environment
  conda activate SAM2
  cd /home/revo/sam2-main
  python /path/to/pipeline/segmentation/sam_segment.py config.yaml
"""

import os
import sys
import json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from processing.contour_extraction import extract_largest_contour
from utils.device import resolve_stage_device


from utils.config import load_config


from utils.bbox import get_accepted_detections
def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def cleanup_previous_outputs(image_stem, image_file, subimages_dir, binary_dir, combined_dir, overlay_dir):
    """Remove stale outputs for one image so reruns stay in sync."""
    prefix = f'{image_stem}_kernel_'

    for directory in [subimages_dir, binary_dir]:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.startswith(prefix) and name.lower().endswith(('.jpg', '.jpeg', '.png')):
                os.remove(os.path.join(directory, name))

    remove_file_if_exists(os.path.join(combined_dir, f'combined_{image_file}'))
    remove_file_if_exists(os.path.join(overlay_dir, f'overlay_{image_file}'))


# ---------------------------------------------------------------------------
# SAM2 setup
# ---------------------------------------------------------------------------

def load_sam2_predictor(config):
    """Load the SAM2 model. Must be called from the sam2_repo_dir."""
    import torch
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        print('ERROR: sam2 package not found.')
        print('Make sure you are in the SAM2 conda env and the sam2 repo is set up.')
        sys.exit(1)

    checkpoint = config['models']['sam2_checkpoint']
    model_cfg  = config['models']['sam2_config']
    device     = resolve_stage_device(config, 'segmentation')

    print(f'Loading SAM2 from: {checkpoint} | device={device}')
    sam2_model = build_sam2(model_cfg, checkpoint, device=device, apply_postprocessing=False)
    predictor  = SAM2ImagePredictor(sam2_model)
    print('SAM2 loaded.')
    return predictor


# ---------------------------------------------------------------------------
# Per-kernel segmentation
# ---------------------------------------------------------------------------

def segment_image(predictor, image_bgr, boxes):
    """
    Run SAM2 on a full image using YOLO bounding boxes as prompts.
    Returns a list of binary masks, one per box, in full-image pixel space.

    Args:
        predictor : SAM2ImagePredictor (already loaded)
        image_bgr : full image as a BGR numpy array
        boxes     : list of [x1, y1, x2, y2] boxes from YOLO

    Returns:
        masks: list of (H, W) uint8 arrays with values 0 or 255
    """
    import torch

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Set the image once — SAM2 encodes it as embeddings once and reuses them
    # for all boxes in this image. This is more efficient than calling set_image
    # inside the loop.
    predictor.set_image(image_rgb)

    masks = []
    for box in boxes:
        x1, y1, x2, y2 = box
        input_box = np.array([x1, y1, x2, y2], dtype=np.float32)[None, :]  # shape (1, 4)

        pred_masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=False,  # we only want one mask per box
        )

        # pred_masks shape: (1, H, W) — take the first (and only) mask
        binary_mask = (pred_masks[0] > 0.5).astype(np.uint8) * 255
        masks.append(binary_mask)

    return masks


# ---------------------------------------------------------------------------
# Subimage and mask saving
# ---------------------------------------------------------------------------

def crop_kernel(image_bgr, binary_mask_full, box, padding=20):
    """
    Crop one kernel from the full image using its bounding box + padding.
    Applies the SAM2 mask so the background is neutral gray.

    Args:
        image_bgr         : full original image (BGR)
        binary_mask_full  : full-image binary mask for this kernel (H, W, uint8)
        box               : [x1, y1, x2, y2] YOLO bounding box
        padding           : pixels of extra space added around the box

    Returns:
        subimage     : cropped BGR image with background set to neutral gray
        cropped_mask : binary mask in subimage coordinate space (0 or 255)
        crop_coords  : (x1_pad, y1_pad, x2_pad, y2_pad) actual crop used
    """
    H, W = image_bgr.shape[:2]
    x1, y1, x2, y2 = box

    # Add padding but clamp to image boundaries
    x1p = max(0, x1 - padding)
    y1p = max(0, y1 - padding)
    x2p = min(W, x2 + padding)
    y2p = min(H, y2 + padding)

    # Crop both the image and the mask to the padded bbox
    cropped_img  = image_bgr[y1p:y2p, x1p:x2p].copy()
    cropped_mask = binary_mask_full[y1p:y2p, x1p:x2p].copy()

    # Apply mask: set background pixels to neutral gray (128,128,128)
    # Gray is safe for all maize kernel colors (white/yellow/orange/red/purple/black)
    # and ensures the ResNet axis model can distinguish kernel edges from background.
    subimage = np.full_like(cropped_img, fill_value=128)
    subimage[cropped_mask > 0] = cropped_img[cropped_mask > 0]

    return subimage, cropped_mask, (x1p, y1p, x2p, y2p)


def pad_to_canvas(image, canvas_size=640, pad_value=0):
    """
    Put a crop on a fixed square canvas without resizing the kernel.

    This preserves the original pixel scale. If a rare crop is larger than the
    target canvas, it is center-cropped instead of being squeezed.
    """
    if canvas_size is None or int(canvas_size) <= 0:
        return image, False

    canvas_size = int(canvas_size)
    h, w = image.shape[:2]
    was_cropped = False

    if h > canvas_size or w > canvas_size:
        y1 = max(0, (h - canvas_size) // 2)
        x1 = max(0, (w - canvas_size) // 2)
        image = image[y1:y1 + min(h, canvas_size), x1:x1 + min(w, canvas_size)].copy()
        h, w = image.shape[:2]
        was_cropped = True

    if image.ndim == 2:
        canvas = np.full((canvas_size, canvas_size), pad_value, dtype=image.dtype)
        y = (canvas_size - h) // 2
        x = (canvas_size - w) // 2
        canvas[y:y + h, x:x + w] = image
    else:
        canvas = np.full((canvas_size, canvas_size, image.shape[2]), pad_value, dtype=image.dtype)
        y = (canvas_size - h) // 2
        x = (canvas_size - w) // 2
        canvas[y:y + h, x:x + w, :] = image

    return canvas, was_cropped


def write_image(path, image):
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.jpg', '.jpeg'}:
        cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    else:
        cv2.imwrite(path, image)


# ---------------------------------------------------------------------------
# Full-image combined outputs (for ImageJ / Momocs)
# ---------------------------------------------------------------------------

def make_combined_mask(image_shape, all_masks_full):
    """
    Merge all per-kernel full-image masks into one combined binary mask.
    Any pixel that belongs to any kernel is white (255).
    """
    combined = np.zeros(image_shape[:2], dtype=np.uint8)
    for mask in all_masks_full:
        combined = cv2.bitwise_or(combined, mask)
    return combined


def make_milk_yellow_overlay(image_bgr, combined_mask, alpha=0.5, color_bgr=(100, 200, 230)):
    """
    Create the milk-yellow colored overlay on the original image.
    Kernel regions get the warm yellow color blended over them.
    This is the visualization that resembles real maize kernel color.
    """
    result = image_bgr.copy()
    color_layer = np.full_like(image_bgr, color_bgr, dtype=np.uint8)
    kernel_pixels = combined_mask > 0
    result[kernel_pixels] = cv2.addWeighted(
        image_bgr, 1 - alpha,
        color_layer, alpha, 0
    )[kernel_pixels]
    return result


# ---------------------------------------------------------------------------
# Main stage runner
# ---------------------------------------------------------------------------

def run(config_path):
    config = load_config(config_path)

    image_dir    = config['input']['image_dir']
    ext          = config['input']['image_extension']
    output_dir   = config['output']['base_dir']
    bbox_json    = os.path.join(output_dir, config['output']['bounding_boxes_json'])

    subimages_dir  = os.path.join(output_dir, config['output']['subimages_dir'])
    binary_dir     = os.path.join(output_dir, config['output']['masks_binary_dir'])
    combined_dir   = os.path.join(output_dir, config['output']['masks_combined_dir'])
    overlay_dir    = os.path.join(output_dir, config['output']['masks_overlay_dir'])

    padding        = config['segmentation']['padding']
    canvas_size    = int(config.get('segmentation', {}).get('subimage_canvas_size', 640))
    color_bgr      = tuple(config['visualization']['milk_yellow_bgr'])
    alpha          = config['visualization']['overlay_alpha']
    save_qc        = bool(config.get('visualization', {}).get('save_qc', True))

    contours_json = os.path.join(output_dir,
                                  config['output'].get('kernel_contours_json', 'kernel_contours.json'))
    contours_dir = os.path.join(output_dir,
                                config['output'].get('contours_dir', 'contours'))
    os.makedirs(contours_dir, exist_ok=True)

    min_contour_area = int(config.get('contour_filtering', {}).get('min_contour_area', 500))

    for d in [subimages_dir, binary_dir, combined_dir, overlay_dir]:
        os.makedirs(d, exist_ok=True)

    # Load bounding boxes from Stage 2
    if not os.path.exists(bbox_json):
        print(f'ERROR: Bounding boxes file not found: {bbox_json}')
        print('Run detection/yolo_detect.py first.')
        sys.exit(1)

    with open(bbox_json, 'r') as f:
        all_boxes = json.load(f)

    # Load SAM2
    # NOTE: SAM2 needs to be run from its repo directory for config loading to work
    sam2_dir = config['models']['sam2_repo_dir']
    if os.getcwd() != sam2_dir:
        print(f'Changing directory to SAM2 repo: {sam2_dir}')
        os.chdir(sam2_dir)

    predictor = load_sam2_predictor(config)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    print(f'Segmenting kernels in {len(image_files)} images...')

    all_contours = []  # collect per-kernel contours for single-file JSON (backward compat)
    total_contours = 0

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_stem = os.path.splitext(image_file)[0]
        image_contours: dict[str, list] = {}  # per-kernel contours for this image
        cleanup_previous_outputs(
            image_stem,
            image_file,
            subimages_dir,
            binary_dir,
            combined_dir,
            overlay_dir,
        )

        accepted = get_accepted_detections(all_boxes.get(image_file, []))
        boxes = [item['box'] for item in accepted]

        if not boxes:
            print(f'  {image_file}: no accepted boxes found, skipping')
            continue

        print(f'  {image_file}: {len(boxes)} kernels')

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f'  WARNING: Could not read {image_path}')
            continue

        # Run SAM2 for all kernels in this image
        masks_full = segment_image(predictor, image_bgr, boxes)

        # Save per-kernel outputs
        for det, mask_full in zip(accepted, masks_full):
            kernel_name = f"{image_stem}_kernel_{det['kernel_id']:03d}.jpg"

            # Crop subimage and mask
            subimage, cropped_mask, _ = crop_kernel(image_bgr, mask_full, det['box'], padding)
            subimage, image_canvas_cropped = pad_to_canvas(subimage, canvas_size=canvas_size, pad_value=0)
            cropped_mask, mask_canvas_cropped = pad_to_canvas(cropped_mask, canvas_size=canvas_size, pad_value=0)
            if image_canvas_cropped or mask_canvas_cropped:
                print(f"    WARNING: {kernel_name} crop exceeded {canvas_size}x{canvas_size}; center-cropped without resizing.")

            # Save the cropped kernel image (neutral-gray background)
            write_image(os.path.join(subimages_dir, kernel_name), subimage)

            # Save the binary mask in subimage coordinate space
            # white (255) = kernel pixels, black (0) = background
            write_image(os.path.join(binary_dir, kernel_name), cropped_mask)

            # Extract contour from the cropped binary mask
            contour = extract_largest_contour(cropped_mask, min_area=min_contour_area)
            if contour is not None:
                entry = {
                    'kernel_name': kernel_name,
                    'contour': [[float(x), float(y)] for x, y in contour],
                }
                all_contours.append(entry)
                image_contours[kernel_name] = entry['contour']

        # Save full-image combined mask (all kernels together)
        combined_mask = make_combined_mask(image_bgr.shape, masks_full)
        combined_path = os.path.join(combined_dir, f'combined_{image_file}')
        cv2.imwrite(combined_path, combined_mask)

        # Save milk-yellow overlay (full image with colored kernels)
        overlay = make_milk_yellow_overlay(image_bgr, combined_mask, alpha, color_bgr)
        overlay_path = os.path.join(overlay_dir, f'overlay_{image_file}')
        cv2.imwrite(overlay_path, overlay)
        print(f'    Saved overlay → {os.path.basename(overlay_path)}')
        print(f'    Saved {len(boxes)} subimages and binary masks')

        # Write per-image contour file
        if image_contours:
            img_contour_file = os.path.join(contours_dir, f'{image_file}_contours.json')
            with open(img_contour_file, 'w') as f:
                json.dump(image_contours, f)
            total_contours += len(image_contours)

    # Save per-kernel contours as JSON
    with open(contours_json, 'w') as f:
        json.dump(all_contours, f, indent=2)
    print(f'\nSegmentation complete.')
    print(f'  Subimages:        {subimages_dir}')
    print(f'  Binary masks:     {binary_dir}')
    print(f'  Combined masks:   {combined_dir}')
    print(f'  Overlay images:   {overlay_dir}')
    print(f'  Contours (single): {contours_json}  ({len(all_contours)} kernels)')
    print(f'  Contours (per-img): {contours_dir}/  ({total_contours} kernels, {len(image_files)} images)')
    print('Check the overlay images to confirm kernels are correctly segmented.')

    # --- QC cleanup: remove disabled outputs ---
    _cleanup_qc(config, output_dir)


def _cleanup_qc(config, output_dir):
    import shutil
    vis = config.get('visualization', {})

    def _rm(path_key, default_name):
        full = os.path.join(output_dir, config['output'].get(path_key, default_name))
        if os.path.exists(full):
            shutil.rmtree(full) if os.path.isdir(full) else os.remove(full)
            print(f'  Cleaned: {os.path.basename(full)}')

    if not vis.get('save_masks_overlay', True):
        _rm('masks_overlay_dir', 'masks_overlay')
    if not vis.get('save_masks_combined', True):
        _rm('masks_combined_dir', 'masks_combined')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    run(config_path)
