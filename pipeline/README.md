# Maize Kernel Phenotyping Pipeline

Automated pipeline that converts tray photographs of maize kernels into per-kernel and
per-plant phenotypic tables ready for GWAS and downstream shape analysis. It consists of
**8 stages (indexed 0–7)** orchestrated by `main.py` as subprocesses running in isolated
Conda environments (YOLO, SAM2, and PaddleOCR cannot share one environment).

## Pipeline Overview

Run order:

```text
pre_ocr -> ocr -> detection -> segmentation -> measurements -> shapes -> vae_encode -> assembly
```

| Stage | Name        | Script                              | Env   | What it does |
|-------|-------------|-------------------------------------|-------|--------------|
| 0     | pre_ocr     | `ocr/yolo_label_detect.py`          | yolo  | YOLO11 detects the paper label and the weight-screen region; YOLOv8n reads the seven-segment weight digits |
| 1     | ocr         | `ocr/metadata_extraction.py`        | paddle| PaddleOCR reads the plant name; combines it with the weight; detects the blue/green tray for px→mm calibration |
| 2     | detection   | `detection/yolo_detect.py`          | yolo  | YOLO11x detects kernel bounding boxes, filtered by tray containment, aspect ratio, size and overlap rules |
| 3     | segmentation| `segmentation/sam_segment.py`       | sam2  | SAM2 segments each kernel into a binary mask; saves subimages + masks + contours |
| 4     | measurements| `measurements/kernel_metrics.py`    | base  | ResNet predicts the directed main axis; computes morphological traits per kernel |
| 5     | shapes      | `processing/representative_shape.py`| base  | Builds the plant-level 100-point median width profile |
| 6     | vae_encode  | `vae/vae_encode.py`                 | base  | Encodes 100-dim profiles into 5-dim latent traits (GWAS input) |
| 7     | assembly    | `output/assembler.py`               | base  | Exports the final individual and plant-median tables |

## Stage Details

### Stage 0 — pre_ocr (`ocr/yolo_label_detect.py`)

- Reads the raw tray images.
- YOLO11 (`models.label_weight_yolo`) detects two object classes: the paper `label` and
  the digital `weight_screen`.
- The weight-screen crop is passed to a lightweight YOLOv8n digit model
  (`models.weight_digit_yolo`) that detects individual digits 0–9. Digits are sorted by
  x-coordinate and a decimal point is inserted two digits from the right
  (scale convention, e.g. `1254` → `12.54`).
- Writes `yolo_label_weight_boxes.json` (label/weight-screen boxes + weight) and
  weight-digit QC crops.

### Stage 1 — ocr (`ocr/metadata_extraction.py`)

- Reads the raw tray images and `yolo_label_weight_boxes.json`.
- PaddleOCR reads the plant name from the label region. If YOLO detection or OCR fails,
  it falls back to a split-region heuristic (upper portion upscaled before OCR).
- The weight parsed in Stage 0 is combined with the plant name.
- When calibration is enabled, the blue/green tray is detected via HSV segmentation and
  the pixel-to-millimeter factor is derived from its known physical size
  (`calibration.tray_size_mm`).
- Writes `metadata.csv` (plant name, weight, calibration info per image).

### Stage 2 — detection (`detection/yolo_detect.py`)

- Reads the raw tray images.
- YOLO11x (`models.yolo_detection`) predicts kernel bounding boxes.
- Boxes are filtered before they reach SAM2:
  1. **Tray containment** — only boxes on the detected tray are kept.
  2. **Aspect ratio** — boxes with aspect ratio > `max_box_aspect_ratio` are removed.
  3. **Statistical size** — boxes whose area or side length deviates too far from the
     tray-internal median are removed (0.35–3.0× area, 0.45–2.2× side).
  4. **Nested overlap suppression** — when one box contains another (overlap > 86%),
     the more plausible one is kept.
- Writes `yolo_bounding_boxes.json` (boxes + filtering metadata + kept IDs) and
  `detect_results.csv` (one row per image for quick QC).

### Stage 3 — segmentation (`segmentation/sam_segment.py`)

- Reads the raw tray images and `yolo_bounding_boxes.json`.
- Each accepted YOLO box becomes a box prompt for SAM2 (`sam2.1_hiera_large`). The full
  image is encoded once and all kernels are segmented in a single pass.
- Each kernel is cropped with 20 px padding (`segmentation.padding`); kernel pixels keep
  their original RGB values and the background is filled with neutral gray.
- Writes:
  - `subimages/` — one cropped RGB subimage per kernel,
  - `masks_binary/` — one cropped binary mask per kernel,
  - `contours/` — per-image contour JSON files (fast lazy loading),
  - `kernel_contours.json` — legacy single-file contour JSON (backward compat).

### Stage 4 — measurements (`measurements/kernel_metrics.py`)

- Reads the cropped SAM binary masks directly from `masks_binary/`.
- Each mask is cleaned: hole filling, small-noise removal, light smoothing, and
  largest-contour extraction.
- The centroid is computed with `cv2.moments`.
- The trained ResNet angle model (`models.resnet_axis`) predicts the directed main-axis
  vector. The axis is passed through the centroid and intersected with the cleaned
  contour to recover the bottom/top endpoints.
- Morphological traits are computed: length, W25/W50/W75, max width, area, perimeter,
  and circularity. Pixel values are converted to mm using the per-image calibration.
- Kernels with circularity above `round_circularity_threshold` (0.90) are marked as
  round and excluded from oriented measurements.
- Writes `measurements.csv`, `measurements.parquet`, `axis_results.json` (axis endpoints,
  reused by Stage 5), and full-tray `measurement_overlay/` images.

### Stage 5 — shapes (`processing/representative_shape.py`)

- Reads the cropped masks from `masks_binary/` and the pre-computed axes from
  `axis_results.json` (ResNet is not re-run).
- For each kernel, a 100-point half-width profile is sampled along the crown→pedicel
  axis (`measurements.num_width_samples`).
- Within each plant, the profiles are aggregated with an **element-wise median** to form
  the representative 100-point median half-width profile. The full-width profile
  (2 × half-width) is exported for PCA / VAE.
- Round kernels can be excluded from the plant profile when
  `representative_shape.skip_round_kernels: true`
  (threshold `representative_shape.round_circularity_threshold`, default 0.95).
- Writes `median_outlines.csv`, `rep_width_profiles.txt`, `kernel_width_profiles.txt`,
  and per-plant plots under `representative_shapes/`.

### Stage 6 — vae_encode (`vae/vae_encode.py`)

- Reads `rep_width_profiles.txt` and the VAE checkpoint (`models.vae_checkpoint`).
- Encodes each plant-level 100-dim width profile into a 5-dim latent vector.
- Writes `latent_traits.csv` (plant ID + latent 1–5), the GWAS input.

### Stage 7 — assembly (`output/assembler.py`)

- Reads `measurements.csv`, `median_outlines.csv`, and `metadata.csv`.
- Attaches plant name and weight to each kernel, computes `weight_per_kernel_g`, and
  drops diagnostic columns.
- Writes `final_output_individual.csv` (per-kernel table) and
  `final_output_plant_median.csv` (plant-level median table).

## Running the Pipeline

```bash
cd pipeline/

# Full run (all 8 stages)
python main.py config.yaml

# Run a single stage
python main.py config.yaml --stage detection

# Resume from a given stage (skips earlier stages)
python main.py config.yaml --from-stage shapes
```

The Python executable of each Conda environment is configured under `environments:` in
`config.yaml`. Set `input.image_dir` and `output.base_dir` before running. All output
folders are created automatically.

## Device Selection

GPU selection is controlled by one shared value:

```yaml
runtime:
  gpu_device: "cuda:1"
```

Stages follow `runtime.gpu_device` when their own device is `null`:

```yaml
detection:
  device: null
ocr:
  device: null
segmentation:
  device: null
measurements:
  axis_model_device: null   # ResNet axis model device
```

Set a stage device to `cpu`, `cuda:0`, etc. only when that stage should use a different
device.

## Core Axis and Geometry Logic

For each kernel:

1. Clean the binary mask, keep one filled kernel body, and compute the centroid with
   `cv2.moments`.
2. Sample 360 ordered outline points, then resample them to equal arc distance.
3. Load the cropped RGB subimage and run the ResNet angle model to predict
   `[cos(θ), sin(θ)]`.
4. Treat the directed model vector as bottom-to-top (pedicel→crown), pass the line
   through the centroid, and intersect it with the cleaned contour.
5. Use the two contour intersections as bottom/top endpoints, then measure length,
   W25/W50/W75, max width, area, perimeter, and circularity.

Key ResNet-axis parameters:

```yaml
models:
  resnet_axis: /path/to/resnet/runs3/weights/best.pt

measurements:
  axis_source: resnet
  axis_model_device: null
  axis_model_imgsz: 0
  axis_model_preprocess: "square"   # current model: square padding
  axis_model_background: "gray"     # current model: gray background
  outline_resample_points: 360
  round_kernel_filter_enabled: true
  round_circularity_threshold: 0.90
```

## Round Kernel Logic

```text
circularity = 4*pi*A / P^2
```

- Stage 4 treats a kernel as round when `circularity > 0.90`
  (`measurements.round_circularity_threshold`) and leaves its length/width blank.
- Stage 5 can skip round kernels from the plant median profile when
  `representative_shape.skip_round_kernels: true` (default off), using its own
  threshold `representative_shape.round_circularity_threshold` (default 0.95).

## Key Outputs

Under `output.base_dir`:

| Stage | File / folder | Description |
|-------|---------------|-------------|
| 0/1   | `metadata.csv` | plant name, weight, calibration per tray image |
| 0/1   | `yolo_label_weight_boxes.json` | label/weight-screen boxes + weight |
| 2     | `yolo_bounding_boxes.json` | kernel boxes + filtering metadata + kept IDs |
| 2     | `detect_results.csv` | one row per image (detection QC summary) |
| 3     | `subimages/`, `masks_binary/` | cropped RGB subimage + binary mask per kernel |
| 3     | `contours/`, `kernel_contours.json` | per-kernel contour points |
| 4     | `measurements.csv` / `.parquet` | per-kernel morphological traits |
| 4     | `axis_results.json` | per-kernel axis endpoints (reused by Stage 5) |
| 4     | `measurement_overlay/` | full-tray axis + centroid + top/bottom overlays |
| 5     | `median_outlines.csv` | plant-level median shape summaries |
| 5     | `rep_width_profiles.txt` | per-plant 100-point median width profiles (PCA/VAE input) |
| 5     | `kernel_width_profiles.txt` | per-kernel 100-point width profiles (GUI hover) |
| 5     | `representative_shapes/` | per-plant median shape plots |
| 6     | `latent_traits.csv` | 5-dim VAE latent traits per plant (GWAS input) |
| 7     | `final_output_individual.csv` | per-kernel table with metadata attached |
| 7     | `final_output_plant_median.csv` | plant-level median table |

Optional QC images (`qc_visualizations/`) are gated by the `visualization.*` flags in
`config.yaml`; most are disabled by default.
