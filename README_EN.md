<div align="center">

**🌐 Language / 语言**

[![中文版](https://img.shields.io/badge/%E4%B8%AD%E6%96%87-%E4%B8%AD%E6%96%87%E7%89%88-2ea44f?style=for-the-badge)](README.md)　[![English](https://img.shields.io/badge/English-English-0969da?style=for-the-badge)](README_EN.md)

</div>

# GrainProfiler — Maize Kernel High-Throughput Morphology Phenotyping System

> **Maize Kernel Morphology Phenotyping System**
> A fully automatic 2D maize-kernel morphology pipeline: **tray photo in → 8-stage processing → GWAS-ready trait data out**, together with a breeder-facing desktop viewer, GrainProfiler.
>
> Processed **15,287 ear samples / 386,386 maize kernels**, covering 61 maize accessions (hybrids, inbred lines, landraces).

> **🚀 Desktop software for rapid result retrieval**
> 1. Download `grainprofiler.exe` (Windows program) from this repository
> 2. Double-click to run — **no Python / GPU / conda required**
> 3. Open the app → click "Open Folder" → select a pipeline result directory
>
> See [§6.3](#63-use-the-grainprofiler-desktop-app)

---

## Table of Contents

- [GrainProfiler — Maize Kernel High-Throughput Morphology Phenotyping System](#grainprofiler--maize-kernel-high-throughput-morphology-phenotyping-system)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Key Features](#2-key-features)
  - [3. Directory Structure](#3-directory-structure)
  - [4. Requirements](#4-requirements)
  - [5. Installation](#5-installation)
  - [6. Quick Start](#6-quick-start)
    - [6.1 Configuration](#61-configuration)
    - [6.2 Run the Pipeline](#62-run-the-pipeline)
    - [6.3 Use the GrainProfiler Desktop App](#63-use-the-grainprofiler-desktop-app)
  - [7. The 8 Pipeline Stages](#7-the-8-pipeline-stages)
    - [Inter-stage data flow (bridge files)](#inter-stage-data-flow-bridge-files)
    - [Stage 2 — 4-level cascade filter (deterministic rules)](#stage-2--4-level-cascade-filter-deterministic-rules)
    - [Stage 4 — directed axis and morphology](#stage-4--directed-axis-and-morphology)
  - [8. Model Zoo](#8-model-zoo)
    - [Key design decisions (WHY)](#key-design-decisions-why)
    - [ResNet training details](#resnet-training-details)
    - [β-VAE training details](#β-vae-training-details)
  - [9. Model Weights](#9-model-weights)
  - [10. Input / Output Data Format](#10-input--output-data-format)
    - [Input](#input)
    - [Output (key files)](#output-key-files)
  - [11. GrainProfiler Desktop App](#11-grainprofiler-desktop-app)
  - [12. Manual Annotation Tool](#12-manual-annotation-tool)
  - [13. FAQ](#13-faq)
  - [14. Downstream Analysis Scripts](#14-downstream-analysis-scripts)
  - [15. Copyright & License](#15-copyright--license)

---

## 1. Project Overview

GrainProfiler is a "**photo in, traits out**" maize-kernel morphology phenotyping system. Given a raw kernel tray photo (5408×4056), it automatically performs sample-information recognition, weight reading, kernel detection, instance segmentation, kernel direction prediction, morphological measurement, continuous shape representation, and latent-trait encoding — finally outputting **per-kernel** and **per-sample** trait tables ready for downstream GWAS.

**Motivation**: traditional kernel phenotyping relies on manual measurement or semi-automatic image processing, which is low-throughput and only yields "predefined" low-dimensional discrete traits (kernel length, width, area, …). These cannot express continuous structural variation such as outline fullness, position of maximum width, crown expansion, or basal contraction. This project upgrades the methodology from "handcrafted metrics" to "data-driven representation learning", closing the loop from images to high-dimensional shape representations to genetic analysis.

**Tech stack**: YOLO11 / YOLOv8n (detection), PaddleOCR (label text), SAM2.1 Hiera-L (instance segmentation), a custom ResNet (directed-axis regression), β-VAE (unsupervised latent traits), PySide6 (desktop GUI), PyInstaller (packaging), Python 3.10 / PyTorch 2.4+.

**Overall workflow**:

![GrainProfiler overall workflow](project_figs/workflow_figure.png)

---

## 2. Key Features

1. **End-to-end automatic workflow**: YOLO11 (label + scale screen + kernels) + SAM2 + PaddleOCR + YOLOv8n (seven-segment digits), turning raw images directly into a structured database.
2. **Kernel direction prediction**: a custom ResNet regresses the "crown → pedicel" directed axis ((cosθ, sinθ) vector regression), removing orientation noise from randomly placed kernels.
3. **100-dim continuous full-width profile**: the 2D outline is sampled into a 100-dim normalized width distribution along the directed axis, moving from discrete scalars to a continuous shape description.
4. **Latent morphology discovery**: PCA (first two components explain **90.84%** of variance) + β-VAE (5-dim non-orthogonal latent traits capturing fullness, taper, width redistribution, and other nonlinear features).
5. **Desktop app GrainProfiler**: interactive result inspection, calibration correction, and latent-space exploration with three-view cross-linking.

**Key quantitative results**

| Component | Metric |
|-----------|--------|
| YOLO11 label/scale-screen detection | P=0.997, R=1.000, mAP50=0.995, mAP50-95=0.939 |
| YOLO11x kernel detection | test set 821 TP / 3 FP; count vs. manual R²≈1.0 |
| End-to-end label recognition | 96.80% |
| End-to-end scale reading | 96.00% |
| Directed-axis prediction | MAE 3.38°, P50=2.35°, 80%<5°, 95.4%<10° |
| PCA | PC1=80.25%, PC2=10.59%, cumulative 90.84% |
| β-VAE | 100-dim → 5-dim latent, β=0.001 |

---

## 3. Directory Structure

```
seed_project_v1.0/
├── README.md                     # Chinese README
├── README_EN.md                  # English README (this file)
├── LICENSE                       # proprietary license (closed-source · all rights reserved)
├── .gitignore                    # git ignore rules (data / caches / large files)
├── .gitattributes                # Git LFS tracking rules (grainprofiler.exe)
├── grainprofiler.exe                 # Windows desktop app (stored via Git LFS, double-click to run)
├── grainprofiler.spec                # PyInstaller build spec (produces grainprofiler.exe)
├── yolo_environment.yml          # conda env: YOLO detection / digits
├── SAM2_environment.yml          # conda env: SAM2 segmentation / measurements
├── paddle_environment.yml        # conda env: PaddleOCR label text
│
├── pipeline/                     # 8-stage pipeline core
│   ├── main.py                   # orchestrator (pre_ocr→ocr→detection→segmentation→measurements→shapes→vae_encode→assembly)
│   ├── config.yaml               # global config (model paths / thresholds / calibration / output)
│   ├── env.local.yaml.example    # local override template (copy to env.local.yaml)
│   ├── README.md                 # pipeline notes
│   ├── ocr/                      # Stage 0-1: label/scale-screen detection + OCR
│   │   ├── yolo_label_detect.py  # YOLO11 label/scale-screen ROIs + YOLOv8n seven-segment digits
│   │   ├── metadata_extraction.py# PaddleOCR label text + weight parsing + calibration
│   │   ├── qc_render.py          # OCR QC rendering (view layer)
│   │   └── text_parse.py         # text normalization & candidate parsing
│   ├── detection/                # Stage 2: kernel detection
│   │   └── yolo_detect.py        # YOLO11x detection + 4-level cascade filter
│   ├── segmentation/             # Stage 3: instance segmentation
│   │   └── sam_segment.py        # SAM2 masks + RGB crops (neutral-gray bg)
│   ├── measurements/             # Stage 4: directed axis + morphology
│   │   ├── kernel_metrics.py     # facade: orchestration + ResNet axis + record assembly + run()
│   │   ├── geometry.py           # pure geometry primitives (resampling/intersections/width/area/circularity)
│   │   ├── axis.py               # axis candidate generation / multi-cue scoring / selection / refinement
│   │   └── qc.py                 # measurement QC rendering
│   ├── processing/               # Stage 5: representative shape
│   │   ├── representative_shape.py # per-plant median outline + 100-dim width profile
│   │   └── contour_extraction.py   # contour extraction (largest connected component)
│   ├── vae/                      # Stage 6: VAE latent-trait encoding
│   │   ├── vae_encode.py         # 100-dim profile → 5-dim latent traits
│   │   ├── model.py              # VAE Encoder (used at encoding time)
│   │   └── vae_checkpoint.pt     # trained VAE weights (shipped, tiny)
│   ├── output/                   # Stage 7: final CSV assembly
│   │   └── assembler.py          # final_output_individual / plant_median
│   └── utils/                    # shared utilities
│       ├── config.py             # shared config loading (env.local.yaml override)
│       ├── kernel_id.py          # kernel identity parsing (single source of truth)
│       ├── bbox.py               # YOLO bbox JSON parsing (single source of truth)
│       ├── calibration.py        # tray calibration (mm/px)
│       ├── device.py             # GPU device selection
│       └── visualization.py      # drawing helpers
│
├── grainprofiler/                    # desktop GUI source (Windows build source)
│   ├── main.py                   # entry point
│   ├── __main__.py               # python -m grainprofiler entry
│   ├── __init__.py               # package marker
│   ├── app/                      # main window + data loading
│   │   ├── __init__.py           # package marker
│   │   ├── main_window.py        # QMainWindow: three-pane workspace + analysis panels
│   │   ├── data_loader.py        # result loading (Parquet-first + lazy contours)
│   │   ├── models.py             # data models
│   │   └── settings.py           # settings
│   ├── widgets/                  # panel widgets
│   │   ├── __init__.py           # package marker
│   │   ├── welcome_widget.py     # welcome page
│   │   ├── nav_panel.py          # sample list / filter / theme
│   │   ├── sample_view.py        # QGraphicsView interactive tray photo + contour overlay
│   │   ├── median_chart.py       # half-width profile chart (cross-linked)
│   │   ├── sample_info_panel.py  # metadata + measurement table
│   │   ├── kernel_detail_dialog.py # per-kernel detail
│   │   ├── pca_window.py         # PCA scatter
│   │   ├── vae_latent_window.py  # VAE latent real-time decode (ONNX)
│   │   ├── sample_analysis_window.py # trait distribution
│   │   └── similarity_boxplot.py # similarity boxplot
│   ├── graphics/                 # QGraphics items
│   │   ├── __init__.py           # package marker
│   │   ├── kernel_contour_item.py # kernel contour item
│   │   ├── axis_line_item.py     # axis line item
│   │   └── axis_endpoint_item.py # axis endpoint item
│   ├── models/                   # Qt table models
│   │   ├── __init__.py           # package marker
│   │   ├── pandas_model.py       # pandas table model
│   │   └── sort_filter_proxy.py  # sort/filter proxy
│   ├── utils/                    # utilities
│   │   ├── __init__.py           # package marker
│   │   ├── coordinate_transform.py # coordinate transforms
│   │   ├── export.py             # CSV export
│   │   ├── image_conversion.py   # image conversion
│   │   └── photo_finder.py       # photo finder
│   └── resources/                # cursor/icon resources
│       ├── cursors.py            # cursor definitions
│       └── gen*.py / generate_assets.py # asset generation scripts (one-off)
│
├── onnx_models/                  # VAE Decoder ONNX (used by GUI latent window)
│   ├── profile_vae_latent5.onnx        # decoder graph
│   ├── profile_vae_latent5.onnx.data   # decoder weights (external data)
│   ├── vae_col_mean.npy          # training per-position mean (denormalization)
│   └── vae_col_std.npy           # training per-position std (denormalization)
├── grainprofiler_minifig/            # GUI icon resources
│   ├── 图标.png                  # app icon
│   └── 玉米.png                  # maize icon
│
├── resnet/                       # directed-axis regression model (custom ~2.2M)
│   ├── model.py                  # ResNetAngleRegressor + BasicBlock (3/4-channel)
│   ├── preprocess.py             # square/crop/mask/none preprocessing
│   ├── train_resnet_angle.py     # training (direction cosine loss)
│   ├── predict_resnet_angle.py   # inference
│   ├── test_resnet_angle.py      # evaluation
│   ├── plot_results.py           # standalone plotting (no PyTorch required)
│   ├── config.yaml               # training config
│   ├── run_train.sh              # training launch script
│   ├── run_test.sh               # test launch script
│   └── README.md                 # notes
│
├── vae/                          # β-VAE unsupervised shape traits
│   ├── model.py                  # 1D-CNN Encoder/Decoder (100→50→25→5)
│   ├── dataset.py                # data loading + per-position Z-score
│   ├── train_vae.py              # training (MSE + β·KL)
│   ├── export_onnx.py            # Decoder → ONNX
│   ├── interpret_latents.py      # latent-dimension perturbation analysis
│   ├── latent_shape_explorer.py  # extreme-decoding visualization
│   ├── latent_perturbation_grid.py # perturbation grid
│   ├── reconstruct_samples.py    # representative-sample reconstruction
│   ├── reconstruct_test_rmse.py  # per-test-sample RMSE boxplot + table
│   ├── plot_rmse_iou_boxplot.py  # RMSE/IoU boxplots (local redraw, no PyTorch)
│   ├── latent_load_curves.ipynb  # loading-curve analysis
│   ├── latent_load_curves.png    # loading-curve plot
│   ├── config.yaml               # training config
│   ├── run_*.sh                  # launch scripts
│   ├── __init__.py               # package marker
│   └── README.md                 # notes
│
├── label_mini_program/           # manual annotation tool
│   ├── angle_labeler.py          # OpenCV interactive angle labeling
│   ├── centroid_overlay.py       # centroid overlay
│   └── label_program_guide.ipynb # labeling guide
│
├── downstream_analysis_scr/      # downstream analysis scripts
│   ├── 1.PCA_analyze.R           # PCA dimensionality reduction
│   ├── 2.correlation heatmap.R   # correlation heatmap
│   └── 3.seed_size_r2.ipynb      # seed-size reproducibility evaluation
│
└── project_figs/                 # figures & demos
    ├── workflow_figure.png       # overall workflow diagram
    ├── Main Panel Interaction.gif # main-panel interaction demo
    └── VAE_latent_explorer.gif   # VAE latent-explorer demo
```

---

## 4. Requirements

The pipeline uses **3 isolated conda environments**, because YOLO (ultralytics), SAM2, and PaddleOCR have conflicting dependencies and cannot share one environment:

| Environment file | Python | Key deps | Purpose | Hardware |
|------------------|--------|----------|---------|----------|
| `yolo_environment.yml` | 3.10 | PyTorch 2.4.1, ultralytics 8.3.14 | YOLO11/11x detection, YOLOv8n digits | CUDA 12.1 |
| `SAM2_environment.yml` | 3.10 | PyTorch 2.5.0, sam-2 1.0, ultralytics 8.3.31 | SAM2 segmentation, measurements (numpy/shapely/pandas/cv2) | CUDA 12.1 |
| `paddle_environment.yml` | 3.10.20 | paddlepaddle-gpu 2.5.2, paddleocr 2.7.0.3 | PaddleOCR label text | CUDA 12.1 |

**Hardware**: NVIDIA GPU (CUDA 12.1, ≥ 12 GB VRAM recommended; SAM2 Hiera-L is large).

> The desktop app `grainprofiler.exe` has **no** such requirements — it runs on any Windows 10/11 machine without Python/GPU.

---

## 5. Installation

```bash
# 1) clone / unzip this repository
cd seed_project_v1.0

# 2) create the three conda environments
conda env create -f yolo_environment.yml   -n yoloenv
conda env create -f SAM2_environment.yml   -n SAM2
conda env create -f paddle_environment.yml -n paddle

# 3) install the SAM2 repo (the segmentation stage depends on its source)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
# download sam2.1_hiera_large.pt into sam2/checkpoints/
```

---

## 6. Quick Start

### 6.1 Configuration

Edit `pipeline/config.yaml`, or (recommended) copy the template and only override machine-specific paths:

```bash
cp pipeline/env.local.yaml.example pipeline/env.local.yaml
# edit env.local.yaml and fill in:
#   environments.* (the three environment Python paths)
#   runtime.gpu_device
#   input.image_dir (input tray-photo directory)
#   models.* (model weight paths)
#   output.base_dir (result output directory)
```

`env.local.yaml` is recursively merged over `config.yaml`; scientific settings (detection thresholds, measurement parameters, calibration, …) stay in `config.yaml`.

### 6.2 Run the Pipeline

```bash
# full run (8 stages in order, across 3 environments)
conda activate SAM2
cd pipeline
python main.py config.yaml

# run a single stage
python main.py config.yaml --stage detection

# resume from a stage
python main.py config.yaml --from-stage segmentation
```

### 6.3 Use the GrainProfiler Desktop App

**Desktop software for rapid result retrieval:**

1. Find `grainprofiler.exe` (Windows program, ~148 MB) in the repository root.
2. Double-click `grainprofiler.exe` to run — **no Python, GPU, or conda required**.
3. In the app, click "Open Folder" and select a pipeline result directory (the folder containing `measurements.csv`).
4. Browse samples, inspect kernel outlines and measurements, filter, and export CSVs.

See [§11](#11-grainprofiler-desktop-app) for demo recordings (main-panel interaction, VAE latent explorer).

> If Windows shows "Windows protected your PC", click "More info" → "Run anyway".

**Developers (optional):**

```bash
# run from source (requires PySide6)
pip install PySide6
python grainprofiler/main.py

# rebuild the exe
pip install pyinstaller
pyinstaller grainprofiler.spec   # output at dist/grainprofiler.exe
```

---

## 7. The 8 Pipeline Stages

```
raw tray photo (.jpg, 5408×4056)
    │
    ▼
[Stage 0: pre_ocr]    YOLO11 → label + scale-screen ROIs; YOLOv8n → seven-segment digits
[Stage 1: ocr]        PaddleOCR → plant_name + weight_g + tray calibration (mm/px)
[Stage 2: detection]  YOLO11x → kernel boxes + 4-level cascade filter
[Stage 3: segmentation] SAM2.1 Hiera-L → kernel binary masks + RGB crops (neutral-gray bg)
[Stage 4: measurements] ResNet → directed main axis + 10+ morphological traits + 100-dim width profile
[Stage 5: shapes]     per-sample median 100-dim width profile + representative-shape plots
[Stage 6: vae_encode] β-VAE encoding → 5-dim latent traits (latent_traits.csv)
[Stage 7: assembly]   merge into final_output_individual / plant_median.csv
    │
    ▼
downstream: PCA / β-VAE / correlation heatmaps → GWAS
```

### Inter-stage data flow (bridge files)

| Stage | Output file | Contents | Consumed by |
|-------|-------------|----------|-------------|
| 0→1 | `yolo_label_weight_boxes.json` | label/scale-screen boxes + digits + weight_g | OCR |
| 1→4,5,7 | `metadata.csv` | plant_name + weight_g + mm/px calibration | measurement/shape/assembly |
| 2→3,4 | `yolo_bounding_boxes.json` | kernel boxes (filter reasons + accepted list) | segmentation/measurement |
| 3→4,5 | `subimages/` + `masks_binary/` | per-kernel RGB crops (gray bg) + binary masks | measurement/shape |
| 3→4,5 | `contours/` + `kernel_contours.json` | outline points (per-image + single-file compat) | measurement/shape |
| 4→5 | `axis_results.json` | per-kernel bottom/top endpoints + axis length | shape |
| 4→7 | `measurements.csv / .parquet` | per-kernel morphology table | assembly |
| 5→7 | `median_outlines.csv` | per-plant median table | assembly |
| 5→6 | `rep_width_profiles.txt` | per-plant 100-dim width profile | VAE/PCA |
| 6→GWAS | `latent_traits.csv` | plant_id + latent_1..5 | GWAS |

### Stage 2 — 4-level cascade filter (deterministic rules)

1. **Tray constraint**: box center must be inside the tray and overlap > 55%; edge boxes clipped to tray ROI.
2. **Aspect ratio**: AR < 4.0 (drop thin artifacts).
3. **Statistical size outlier**: relative to in-tray median, 0.35–3.0× (area), 0.45–2.2× (side length).
4. **Nested-overlap suppression**: two boxes overlap > 86% and area ratio > 1.15 → keep the box closer to the median size.
   (Plus a multi-center filter: drop large boxes containing centers of multiple other boxes.)

### Stage 4 — directed axis and morphology

- Mask cleanup → centroid (`cv2.moments`) → equal-arc contour resampling (360 points).
- **ResNet predicts a directed unit vector (cosθ, sinθ)** → line through centroid intersects the contour → bottom/top endpoints.
- **Multi-cue scoring to determine crown/pedicel**: endpoint sharpness (multi-scale windows), tip taper, width monotonicity, area symmetry, width symmetry, LAB color symmetry — weighted fusion (area 0.25 / width 0.30 / color 0.25 / length 0.10 / endpoint 0.10).
- Measurements: main-axis length, max width, W25/W50/W75, area (Shoelace), perimeter, circularity (4πA/P²), length/width ratio, eccentricity.
- Width profile: perpendicular widths at 100 equally spaced positions along the axis (Shapely line–contour intersections).
- Round-kernel filter: circularity > 0.90 → flagged "Round" (length/width skipped).
- Physical calibration: HSV detection of the 100 mm blue/green tray → mm/px (~0.052–0.054).

---

## 8. Model Zoo

| # | Model | Size | Input | Output | Env |
|---|-------|------|-------|--------|-----|
| 1 | YOLO11 | 56.8M | tray photo | label + weight_screen boxes | yoloenv |
| 2 | YOLO11x | 56.8M | tray photo | kernel boxes (mAP50=0.995) | yoloenv |
| 3 | YOLOv8n | - | weight-screen crop 1280×512 | 0-9 digit sequence | yoloenv |
| 4 | SAM2.1 Hiera-L | - | full image + YOLO box prompts | binary masks | sam2 |
| 5 | ResNet (custom) | ~2.2M | RGB kernel 256×256 | (cosθ, sinθ) directed axis | base |
| 6 | β-VAE | ~19K | 100-dim width profile | 5-dim latent vector | base |

### Key design decisions (WHY)

1. **Seven-segment digits use YOLOv8n, not OpenCV segmentation**: segment gaps are unstable under varying light/angle; end-to-end 10-class small-object detection is naturally robust.
2. **SAM2 crops use neutral gray (128,128,128) background**: black kernels vanish on black, white kernels on white; mid-gray is the best compromise across the full color range, and lands near (0,0,0) after ImageNet normalization.
3. **Orientation uses ResNet regression, not PCA/image moments**: moments/PCA only give an undirected axis; vector regression avoids the angle wrap-around and correctly supports direction-dependent W25/W50/W75.
4. **β-VAE rather than pure PCA**: PCA is linear-orthogonal; β-VAE learns a nonlinear manifold. β=0.001 keeps reconstruction fidelity, and 5 latent dims suit GWAS better than 100 raw dims.

### ResNet training details

- Preprocessing: `square` (gray-pad short side → resize 256×256).
- Loss: `1 - cosine_similarity(pred, target)` (+ λ·|||pred||−1|², norm_loss_weight=0).
- AdamW (lr=1e-4, wd=1e-4), CosineAnnealingLR, early-stopping patience=30, batch 32, seed 42, 80/10/10 split.
- Augmentation: brightness/contrast jitter ±4%, Gaussian noise 30% prob σ=8px.

### β-VAE training details

- Architecture: 1D-CNN Encoder (100→50→25→5) + Decoder (5→25→50→100).
- Loss: MSE(recon, x) + β·KL(𝒩(μ,σ²)‖𝒩(0,I)), β=0.001.
- AdamW (lr=5e-4), ReduceLROnPlateau, early-stopping patience=50.
- Input standardization: per-position Z-score; output `latent_traits.csv` feeds GAPIT/GEMMA/FarmCPU directly.

---

## 9. Model Weights

| Weight | Path (config.yaml) | Distributed? | Notes |
|--------|--------------------|--------------|-------|
| YOLO11 label/scale-screen | `models.label_weight_yolo` | ❌ | train or obtain yourself |
| YOLO11x kernels | `models.yolo_detection` | ❌ | train or obtain yourself |
| YOLOv8n digits | `models.weight_digit_yolo` | ❌ | train or obtain yourself |
| SAM2.1 Hiera-L | `models.sam2_checkpoint` | ❌ | download from [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| ResNet directed axis | `models.resnet_axis` | ❌ | train with `resnet/train_resnet_angle.py` |
| β-VAE | `pipeline/vae/vae_checkpoint.pt` | ✅ included | tiny (~19K params) |
| VAE Decoder ONNX | `onnx_models/` | ✅ included | real-time decode in the GUI latent window |

> Training scripts live in `resnet/` and `vae/`; YOLO training follows the standard ultralytics workflow. Configure weight paths in `config.yaml` (override via `env.local.yaml`).

---

## 10. Input / Output Data Format

### Input

- Tray photos (`.jpg`, 5408×4056), one per ear sample.
- Imaging platform: GP-2000 overhead camera, fixed 310 mm height, unified exposure/ISO/white balance; 3D-printed 100 mm blue/green weighing tray as the physical calibration reference (~19 px/mm); matte black rubber mat background; kernels placed randomly; label (sample ID / QR code) in view.

### Output (key files)

| File | Granularity | Key fields |
|------|-------------|------------|
| `final_output_individual.csv` | per-kernel | kernel_name, plant_name, weight_g, length_mm, max_width_mm, width_25/50/75pct_mm, area_mm2, perimeter_mm, circularity, length_width_ratio, eccentricity |
| `final_output_plant_median.csv` | per-plant | plant_name, weight_g, n_kernels, median_* series, weight_per_kernel_g |
| `measurements.csv / .parquet` | per-kernel | full morphology + diagnostic fields |
| `rep_width_profiles.txt` | per-plant | plant_name + 100-dim width profile (PCA/VAE input) |
| `latent_traits.csv` | per-plant | plant_id + latent_1..5 (GWAS input) |
| `metadata.csv` | per-image | image_name, plant_name, weight_g, mm_per_px, tray info, raw OCR text |
| `axis_results.json` | per-kernel | bottom/top endpoints, axis length, shape label |
| `subimages/` + `masks_binary/` | per-kernel | RGB crops (gray bg) + binary masks |
| `contours/` | per-image | per-kernel outline points (GUI lazy loading) |

---

## 11. GrainProfiler Desktop App

GrainProfiler is a **read-only data browser** (no GPU/conda/pipeline; just double-click the exe), **Windows 10/11 only**.

**Three-pane workspace**: left (sample list / filter) → middle (tray photo + contour overlay + half-width profile chart) → right (metadata + measurement table).

**Core panels**:

| Panel | Function |
|-------|----------|
| SampleView | interactive tray photo, kernel contour overlay, hover highlight, click select, wheel zoom, ruler tool |
| SampleInfoPanel | metadata (editable, locked), per-plant statistics, measurement table (sort/filter/export CSV) |
| KernelDetailDialog | per-kernel crop + contour + axis endpoints + full metrics |
| MedianChart | half-width profile chart: gray = single kernel, blue = median, hover red highlight |
| PCAPanel | numpy PCA scatter + PC1-5 score table + representative-shape preview |
| VAELatentWindow | ONNX real-time decode: 5-dim latent sliders ↔ reconstructed curve |
| SampleAnalysisWindow | trait distribution: sorted scatter + regression line + multi-trait windows |

**Demo recordings**:

- Main-panel interaction: ![GrainProfiler main-panel interaction demo](<project_figs/Main Panel Interaction.gif>)
- VAE latent explorer: ![GrainProfiler VAE latent-explorer demo](project_figs/VAE_latent_explorer.gif)

**Cross-linking**: overlay kernel outline ↔ median width line ↔ measurement table row are fully synchronized (hover/click highlight each other).

**Data-loading optimizations**: Parquet-first (1-2 s vs CSV 10-15 s), contour LRU lazy loading (last 5 images cached).

**Packaging**: `pyinstaller grainprofiler.spec` (excludes torch/ultralytics/paddle/sklearn to slim the exe).

---

## 12. Manual Annotation Tool

`label_mini_program/angle_labeler.py` is an OpenCV interactive angle labeler for annotating the "crown → pedicel" directed angle of each kernel:

```bash
python label_mini_program/angle_labeler.py
# centroid anchor + rotatable arrow, keyboard fine-tuning,
# outputs cosθ/sinθ and auxiliary cos2θ/sin2θ
```

The training set contains ~8,300 labeled kernels (7,646 after removing ambiguous ones), used to train the directed-axis ResNet.

---

## 13. FAQ

**Q1: `ModuleNotFoundError: No module named 'utils'`**
Each stage script adds `pipeline/` to the search path via `sys.path.insert`. If this error appears, confirm the script has `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` at the top, before any local import.

**Q2: Paths are wrong on a different machine**
Override machine-specific paths (environments / input / models / output) in `pipeline/env.local.yaml`; do not edit the scientific settings in `config.yaml`.

**Q3: Empty/NaN values in the mm columns**
Usually one image's calibration failed. Check the `calibration.fixed_mm_per_px` fallback, or whether the tray color is within `calibration.tray_color_ranges`. v1.0 adds "plant-dominant mm_per_px filling" in `representative_shape.py`.

**Q4: Black-kernel segmentation/measurement issues**
SAM2 crop backgrounds already use neutral gray (128,128,128), so black kernels do not disappear; if issues remain, check `segmentation.padding` and mask-cleanup parameters.

**Q5: The GUI cannot open a result directory**
Confirm the directory contains `measurements.csv` (or `.parquet`) and `metadata.csv`; contour overlays need the `contours/` directory or `kernel_contours.json`.

---

## 14. Downstream Analysis Scripts

The `downstream_analysis_scr/` directory provides 3 downstream analysis scripts actually used in this project, taking the shape/morphology data output by the pipeline to perform dimensionality reduction, correlation visualization, and seed-size reproducibility evaluation.

| Script | Language | Input | Output | Purpose |
|--------|----------|-------|--------|---------|
| `1.PCA_analyze.R` | R (ggplot2 / patchwork / ggrepel / dplyr) | `rep_width_profiles.txt` (per-plant 100-dim width profile) | PCA 2D scatter, variance-explained table, PC1–5 scores, loading curves | PCA dimensionality reduction & visualization of the 100-dim profiles |
| `2.correlation heatmap.R` | R (GGally / hexbin / ggplot2) | merged table: `sample_id` + PC1–5 + 6 morphological traits | correlation heatmap PNG + correlation matrix CSV | publication-level correlation matrix of PCs vs. morphological traits (hexbin + significance stars) |
| `3.seed_size_r2.ipynb` | Python (pandas / sklearn / seaborn) | `final_output_individual.csv` | R²/RMSE summary + per-trait scatter plots | reproducibility evaluation of seed size at central vs. off-central tray positions |

**Script details**:

1. **`1.PCA_analyze.R`**: runs PCA (`prcomp`, center + scale) on each plant's 100-dim width profile, computes per-component variance explained, outputs PC1–5 scores and loadings; draws a 2D scatter with marginal density plots (patchwork layout), auto-labels the farthest sample in each quadrant, and plots PC1/PC2 loading curves along the kernel's relative position (0–1).
2. **`2.correlation heatmap.R`**: combines PCA scores with morphological traits and uses `ggpairs` to draw a publication-level correlation matrix — hexbin + linear regression in the upper triangle, Pearson r + significance stars (`***`/`**`/`*`) in the lower triangle, and density plots on the diagonal — at 600 dpi, also exporting the correlation matrix CSV.
3. **`3.seed_size_r2.ipynb`**: parses `kernel_name` to derive `test_id` and replicate `rep`, treats rep=1 (central position) as ground truth and rep 2/3 (off-central position) as predictions, computes R²/RMSE for 6 traits (length, max-width, area, perimeter, eccentricity, circularity), and plots scatter with a y=x reference line (example: length R²≈0.97, perimeter R²≈0.98, circularity R²≈0.83).

> **Notes**:
> 1. The scripts retain the author's absolute paths (`C:/Users/HP/...`); replace them with your own input/output paths before running.
> 2. The input to `2.correlation heatmap.R` is **not** the direct output of `1.PCA_analyze.R`: it needs an extra merge step that combines the PCA scores (PC1–5) with per-plant morphological traits (length / max width / area / perimeter / circularity / aspect ratio — e.g. from `final_output_plant_median.csv`) into a table with a `sample_id` column plus 11 numeric columns.

---

## 15. Copyright & License

**Copyright © 2026 ChangQGuo. All rights reserved.**

This project is **temporarily closed-source and pre-publication**:

- Copying, modifying, redistributing, commercial use, or any other use **without the author's written permission** is prohibited;
- The associated paper has not been published yet — do not disclose any code, data, models, or results;
- For usage or collaboration, please contact the author (ChangQGuo, guocq03@outlook.com).

---
