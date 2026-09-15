<div align="center">

# GrainProfiler

### High-throughput 2D maize-kernel phenotyping

[中文说明](README_CN.md)

</div>

> Tray photo in → 8-stage processing → per-kernel and per-plant phenotype data out.
> GrainProfiler also includes a Windows desktop viewer for inspecting pipeline results.

The project has processed 15,287 maize samples and 386,386 kernels across 61 germplasm accessions. The repository contains the reproducible pipeline source, training code, the desktop viewer, and deployment assets.

![GrainProfiler workflow](project_figs/Figures_github-01.png)

## Choose a route

### Just inspect results — Windows

1. Download or clone this repository.
2. Double-click `grainprofiler.exe`.
3. Click **Open Folder** and select a pipeline result directory containing `measurements.csv` (or `measurements.parquet`) and `metadata.csv`.

The viewer does not require Python, CUDA, conda, or the pipeline environments.

### Run the full pipeline — GPU + conda

Use this route to process new tray photos or rerun individual stages.

## Pipeline requirements

- NVIDIA GPU with CUDA 12.1; at least 12 GB VRAM is recommended for SAM2 Hiera-L.
- Conda or Miniconda.
- Linux, WSL, or another environment that can run the supplied Python scripts.
- Input tray photos in `.jpg` format. The reference setup uses 5408 × 4056 images.

The pipeline uses three isolated environments because the YOLO, SAM2, and PaddleOCR dependencies are not reliably compatible in one environment.

| Environment file | Environment name | Used for |
|---|---|---|
| `yolo_environment.yml` | `yoloenv` | YOLO11/YOLO11x detection and YOLOv8n digit detection |
| `SAM2_environment.yml` | `SAM2` | SAM2 segmentation and morphology measurement |
| `paddle_environment.yml` | `paddle` | PaddleOCR label recognition |

## Installation

```bash
git clone https://github.com/ChangQGuo/GrainProfiler.git
cd GrainProfiler

conda env create -f yolo_environment.yml   -n yoloenv
conda env create -f SAM2_environment.yml   -n SAM2
conda env create -f paddle_environment.yml -n paddle

# SAM2 source is required by Stage 3.
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
cd ..
# Download sam2.1_hiera_large.pt into the SAM2 checkpoint directory.
```

Download or train the model weights listed in `pipeline/config.yaml`. The VAE checkpoint and the ONNX decoder used by the desktop viewer are included; the YOLO, ResNet, and SAM2 checkpoints must be supplied separately unless they are available from the associated model release.

## Configure and run

Do not edit machine-specific paths directly in `pipeline/config.yaml`. Create a local override file:

```bash
cp pipeline/env.local.yaml.example pipeline/env.local.yaml
```

Set the following values in `pipeline/env.local.yaml`:

- the Python executable for `yoloenv`, `SAM2`, and `paddle`;
- `runtime.gpu_device`;
- the input image directory;
- YOLO, ResNet, and SAM2 checkpoint paths;
- the output directory.

Run from the `pipeline` directory:

```bash
conda activate SAM2
cd pipeline

# Full run: pre_ocr → ocr → detection → segmentation → measurements
#          → shapes → vae_encode → assembly
python main.py config.yaml

# Run one stage
python main.py config.yaml --stage detection

# Resume from a stage using existing intermediate files
python main.py config.yaml --from-stage segmentation
```

## The 8 stages

| Stage | What it does | Main output |
|---|---|---|
| 0 · `pre_ocr` | Detects the label and scale-screen ROIs; detects scale digits | `yolo_label_weight_boxes.json` |
| 1 · `ocr` | Reads sample names, parses weight, and estimates tray calibration | `metadata.csv` |
| 2 · `detection` | Detects kernels and applies tray, shape, size, and overlap filters | `yolo_bounding_boxes.json` |
| 3 · `segmentation` | Uses SAM2 box prompts to create one mask per kernel | `masks_binary/`, `subimages/`, `contours/` |
| 4 · `measurements` | Predicts the directed axis and calculates morphology | `axis_results.json`, `measurements.csv` |
| 5 · `shapes` | Builds 100-point normalized width profiles | `rep_width_profiles.txt`, `median_outlines.csv` |
| 6 · `vae_encode` | Encodes each plant profile into five latent traits | `latent_traits.csv` |
| 7 · `assembly` | Merges image, kernel, plant, weight, and trait data | `final_output_*.csv` |

## Important outputs

- `final_output_individual.csv`: one row per kernel with length, width, area, perimeter, circularity, eccentricity, and related traits.
- `final_output_plant_median.csv`: one row per plant/sample with median morphology and weight information.
- `rep_width_profiles.txt`: one 100-point plant-level width profile per sample; input for PCA and VAE.
- `latent_traits.csv`: five VAE traits per sample; suitable as phenotype input for downstream GWAS workflows.
- `metadata.csv`: sample identity, weight, tray dimensions, and `mm_per_px` calibration.
- `axis_results.json`: per-kernel axis endpoints and measurement status.
- `subimages/`, `masks_binary/`, and `contours/`: intermediate visual and geometric data used by the viewer.

## GrainProfiler desktop viewer

`grainprofiler.exe` is a Windows 10/11 application for result inspection. It can:

- browse samples and tray photographs;
- overlay kernel contours and axis endpoints;
- inspect per-kernel measurements and plant-level median profiles;
- explore PCA and the five-dimensional VAE latent space;
- filter and export table data.

The source is in `grainprofiler/`. To run it from source:

```bash
pip install PySide6
python grainprofiler/main.py
```

## Repository layout

```text
pipeline/                 8-stage processing pipeline
grainprofiler/             desktop viewer source
resnet/                    directed-axis model and training tools
vae/                       β-VAE training and interpretation tools
downstream_analysis_scr/  PCA, correlation, and reproducibility analysis
project_figs/              workflow figure and viewer demos
*.yml                     conda environment definitions
```

## Model summary

1. YOLO11 — label and scale-screen regions
2. YOLO11x — kernel detection
3. YOLOv8n — seven-segment scale digits
4. SAM2.1 Hiera-L — instance masks
5. Custom ResNet — directed kernel axis `(cos θ, sin θ)`
6. β-VAE — 100-point profiles to five latent traits

Training and evaluation scripts are kept in `resnet/` and `vae/`. Scientific settings such as detection thresholds and measurement parameters are in `pipeline/config.yaml`; machine-specific paths belong in `pipeline/env.local.yaml`.

## License

See [LICENSE](LICENSE). This is pre-publication research software. Please contact the author before redistributing the code, data, or model files.
