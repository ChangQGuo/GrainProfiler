# ResNet-Style Kernel Angle Regression

This folder trains a small custom CNN for directed maize-kernel main-axis labels.

Input: one RGB kernel crop.

Output: `cos(theta), sin(theta)`.

Architecture:

```text
RGB image
Conv stem
Residual Block x 2
Residual Block x 2
Residual Block x 3
Residual Block x 2
Global Average Pooling
MLP head
cos(theta), sin(theta)
```

The intended input is a tight RGB crop around one kernel. By default, training is RGB-only:

1. read the single-kernel RGB crop,
2. keep aspect ratio,
3. pad the short side with black to make a square,
4. resize to `256x256`.

The `crop` preprocessing mode is only a fallback for old `640x640` padded crops, where a binary mask is needed to remove excessive background.

Train:

```powershell
cd "C:\Users\HP\Desktop\Academic Presentation\seed_project"
cd a_aguo_test_new\resnet
..\..\.venv_ssocr\Scripts\python.exe train_resnet_angle.py config.yaml
```

Useful variants:

```powershell
.\.venv_ssocr\Scripts\python.exe a_aguo_test_new\resnet\train_resnet_angle.py --imgsz 320 --batch-size 32
.\.venv_ssocr\Scripts\python.exe a_aguo_test_new\resnet\train_resnet_angle.py --preprocess mask --imgsz 320
.\.venv_ssocr\Scripts\python.exe a_aguo_test_new\resnet\train_resnet_angle.py --preprocess none --imgsz 640 --batch-size 8
.\.venv_ssocr\Scripts\python.exe a_aguo_test_new\resnet\train_resnet_angle.py --noise-prob 0.30 --noise-std 8.0
```

`--noise-prob 0.30` applies Gaussian-noise augmentation to about 30% of train samples each epoch. Validation and test samples are not augmented.

Run outputs are saved like YOLO experiments under:

```text
a_aguo_test_new/resnet/runs/angle/train
a_aguo_test_new/resnet/runs/angle/train2
...
```

Main outputs:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `test_metrics.json`
- `test_predictions.png`
- `splits/train.csv`, `splits/val.csv`, `splits/test.csv`

Predict and save overlays:

```powershell
.\.venv_ssocr\Scripts\python.exe a_aguo_test_new\resnet\predict_resnet_angle.py `
  --weights a_aguo_test_new\resnet\runs\angle\train\weights\best.pt `
  --images a_aguo_test_new\image_data\subimages `
  --masks a_aguo_test_new\image_data\masks_binary `
  --csv a_aguo_test_new\image_data\angle_labels_directed.csv `
  --overlay-dir a_aguo_test_new\resnet\runs\angle\train\pred_overlays
```
