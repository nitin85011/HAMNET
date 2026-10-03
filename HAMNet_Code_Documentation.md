# HAMNet Crowd Density Estimation - Code Documentation

## 1. Overview
The notebook implements an image-based crowd density estimation pipeline. It reads Roboflow/RetinaNet-style bounding-box annotations, converts boxes to head-centre points, generates Gaussian density maps, performs point-aware augmentation, trains HAMNet, tunes inference settings on validation data, and evaluates crowd-counting performance on a held-out test set.

The notebook contains several successive implementations: an initial VGG16-BN version, a v3 enhanced VGG16-BN version, a v4 version with optional ConvNeXt-Tiny and tiled inference, and a final model-comparison implementation.

## 2. Notebook Structure
| Cell | Purpose |
|---|---|
| 0 | Google Drive mounting |
| 1 | Dataset extraction |
| 2 | Dataset verification |
| 3 | Annotation CSV inspection |
| 4 | Dependency installation |
| 5 | Initial HAMNet implementation |
| 6 | Empty |
| 7 | HAMNet v3 |
| 8 | HAMNet v4 |
| 9 | Model comparison |
| 10 | Empty |

## 3. Dataset
Expected structure:
```text
/content/crowd_data/
├── train/
├── valid/
└── test/
```
Each split contains images and `_annotations.csv`.

Bounding boxes are converted to head-centre points. Gaussian kernels are placed at these points to form density maps. Later versions support box-size, KNN and fixed sigma modes.

## 4. Data Augmentation
The implementation supports:
- random scaling
- random cropping
- horizontal flipping
- rotation
- brightness/contrast/saturation adjustment
- gamma adjustment
- noise perturbation in earlier variants

Point coordinates are transformed together with images.

## 5. HAMNet Architecture
### Backbone
The v3 implementation uses ImageNet-pretrained VGG16-BN. B3 and B4 features are fused and reduced to 256 channels.

The v4 implementation supports ConvNeXt-Tiny and VGG16.

### Multi-Scale Fusion
Four 3x3 dilated convolutions use dilation rates 1, 2, 3 and 4. Their outputs are concatenated, projected to 256 channels and combined with a residual connection.

### CBAM
Channel attention uses global average/max pooling and an MLP. Spatial attention uses average/max channel projections followed by a 7x7 convolution.

### Density Head
The VGG-based v3 head uses three PixelShuffle x2 stages. The v4 model predicts a stride-8 density map.

Count:
`predicted_count = sum(predicted_density) / density_scale`

## 6. Loss
The v3 composite loss includes:
- L1 density loss
- MSE density loss
- SSIM loss
- global count loss
- multi-scale patch-count loss
- batch bias loss

The v4 loss replaces SSIM with relative count loss.

## 7. Training
The v3 configuration uses:
- crop size: 512x512
- maximum side: 1536
- batch size: 8
- epochs: 120
- patience: 40
- head LR: 2e-4
- backbone LR multiplier: 0.1
- weight decay: 1e-4
- warm-up: 3 epochs
- gradient clipping: 5
- AMP: enabled
- EMA decay: 0.995
- density scale: 100

Optimizer: AdamW.

## 8. Inference
Validation data is used to select:
- scale: 1.0, 1.25 or 1.5
- horizontal flip: enabled/disabled
- optional count calibration

Calibration:
`alpha = sum(gt * pred) / sum(pred^2)`

The selected configuration is then applied to the test set.

The v4 implementation also supports overlapping tiled inference.

## 9. Metrics
The notebook reports:
- MAE
- MSE
- RMSE
- MAPE
- counting accuracy

The exact formulas should be stated in any paper because crowd-counting literature sometimes uses the term MSE differently.

## 10. Model Comparison
The comparison cell evaluates:
- MCNN
- AlexNet
- VGG16
- ResNet50
- CSRNet
- HAMNet

It records parameter count, MAE, MSE, RMSE, MAPE, accuracy, inference time and training time.

## 11. Outputs
Typical outputs include:
- HAMNet checkpoint `.pth`
- training history
- per-image prediction CSV
- comparison JSON files
- `summary.csv`
- `per_image_predictions.csv`
- `comparison.png`

## 12. Reproducibility
The later implementations use seed 42. For reproducible experiments:
1. Select one implementation version.
2. Run it from a clean runtime.
3. Record the exact configuration.
4. Record GPU/software versions.
5. Keep the checkpoint and test prediction CSV.
6. Tune inference parameters only on validation data.

## 13. Important Note
The notebook contains multiple complete implementations. Running cells from different versions out of order can redefine classes, functions and configuration values. For publication experiments, keep the selected version in a separate notebook or Python module and run it from top to bottom.
