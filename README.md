HAMNet Crowd Density Estimation
Code Documentation
Technical documentation for the uploaded HAmnet.ipynb notebook

1. Overview
The notebook implements a deep-learning pipeline for image-based crowd density estimation. It reads Roboflow/RetinaNet-style bounding-box annotations, converts bounding boxes to head-centre points, generates Gaussian density maps, applies point-aware augmentation, trains HAMNet, selects inference settings using the validation set, and evaluates crowd-counting performance on a held-out test set.
The notebook contains several successive implementations. The later cells represent more developed versions of the method, including the v3 VGG16-BN implementation, a v4 implementation with optional ConvNeXt-Tiny and tiled inference, and a final model-comparison cell covering MCNN, AlexNet, VGG16, ResNet50, CSRNet, and HAMNet.
2. Notebook Structure
Cell	Purpose	Implementation
0	Google Drive setup	Mounts /content/drive
1	Dataset extraction	Unzips the Roboflow/RetinaNet dataset
2	Dataset verification	Checks train/valid/test images and _annotations.csv
3	Annotation inspection	Inspects CSV headers and sample rows
4	Dependencies	Installs scipy and pandas
5	HAMNet v1	VGG16-BN + multi-scale fusion + CBAM + PixelShuffle
6	Empty cell	No implementation
7	HAMNet v3	Enhanced VGG16-BN version with EMA, composite loss and validation tuning
8	HAMNet v4	Optional ConvNeXt-Tiny, stride-8 density, tiled inference and relative count loss
9	Model comparison	Compares MCNN, AlexNet, VGG16, ResNet50, CSRNet and HAMNet
10	Empty cell	No implementation

3. Environment and Dependencies
The notebook is written for Google Colab and uses a CUDA-capable PyTorch environment when available.
Package / Component	Role
Python	Main programming language
PyTorch	Model definition, training and tensor operations
TorchVision	Pretrained VGG16-BN/ConvNeXt model components
NumPy	Array and numerical operations
SciPy	KD-tree based nearest-neighbour calculations
Pillow	Image loading and image transformations
pandas / csv	Annotation and tabular-data handling
Matplotlib	Training curves and comparison plots
Google Colab / Drive	Execution environment and checkpoint storage

The notebook explicitly installs scipy and pandas in the dependency cell. PyTorch, TorchVision and Pillow are expected to be available in the Colab environment.
4. Dataset Organization
The notebook expects the extracted dataset at /content/crowd_data with three splits: train, valid and test. Each split contains images and an _annotations.csv file.
/content/crowd_data/
├── train/
│   ├── images
│   └── _annotations.csv
├── valid/
│   ├── images
│   └── _annotations.csv
└── test/
    ├── images
    └── _annotations.csv
The CSV parser is deliberately tolerant of alternative column names for filename and bounding-box coordinates. It searches candidate names such as filename/image, xmin/x_min/x1, ymin/y_min/y1, xmax/x_max/x2 and ymax/y_max/y2.
5. Dataset Preparation
5.1 Bounding-box to point conversion
Each person annotation is represented by a bounding box. The implementation converts each valid box to a point located at the box centre. These points become the supervision locations for density-map generation.
For a bounding box (x_min, y_min, x_max, y_max), the centre is conceptually:
x_c = (x_min + x_max) / 2
y_c = (y_min + y_max) / 2
5.2 Density-map generation
The density-map generator places a normalized Gaussian around every head-centre point. The implementation supports fixed, KNN-based and box-size-based sigma selection in the later notebook versions. In the v3/v4 configuration, the default mode is box-based sigma.
sigma = box_sigma_ratio * sqrt(box_width * box_height)
sigma is clipped to [min_sigma, max_sigma]
density(x) = sum_i Gaussian_i(x)
Each Gaussian is normalized before being added to the density map. The resulting density map is then scaled by the configured density scale so that summing the prediction and dividing by the same scale produces the estimated number of people.
6. Data Augmentation
The augmentation pipeline is point-aware. Image transformations are applied together with the corresponding point-coordinate transformations so that the supervision remains spatially aligned.
Random rescaling in the later versions.
Random crop with point filtering and coordinate translation.
Horizontal flipping with x-coordinate transformation.
Small random rotations with corresponding point rotation.
Brightness, contrast and saturation changes.
Gamma adjustment in the earlier/later implementation variants.
Gaussian/noise-style image perturbation in the earlier implementation.
The v3 configuration uses a 512×512 training crop, maximum image side of 1536 pixels, scale range 0.85–1.5, and scale augmentation probability 0.7. The v4 configuration uses 640×640 crops and repeat=3 random crops per image per epoch.
7. HAMNet Architecture
The core HAMNet pipeline consists of four major neural components: a backbone, multi-scale dilated feature fusion, channel-spatial attention, and a density regression head.
7.1 Backbone
The v3 and comparison implementations use VGG16-BN with ImageNet-pretrained weights. Blocks B1–B4 are used to obtain hierarchical features. B3 and B4 are fused and reduced to 256 channels using a 1×1 convolution. Batch-normalization freezing is supported in the later implementation.
The v4 implementation adds an optional ConvNeXt-Tiny backbone while retaining VGG16 as an available option.
7.2 Multi-Scale Dilated Fusion
The MultiScaleFusion module applies four parallel 3×3 convolutions with dilation rates 1, 2, 3 and 4. The branch outputs are concatenated, projected back to 256 channels with a 1×1 convolution, and combined with a residual connection.
F_ms = Fuse([Conv_d=1(F), Conv_d=2(F), Conv_d=3(F), Conv_d=4(F)]) + F
7.3 CBAM Attention
The CBAM module applies channel attention followed by spatial attention. Channel attention uses global average and maximum pooling followed by a shared MLP. Spatial attention uses channel-wise average and maximum maps followed by a 7×7 convolution.
7.4 Density Regression Head
The VGG-based head uses convolutional layers followed by three PixelShuffle ×2 stages, giving an overall 8× spatial upsampling from the stride-8 feature representation. A final 1×1 convolution produces one density channel, and Softplus ensures a non-negative density output.
The v4 implementation instead predicts a stride-8 density map. This reduces the number of output pixels and is paired with count-preserving downsampling of the ground-truth density map.
7.5 Count estimation
The predicted density is summed spatially and divided by the density scale to obtain the estimated crowd count.
predicted_count = sum(predicted_density) / density_scale
8. Loss Functions
The notebook contains several composite loss formulations. The v3 VGG implementation combines density reconstruction, structural similarity, global count, local patch-count and batch-bias terms.
Term	Purpose
L1	Pixel-level absolute density error
MSE	Penalizes squared density errors
SSIM	Encourages structural similarity between density maps
Global count loss	Aligns the integrated density with the ground-truth count
Patch-count loss	Constrains local/regional crowd counts at multiple scales
Bias loss	Penalizes systematic batch-level over/under-counting
Relative count loss (v4)	Balances count error across sparse and dense images

For the v3 configuration, the documented weights are w_l1=1.0, w_mse=2.0, w_ssim=0.1, w_cnt=0.5, w_patch=5.0 and w_bias=1.0. The v4 configuration removes SSIM and uses relative count loss.
9. Training Strategy
The later v3 training configuration uses the following main settings:
Parameter	v3 setting
Crop size	512 × 512
Maximum image side	1536
Batch size	8
Epochs	120
Early stopping patience	40
Head learning rate	2 × 10^-4
Backbone LR multiplier	0.1
Weight decay	1 × 10^-4
Warm-up	3 epochs
Gradient clipping	5
Mixed precision	Enabled
EMA decay	0.995
Density scale	100

The optimizer is AdamW. A warm-up plus cosine learning-rate schedule is used in the v3 implementation. The model checkpoint is updated when validation MAE improves.
10. Exponential Moving Average (EMA)
The later implementations maintain an EMA copy of the model weights. EMA provides a smoothed parameter version that can be used for validation and final evaluation. The configured decay is 0.995.
11. Inference and Test-Time Augmentation
The v3 implementation evaluates multiple inference configurations on the validation set. Candidate scales are 1.0, 1.25 and 1.5, and horizontal flipping can be enabled or disabled. A scalar count calibration factor alpha is also fitted on validation predictions.
alpha = sum(gt * pred) / sum(pred^2)
calibrated_count = alpha * predicted_count
The selected scale, flip setting and calibration factor are then applied to the test set. This keeps inference-configuration selection separate from the held-out test evaluation.
The v4 implementation additionally supports overlapping sliding-window inference. The tile size is 640 pixels and the default stride is 480 pixels, producing overlapping predictions that are averaged.
12. Evaluation Metrics
The notebook reports the following crowd-counting metrics:
Metric	Definition / interpretation
MAE	Mean absolute difference between predicted and ground-truth counts.
MSE	Mean squared count error.
RMSE	Square root of MSE.
MAPE	Mean absolute percentage error using max(ground-truth count, 1) in the denominator.
Accuracy	Counting accuracy reported by the notebook's count_metrics implementation.

The notebook's comments note an important terminology issue: some crowd-counting papers use the term MSE for what is mathematically RMSE. When reporting results in a paper, the exact formula should be stated to avoid ambiguity.
13. Model Comparison
The final comparison cell trains/evaluates six model names under a common experimental framework:
Model	Role in comparison
MCNN	Multi-column convolutional crowd-counting baseline
AlexNet	CNN baseline
VGG16	VGG-based baseline
ResNet50	Residual CNN baseline
CSRNet	Dilated-convolution crowd-counting baseline
HAMNet	Proposed architecture

The comparison code records parameter count, MAE, MSE, RMSE, MAPE, counting accuracy, inference time and training time. It reports both raw results and tuned results where scale, flip and count calibration are selected using the validation set.
14. Output Files
HAMNet checkpoints are saved to Google Drive, for example hamnet_v3_best.pth or hamnet_v4_best.pth.
Training history is saved by the later implementations.
Per-image test predictions are written to hamnet_v3_test_preds.csv or hamnet_v4_test_preds.csv.
The comparison implementation writes individual JSON result files for each model.
The comparison implementation writes summary.csv and per_image_predictions.csv.
A comparison plot is saved as comparison.png when Matplotlib plotting succeeds.
15. Execution Procedure
1.Open the notebook in Google Colab.
2.Mount Google Drive and place the dataset ZIP at the path expected by the notebook.
3.Run the dataset extraction and verification cells.
4.Inspect the annotation CSV headers before training.
5.Install any missing dependencies.
6.Choose one HAMNet implementation version rather than executing all versions sequentially.
7.Set the dataset path, GPU/CPU device, batch size and output checkpoint path.
8.Run training and monitor validation MAE/RMSE.
9.Load the best validation checkpoint.
10.Select inference scale, horizontal flip and calibration only on the validation set.
11.Run the final test evaluation and save per-image predictions.
12.For baseline comparison, run the final comparison cell separately after confirming the common data protocol.
16. Recommended Versioning Practice
The notebook contains multiple full implementations of HAMNet. Because each later cell redefines classes, functions and CFG, running cells from different versions out of order can cause the active implementation to differ from the intended one. For reproducible experiments, copy the selected version into a separate notebook or Python module and execute it from top to bottom.
Use the v3 implementation when the experiment is based on VGG16-BN, full-resolution density output, SSIM and multi-scale patch loss.
Use the v4 implementation when using ConvNeXt-Tiny/VGG16 options, stride-8 density maps and tiled inference.
Use the final cell only for the multi-model comparison protocol.
Record the exact configuration, checkpoint filename, random seed, GPU type and software versions for every experiment.
17. Reproducibility Notes
The notebook sets random seeds to 42 in the later versions.
CUDA is selected when available; otherwise the code falls back to CPU in final evaluation.
Mixed precision is configurable through the amp setting.
The crop size must be compatible with the network stride in the later implementations.
The validation set is used to select test-time scale/flip/calibration settings.
The test set should not be used to tune these inference parameters.
18. Research Methodology Summary
The implemented workflow can be summarized as:
Bounding boxes
    ↓
Head-centre points
    ↓
Scale-aware Gaussian density maps
    ↓
Point-aware augmentation + normalization
    ↓
Backbone feature extraction
    ↓
Multi-scale dilated feature fusion
    ↓
Channel-spatial attention (CBAM)
    ↓
Density regression head
    ↓
Predicted density map
    ↓
Spatial integration / density summation
    ↓
Estimated crowd count
    ↓
Validation-controlled inference + test evaluation
19. Important Implementation Notes
This documentation describes what is implemented in the uploaded notebook. It does not assume that all architectural variants shown in a methodology diagram are simultaneously active in one executable model. In particular, the notebook contains distinct v1, v3 and v4 implementations, and the final comparison cell introduces additional baseline models.
Before using numerical results in a manuscript, run the selected implementation from a clean runtime and retain the generated checkpoint, configuration and prediction CSV. This avoids accidentally mixing results from different notebook versions.
20. Reference to Source Notebook
This document was prepared directly from the uploaded HAmnet.ipynb notebook, including its dataset-loading pipeline, density-map generation, augmentation, HAMNet modules, loss functions, training procedure, validation-controlled inference and model-comparison code.
