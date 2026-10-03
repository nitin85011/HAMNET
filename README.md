HAMNet Crowd Density Estimation - Code Documentation

Dataset and Annotation Processing
Roboflow/RetinaNet-style bounding boxes
Bounding-box center extraction
Gaussian density-map generation
Train/validation/test split
Point-Aware Data Augmentation
Scaling and cropping
Horizontal flipping
Rotation
Photometric transformations
Synchronized transformation of head points
HAMNet Architecture
VGG16-BN or ConvNeXt-Tiny backbone
B3/B4 feature extraction
Feature reduction to 256 channels
Multi-scale dilated convolution block
CBAM channel and spatial attention
Density regression head

Density Map Generation and Count Estimation

$$ D(x)=\sum_{i=1}^{N}\mathcal{N}(x;x_i,\sigma_i^2) $$

and

$$ \hat{C}=\frac{\sum_{x}\hat{D}(x)}{s} $$

where \(x_i\) represents the annotated head centre, \(\sigma_i\) is the Gaussian bandwidth, and \(s\) is the density scaling factor.

Composite Training Objective

For the v3 model, the loss can be formally presented as

$$ \mathcal{L}= \lambda_1\mathcal{L}_{L1} +\lambda_2\mathcal{L}_{MSE} +\lambda_3\mathcal{L}_{SSIM} +\lambda_4\mathcal{L}_{count} +\lambda_5\mathcal{L}_{patch} +\lambda_6\mathcal{L}_{bias}. $$

The paper should report the actual \(\lambda\) values used in the notebook rather than describing the loss only qualitatively.

Training Configuration

Report the exact settings:

Parameter	Configuration
Crop size	\(512\times512\)
Maximum image side	1536
Batch size	8
Epochs	120
Early stopping patience	40
Head learning rate	\(2\times10^{-4}\)
Backbone LR multiplier	0.1
Weight decay	\(10^{-4}\)
Warm-up	3 epochs
Gradient clipping	5
AMP	Enabled
EMA decay	0.995
Density scale	100
Optimizer	AdamW
Random seed	42

Validation-Based Inference Optimization

The validation set should be used to select the inference scale, horizontal flipping, and calibration. The test set must remain untouched during this selection.

Calibration should be reported as:

$$ \alpha= \frac{\sum_i C_i^{GT}\hat C_i} {\sum_i\hat C_i^2} $$

followed by

$$ \hat C_i^{cal}=\alpha\hat C_i. $$

Evaluation Metrics

Clearly define:

$$ MAE=\frac{1}{N}\sum_{i=1}^{N}|C_i-\hat C_i| $$ $$ MSE=\frac{1}{N}\sum_{i=1}^{N}(C_i-\hat C_i)^2 $$ $$ RMSE=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(C_i-\hat C_i)^2}. $$

If the notebook reports MSE as the square root of the mean squared error, then it should be called RMSE, not MSE. This distinction is particularly important when comparing HAMNet against published crowd-counting results.

Baseline Comparison

The comparison should include MCNN, AlexNet, VGG16, ResNet50, CSRNet, and HAMNet under the same evaluation protocol wherever possible. If pretrained/public implementations use different preprocessing or datasets, that should be explicitly stated rather than presenting the numbers as directly equivalent.

Reproducibility Protocol

The final paper should state that the selected implementation was executed sequentially from a clean runtime using seed 42, with the final configuration, software environment, GPU, checkpoint, and test predictions retained.

One important publication issue

Your documentation describes v3 and v4 as substantially different models:

v3: VGG16-BN + PixelShuffle ×2 density head + SSIM/composite loss.
v4: VGG16/ConvNeXt-Tiny option + stride-8 density prediction + relative-count loss + optional tiled inference.

Therefore, don't describe both as a single HAMNet architecture in the paper. Choose the implementation that produced the reported final experimental results, and identify the other versions as development/ablation versions.

If you give me the actual notebook (.ipynb) or Python file, I can turn this documentation into a publication-ready HAMNet methodology section with the exact equations, architecture diagram, algorithm/pseudocode, training table, and LaTeX code, based on what the code actually implements rather than assumptions.
