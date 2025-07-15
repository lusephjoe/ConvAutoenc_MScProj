# Custom 4D-STEM Autoencoder

A ResNet-based convolutional autoencoder for dimensionality reduction and analysis of 4D Scanning Transmission Electron Microscopy (4D-STEM) diffraction patterns.

## Architecture Overview

The autoencoder implements an overcomplete sparse representation with regularized training for high-quality reconstruction of diffraction patterns. It features:

- **ResNet-based encoder/decoder** with skip connections
- **Image-size agnostic** processing (adaptive pooling)
- **Regularized loss function** with multiple components
- **Sparse embedding layer** with non-negative activations
- **Comprehensive performance evaluation** with PSNR, SSIM, and MSE metrics

## Project Structure

```
Custom_4DSTEM_AE/
├── models/           # Neural network architectures
├── scripts/          # Training and utility scripts
├── data/            # Input data files
├── outputs/         # Generated results and checkpoints
├── experiments.ipynb # Jupyter notebook for experiments
└── requirements.txt  # Python dependencies
```

## Core Components

### 1. Neural Network Models (`models/`)

#### `models/autoencoder.py`
**Main autoencoder architecture with ResNet blocks**

**Classes:**
- **`Encoder`**: ResNet-based encoder with adaptive pooling
  - Input: 256×256 (or any size) diffraction patterns
  - Output: 32-dimensional latent embeddings
  - Architecture: 3 ResNet blocks → final conv → embedding layer
  - Dimension reduction: 256×256 → 64×64 → 16×16 → 4×4

- **`Decoder`**: Upsampling decoder with ResNet blocks
  - Input: 32-dimensional latent vectors
  - Output: Reconstructed diffraction patterns
  - Architecture: Linear → conv → 3 ResNet upsampling blocks
  - Dimension reconstruction: 4×4 → 16×16 → 64×64 → 256×256

- **`Autoencoder`**: Complete model with regularized loss
  - Combines encoder and decoder
  - Implements custom loss: `L = MSE + λ_act·L₁ + λ_sim·L_sim + λ_div·L_div`
  - Methods: `forward()`, `embed()`, `compute_loss()`

**Key Features:**
- Image-size agnostic processing
- Overcomplete embedding design (32 channels)
- Non-negative activations via ReLU
- Multi-component regularized loss

#### `models/blocks.py`
**Modular neural network building blocks**

**Classes:**
- **`ResidualConvBlock`**: 3 sequential conv layers with skip connection
  - 128 filters per layer
  - Batch normalization + ReLU activation
  - Handles channel dimension changes

- **`IdentityBlock`**: Single conv layer with normalization
  - Maintains feature dimensions
  - Used within ResNet blocks

- **`ResNetBlock`**: Complete ResNet block for encoder
  - Combines residual + identity blocks
  - Includes MaxPooling for downsampling

- **`ResNetUpBlock`**: Upsampling ResNet block for decoder
  - Bilinear upsampling + ResNet processing
  - Reconstructs spatial dimensions

- **`EmbeddingLayer`**: Latent space projection
  - Linear layer with ReLU (non-negative activations)
  - Configurable latent dimensions

- **`AdaptiveDecoder`**: Size-agnostic decoder
  - Handles different target image sizes
  - Automatic feature map reshaping

**Loss Components:**
- **`ContrastiveLoss`**: Promotes embedding diversity
- **`DivergenceLoss`**: Encourages activation variance

#### `models/summary.py`
**Model analysis and performance evaluation**

**Functions:**
- **`show()`**: Enhanced model summary with performance metrics
  - Layer-by-layer parameter breakdown
  - Reconstruction quality assessment
  - Generates comparison visualizations

- **`calculate_metrics()`**: Comprehensive reconstruction metrics
  - **PSNR**: Peak Signal-to-Noise Ratio
  - **SSIM**: Structural Similarity Index
  - **MSE**: Mean Squared Error
  - Handles batch processing with statistics

- **`save_comparison_images()`**: Visual comparison generator
  - Side-by-side original vs reconstructed
  - Configurable number of samples
  - High-resolution output (300 DPI)

- **`create_virtual_field_image()`**: Virtual field image generation
  - Creates virtual bright/dark field images from diffraction data
  - Automatic field region detection or custom regions
  - Supports both bright field (central) and dark field (annular) imaging

- **`save_stem_visualization()`**: Comprehensive STEM visualization
  - Raw diffraction pattern visualization (mean original and reconstructed)
  - Virtual bright field and dark field image generation
  - Field region indicators on diffraction patterns
  - Automatic scan shape detection
  - Professional layout with colorbars and proper styling

### 2. Training Scripts (`scripts/`)

#### `scripts/train.py`
**Main training script with PyTorch Lightning**

**Features:**
- **Regularized training** with multiple loss components
- **Train/validation split** (80/20) for proper evaluation
- **Optional real-time metrics** (toggleable for efficiency)
- **Comprehensive logging** with TensorBoard integration
- **Final evaluation** with detailed metrics and visualizations

**Key Arguments:**
```bash
--data           # Path to training data (.pt file)
--output_dir     # Directory for outputs and checkpoints
--epochs         # Number of training epochs
--batch          # Batch size
--latent         # Latent dimension (default: 32)
--lambda_act     # L1 regularization coefficient (default: 1e-4)
--lambda_sim     # Contrastive regularization (default: 5e-5)
--lambda_div     # Divergence regularization (default: 2e-4)
--input_size     # Input image size (default: 256)
--realtime_metrics  # Enable real-time PSNR/SSIM tracking
```

**Usage Example:**
```bash
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 50 \
    --batch 128 \
    --latent 32 \
    --device cpu \
    --realtime_metrics
```

#### `scripts/convert_dm4.py`
**Digital Micrograph file converter**

Converts .dm4 files to PyTorch tensors for training.

**Features:**
- Downsampling options for memory efficiency
- Binning and averaging modes
- Normalization to [0,1] range
- Progress tracking

**Usage:**
```bash
python scripts/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_tensor.pt \
    --downsample 8 \
    --mode bin
```

#### `scripts/generate_embeddings.py`
**Latent space embedding generator**

Generates embeddings from trained models for analysis.

**Features:**
- Batch processing for large datasets
- Checkpoint loading
- Configurable output formats (.pt, .npy)

**Usage:**
```bash
python scripts/generate_embeddings.py \
    --input data/train_tensor.pt \
    --checkpoint outputs/ae.ckpt \
    --batch_size 2048 \
    --output outputs/embeddings.pt
```

#### `scripts/visualise_scan_latents.py`
**Latent space visualization tool**

Creates spatial maps of latent dimensions across scan positions.

**Features:**
- Configurable scan grid dimensions
- Multiple visualization modes (bright-field, etc.)
- Mosaic generation for all latent dimensions
- High-quality output figures

**Usage:**
```bash
python scripts/visualise_scan_latents.py \
    --raw data/train_tensor.pt \
    --latents outputs/embeddings.pt \
    --scan 42 114 \
    --virtual bf \
    --outfig outputs/latent_mosaic.png
```

#### `scripts/preprocess.py`
**Data preprocessing utilities**

Additional data processing and preparation tools.

### 3. Experimental Workflow (`experiments.ipynb`)

**Interactive Jupyter notebook** containing:
- Data loading and preprocessing examples
- Model training demonstrations
- Visualization of results
- Latent space analysis
- Performance evaluation

## Loss Function Components

The regularized loss function combines multiple terms:

```
L = MSE(y, ŷ) + λ_act·L₁(a) + λ_sim·L_sim + λ_div·L_div
```

Where:
- **MSE**: Reconstruction fidelity
- **L₁**: Sparsity regularization (promotes sparse activations)
- **L_sim**: Contrastive similarity (encourages embedding diversity)
- **L_div**: Activation divergence (prevents mode collapse)

## Performance Metrics

The system tracks comprehensive reconstruction quality metrics:

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in dB
- **SSIM (Structural Similarity Index)**: Perceptual similarity measure (0-1)
- **MSE (Mean Squared Error)**: Pixel-wise reconstruction error

## Output Files

### Generated During Training:
- **`ae.ckpt`**: Trained model checkpoint
- **`loss_curve.png`**: Training loss progression
- **`reconstruction_comparison.png`**: Pre-training pattern comparison
- **`stem_visualization.png`**: Pre-training STEM analysis with virtual fields
- **`final_reconstruction_comparison.png`**: Post-training pattern comparison
- **`final_stem_visualization.png`**: Post-training STEM analysis
- **`tb_logs/`**: TensorBoard logging files

### Generated During Analysis:
- **`embeddings.pt`**: Latent space embeddings
- **`latent_mosaic.png`**: Spatial latent dimension maps

## Dependencies

Core requirements (see `requirements.txt`):
- PyTorch 2.2+
- PyTorch Lightning 2.2+
- scikit-image (for SSIM/PSNR)
- matplotlib (for visualization)
- numpy, h5py, tqdm
- hyperspy (for .dm4 file handling)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Convert your .dm4 data:**
   ```bash
   python scripts/convert_dm4.py --input your_data.dm4 --output data/train_tensor.pt
   ```

3. **Train the model:**
   ```bash
   python scripts/train.py --data data/train_tensor.pt --output_dir outputs --device cpu
   ```

4. **Generate embeddings:**
   ```bash
   python scripts/generate_embeddings.py --input data/train_tensor.pt --checkpoint outputs/ae.ckpt --output outputs/embeddings.pt
   ```

5. **Visualize results:**
   ```bash
   python scripts/visualise_scan_latents.py --raw data/train_tensor.pt --latents outputs/embeddings.pt --scan 42 114 --outfig outputs/results.png
   ```

## Model Architecture Details

### Encoder Path:
```
Input (1 channel) 
→ conv_input + BN + ReLU (64 channels)
→ conv_pre + BN + ReLU (128 channels)
→ ResNet Block 1 (ConvBlock + IdentityBlock + MaxPool2d)
→ ResNet Block 2 (ConvBlock + IdentityBlock + MaxPool2d)
→ ResNet Block 3 (ConvBlock + IdentityBlock + MaxPool2d)
→ conv_post + BN + ReLU (64 channels)
→ conv_final + BN + ReLU (1 channel)
→ Adaptive Pooling → Embedding (32D)
```

### Decoder Path:
```
32D embedding → [Linear] → 4×4×128
              → [Conv] → 128 channels
              → [ResNet Up Block] → 16×16
              → [ResNet Up Block] → 64×64
              → [ResNet Up Block] → 256×256
              → [Conv] → 1 channel
```

## STEM Visualization Features

The system provides comprehensive STEM analysis capabilities similar to professional microscopy software:

### **Virtual Field Imaging**
- **Bright Field**: Integrates intensity from central diffraction region
  - Default: Central 1/8 of image size
  - Visualizes crystal structure and morphology
  - Displayed with grayscale colormap

- **Dark Field**: Integrates from annular region excluding central spot
  - Default: Annular region from 1/6 to 1/3 of image radius
  - Highlights defects, grain boundaries, and strain fields
  - Displayed with hot colormap

### **Automatic Analysis**
- **Scan Shape Detection**: Automatically determines scan grid dimensions
- **Field Region Detection**: Intelligently selects appropriate regions
- **Comparison Generation**: Side-by-side original vs reconstructed analysis

### **Professional Visualization**
- **Field Region Indicators**: Red boxes (bright field), blue boxes (dark field)
- **Colorbars**: Proper intensity scaling for all images
- **Layout Optimization**: Adaptive subplot arrangement
- **High Resolution**: 300 DPI output for publication quality

### **Usage Examples**
```python
# Automatic generation during training
python scripts/train.py --data data.pt --output_dir outputs

# Manual STEM visualization
from models.summary import save_stem_visualization
save_stem_visualization(
    original_data, reconstructed_data,
    "stem_analysis.png",
    scan_shape=(64, 64),
    bright_field_region=(28, 36, 28, 36),
    dark_field_region=(20, 44, 20, 44)
)
```

## Advanced Features

- **Image-size agnostic**: Works with any input size through adaptive pooling
- **Efficient training**: Optional real-time metrics for faster training
- **Comprehensive logging**: All loss components tracked separately
- **Flexible regularization**: Adjustable coefficients for different datasets
- **Advanced STEM visualization**: 
  - Raw diffraction pattern analysis
  - Virtual bright field and dark field imaging
  - Automatic field region detection
  - Scan shape auto-detection
  - Professional scientific visualization layout

## Citation

If you use this code in your research, please cite:
```
[TBD]
```