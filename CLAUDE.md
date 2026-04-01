# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DINOv3 application workspace containing:
- `dinov3/` - The DINOv3 repository (Meta AI's self-supervised vision foundation model)
- `dataset/` - Symlink to classification training data (`/home/mi/BackupData/TrainData/classification/`)
- `pretrained/` - Downloaded pretrained model weights
- `down.py` - Example inference script using Hugging Face Transformers

## Environment Setup

```bash
# Create conda environment (recommended)
micromamba env create -f dinov3/conda.yaml
micromamba activate dinov3

# Or install via pip
pip install -e dinov3/
```

Requires Python 3.11+, PyTorch >= 2.7.1 with CUDA support.

## Key Commands

All commands should be run with `PYTHONPATH=.` from the `dinov3/` directory.

### Training
```bash
PYTHONPATH=. python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 4 \
  --config-file dinov3/configs/train/vitl_im1k_lin834.yaml \
  --output-dir <OUTPUT_DIR> \
  train.dataset_path=ImageNet22k:root=<DATASET_PATH>:extra=<DATASET_PATH>
```

### Evaluation
```bash
# k-NN classification
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/knn.py ...

# Linear classification
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/linear.py ...

# Semantic segmentation (ADE20K)
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/segmentation/run.py \
  model.dino_hub=dinov3_vit7b16 \
  config=dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  datasets.root=<DATASET_PATH>

# Depth estimation (NYUv2)
PYTHONPATH=. python -m dinov3.run.submit dinov3/eval/depth/run.py \
  model.dino_hub=dinov3_vit7b16 \
  config=dinov3/eval/depth/configs/config-nyu.yaml \
  datasets.root=<DATASET_PATH>
```

## Model Loading

### Via PyTorch Hub (local weights)
```python
import torch
REPO_DIR = "dinov3"  # Path to the dinov3 repo

# Load backbone with local weights
model = torch.hub.load(REPO_DIR, 'dinov3_vits16', source='local',
                       weights='pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
```

### Via Hugging Face Transformers
```python
from transformers import pipeline
feature_extractor = pipeline(
    model="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    task="image-feature-extraction",
)
features = feature_extractor(image)
```

## Architecture

### Package Structure (`dinov3/dinov3/`)
- `models/` - Model definitions (Vision Transformer, ConvNeXt)
- `hub/` - PyTorch Hub loaders for pretrained models (backbones, classifiers, depthers, detectors, segmentors, dinotxt)
- `eval/` - Evaluation scripts (knn.py, linear.py, log_regression.py, segmentation/, depth/, detection/, text/)
- `train/` - Training code with DINO self-distillation
- `data/` - Dataset implementations (ImageNet, ImageNet22k, etc.)
- `configs/` - YAML configuration files for training

### Model Variants
- **ViT**: vits16 (21M), vits16plus (29M), vitb16 (86M), vitl16 (300M), vith16plus (840M), vit7b16 (6.7B)
- **ConvNeXt**: tiny, small, base, large

### Pretrained Weights in `pretrained/backbone/`

**ViT Models:**
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

**ConvNeXt Models:**
- `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth`
- `dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth`
- `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth`
- `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth`

### Adapter Heads in `pretrained/adapter/`
- `dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth` (ImageNet classification head)

## Image Transforms

For LVD-1689M weights (web images):
```python
from torchvision.transforms import v2
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

For SAT-493M weights (satellite imagery), use mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143).

## Linting/Type Checking

```bash
ruff check dinov3/
mypy dinov3/
```

## Custom Classification Framework

The `dinov3_classifier/` package provides an object-oriented interface for custom classification tasks:

### Structure
```
dinov3_classifier/
├── config.py      # Configuration management
├── model.py       # DINOv3Classifier model
├── data.py        # ClassificationDataset class
├── trainer.py     # ClassifierTrainer class
├── evaluator.py   # Evaluator class
└── predictor.py   # Predictor class
```

### Usage

**Training:**
```bash
python train.py \
    --data-dir dataset/classification/abnormal26/abnormal_dataset \
    --model-type vits16 \
    --batch-size 64 \
    --epochs 100
```

**Evaluation:**
```bash
python evaluate.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --data-dir dataset/classification/abnormal26/abnormal_dataset \
    --split test
```

**Inference:**
```bash
# Single image
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image path/to/image.jpg

# Directory of images
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image-dir path/to/images
```
