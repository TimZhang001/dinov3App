# DINOv3 Classification Framework - Agent Guide

## Project Overview

This is a **DINOv3-based image classification framework** that provides an object-oriented interface for training, evaluating, and performing inference with DINOv3 pretrained models. The project wraps Meta AI's DINOv3 self-supervised vision foundation model with a flexible classification head for custom datasets.

### Key Technologies
- **Python 3.11+** with PyTorch 2.0+
- **DINOv3** - Meta AI's self-supervised vision foundation model
- **torchvision** - Image transforms and utilities
- **scikit-learn** - Evaluation metrics and analysis
- **matplotlib/seaborn** - Visualization

### Architecture
The framework consists of two main components:
1. **DINOv3 Core** (`dinov3/`) - Meta AI's official DINOv3 repository with pretrained models
2. **Classification Framework** (`dinov3_app/`) - Custom object-oriented wrapper for classification tasks

## Project Structure

```
dinov3App/
├── dinov3/                      # DINOv3 official repository (git-ignored)
├── dinov3_app/                  # Custom classification framework
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management
│   ├── model.py                # DINOv3Classifier model definition
│   ├── data.py                 # ClassificationDataset class
│   ├── trainer.py              # ClassifierTrainer class
│   ├── evaluator.py            # Evaluation utilities
│   └── predictor.py            # Inference utilities
├── pretrained/
│   ├── backbone/               # Pretrained backbone weights
│   └── adapter/                # Pretrained adapter heads
├── dataset/                    # Symlink to training data
├── train.py                    # Training entry point
├── evaluate.py                 # Evaluation entry point
├── inference.py                # Inference entry point
├── run_train.sh               # Training script example
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # Claude Code guidance
└── README.md                   # Project documentation
```

## Supported Models

### ViT Models
| Model | Parameters | Feature Dim | Weights File |
|-------|------------|-------------|--------------|
| vits16 | 21M | 384 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` |
| vits16plus | 29M | 384 | `dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth` |
| vitb16 | 86M | 768 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |

### ConvNeXt Models
| Model | Parameters | Feature Dim | Weights File |
|-------|------------|-------------|--------------|
| convnext_tiny | 29M | 768 | `dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth` |
| convnext_small | 50M | 768 | `dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth` |
| convnext_base | 89M | 1024 | `dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth` |
| convnext_large | 198M | 1536 | `dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth` |

## Building and Running

### Environment Setup

```bash
# Option 1: Using conda (recommended)
micromamba env create -f dinov3/conda.yaml
micromamba activate dinov3

# Option 2: Using pip
pip install -r requirements.txt
```

### Download Pretrained Weights

Download pretrained weights from [Meta AI DINOv3 downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) and place them in `pretrained/backbone/`.

### Training

**Basic Training (Linear Probing - Frozen Backbone):**
```bash
python train.py \
    --data-dir dataset/classify/abnormal26/abnormal_dataset \
    --model-type vits16 \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001
```

**Training with Hidden Layer:**
```bash
python train.py \
    --data-dir dataset/classify/abnormal26/abnormal_dataset \
    --model-type vits16 \
    --hidden-dim 256 \
    --batch-size 64
```

**Partial Fine-tuning (Unfreeze Last 2 Layers):**
```bash
python train.py \
    --data-dir dataset/classify/abnormal26/abnormal_dataset \
    --model-type convnext_base \
    --unfreeze-layers 2 \
    --lr 0.0001
```

**Full Fine-tuning (Unfreeze Entire Backbone):**
```bash
python train.py \
    --data-dir dataset/classify/abnormal26/abnormal_dataset \
    --model-type convnext_base \
    --unfreeze-backbone \
    --batch-size 32 \
    --lr 0.0001
```

**Using Training Script:**
```bash
./run_train.sh
```

### Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --data-dir dataset/classify/abnormal26/abnormal_dataset \
    --split test \
    --output-dir eval_results
```

**Evaluation Outputs:**
- `metrics.json` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `per_class_metrics.png` - Per-class metrics charts
- `predictions.npz` - Raw predictions

### Inference

**Single Image Prediction:**
```bash
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --top-k 5
```

**Batch Prediction:**
```bash
python inference.py \
    --checkpoint outputs/classifier_vits16/checkpoints/best_model.pth \
    --image-dir path/to/images/ \
    --output predictions.json
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | Required | Path to dataset directory |
| `--model-type` | vits16 | Model architecture type |
| `--output-dir` | outputs/classifier | Output directory |
| `--batch-size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--epochs` | 100 | Number of epochs |
| `--resize` | 256 | Image resize size |
| `--hidden-dim` | 0 | Hidden layer dimension (0 = no hidden layer) |
| `--unfreeze-layers` | 0 | Unfreeze last N layers (0 = freeze all, -1 = unfreeze all) |
| `--unfreeze-backbone` | False | Fully unfreeze backbone |
| `--resume` | None | Resume from checkpoint |

## Backbone Freezing Strategies

| Setting | Behavior |
|---------|----------|
| Default | Backbone completely frozen, only train classifier head |
| `--unfreeze-layers 2` | Unfreeze last 2 layers, freeze rest |
| `--unfreeze-layers -1` or `--unfreeze-backbone` | Fully unfreeze backbone |

**ViT Models:** `unfreeze-layers N` unfreezes last N transformer blocks
**ConvNeXt Models:** `unfreeze-layers N` unfreezes last N stages (4 stages total)

## Classifier Head Architecture

**Default (no hidden layer):**
```
LayerNorm -> Linear(feature_dim, num_classes)
```

**With hidden layer (`--hidden-dim 256`):**
```
LayerNorm -> Linear(feature_dim, hidden_dim) -> GELU -> Linear(hidden_dim, num_classes)
```

## Dataset Format

Organize datasets in ImageFolder format:
```
dataset/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   └── ...
└── test/
    ├── class_1/
    └── ...
```

## Output Structure

Training generates:
```
outputs/classifier_vits16_dataset_timestamp/
├── config.json              # Training configuration
├── history.json             # Training history
├── checkpoints/
│   ├── best_model.pth       # Best model checkpoint
│   ├── final_model.pth      # Final model checkpoint
│   └── checkpoint_epoch_*.pth  # Periodic checkpoints
```

## Python API Usage

### Training
```python
from dinov3_app import Config, ClassifierTrainer

config = Config(
    data_dir="path/to/dataset",
    model_type="vits16",
    epochs=100,
    batch_size=64,
)
trainer = ClassifierTrainer(config)
trainer.train()
```

### Evaluation
```python
from dinov3_app.evaluator import ClassifierEvaluator

evaluator = ClassifierEvaluator(
    checkpoint_path="outputs/classifier/checkpoints/best_model.pth",
    weights_path="pretrained/backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
evaluator.run_full_evaluation("path/to/dataset", split="test")
```

### Inference
```python
from dinov3_app.predictor import ClassifierPredictor

predictor = ClassifierPredictor(
    checkpoint_path="outputs/classifier/checkpoints/best_model.pth",
    weights_path="pretrained/backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
result = predictor.predict_single("path/to/image.jpg")
print(f"Predicted: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
```

## Development Conventions

### Code Style
- **Object-oriented design** with clear separation of concerns
- **Type hints** throughout the codebase
- **Dataclass-based configuration** for type safety
- **Modular architecture** allowing easy extension

### File Organization
- `config.py` - All configuration management
- `model.py` - Model definition and loading
- `data.py` - Dataset and data loading
- `trainer.py` - Training logic
- `evaluator.py` - Evaluation metrics and visualization
- `predictor.py` - Inference utilities

### Image Transforms
For LVD-1689M weights (standard web images):
```python
from torchvision.transforms import v2
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

## Testing

**Note:** No formal test suite is currently implemented. Testing is done through:
- Manual training runs with small datasets
- Evaluation on validation/test splits
- Visual inspection of training curves and metrics

## Common Issues and Solutions

### Memory Issues
- Reduce `--batch-size`
- Use smaller model variants (vits16 instead of vitb16)
- Enable gradient checkpointing (if implemented)

### Slow Training
- Increase `--num-workers` for data loading
- Use mixed precision training (if implemented)
- Ensure CUDA is properly configured

### Poor Performance
- Try different learning rates (0.001 for linear probing, 0.0001 for fine-tuning)
- Adjust `--unfreeze-layers` for your dataset size
- Ensure proper data augmentation
- Check dataset quality and balance

## References

- [DINOv3 Paper](https://arxiv.org/abs/2508.10104)
- [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)
- [Hugging Face DINOv3](https://huggingface.co/docs/transformers/model_doc/dinov3)

## License

Project code: MIT License
DINOv3 models: [Original license](dinov3/LICENSE.md)