"""Configuration management for DINOv3 classification and segmentation."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Config:
    """Training and inference configuration for classification and segmentation tasks."""

    # Task type
    task: str = "classification"  # "classification" or "segmentation"

    # Data paths
    data_dir: str = ""
    weights_path: str = ""
    output_dir: str = "outputs"

    # Model settings
    model_type: str = "vits16"  # vits16, vits16plus, vitb16, convnext_*
    num_classes: int = 0
    freeze_backbone: bool = True
    hidden_dim: int = 0  # Hidden layer dimension in classifier (0 = no hidden layer)
    unfreeze_layers: int = 0  # Number of last backbone layers to unfreeze (0 = freeze all, -1 = unfreeze all)

    # Segmentation-specific settings
    use_decoder: bool = True
    decoder_channels: int = 256
    loss_type: str = "bce"  # bce, dice, focal
    pos_weight: float = 1.0
    face_detector_config: Dict = field(default_factory=dict)

    # Training settings
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    resize_size: int = 256
    num_workers: int = 4

    # Optimization
    weight_decay: float = 0.05
    lr_scheduler: str = "cosine"  # cosine, step, exponential
    warmup_epochs: int = 5
    grad_clip: float = 1.0  # Gradient clipping (0 = disabled)

    # Logging
    print_freq: int = 10
    save_freq: int = 10

    # System
    device: str = "cuda"
    seed: int = 42

    # Checkpoint
    resume: Optional[str] = None
    save_every: int = 10

    # Metadata
    timestamp: str = ""
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Convert string paths to proper format and set timestamp."""
        if self.data_dir:
            self.data_dir = str(Path(self.data_dir).resolve())
        if self.weights_path:
            self.weights_path = str(Path(self.weights_path).resolve())
        self.output_dir = str(Path(self.output_dir).resolve())

        # Set timestamp if not provided
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


# Pretrained weights registry
PRETRAINED_WEIGHTS = {
    # ViT models
    "vits16": "pretrained/backbone/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vits16plus": "pretrained/backbone/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "vitb16": "pretrained/backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    # ConvNeXt models
    "convnext_tiny": "pretrained/backbone/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth",
    "convnext_small": "pretrained/backbone/dinov3_convnext_small_pretrain_lvd1689m-296db49d.pth",
    "convnext_base": "pretrained/backbone/dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth",
    "convnext_large": "pretrained/backbone/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth",
}

# ImageNet normalization for LVD-1689M weights
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
