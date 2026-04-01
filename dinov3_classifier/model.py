"""DINOv3 Classifier model definition."""

import sys
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

# Add dinov3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "dinov3"))

from dinov3.hub.backbones import (
    dinov3_vits16,
    dinov3_vits16plus,
    dinov3_vitb16,
    dinov3_convnext_tiny,
    dinov3_convnext_small,
    dinov3_convnext_base,
    dinov3_convnext_large,
)


class DINOv3Classifier(nn.Module):
    """
    DINOv3 backbone with linear classification head.

    Args:
        model_type: Model architecture (vits16, vits16plus, vitb16, convnext_tiny, convnext_small, convnext_base, convnext_large)
        weights_path: Path to pretrained weights
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone weights
    """

    # Model registry
    MODEL_REGISTRY = {
        # ViT models
        "vits16": dinov3_vits16,
        "vits16plus": dinov3_vits16plus,
        "vitb16": dinov3_vitb16,
        # ConvNeXt models
        "convnext_tiny": dinov3_convnext_tiny,
        "convnext_small": dinov3_convnext_small,
        "convnext_base": dinov3_convnext_base,
        "convnext_large": dinov3_convnext_large,
    }

    # Feature dimensions (last stage channels for ConvNeXt)
    FEATURE_DIMS = {
        # ViT models
        "vits16": 384,
        "vits16plus": 384,
        "vitb16": 768,
        # ConvNeXt models (last stage dims)
        "convnext_tiny": 768,
        "convnext_small": 768,
        "convnext_base": 1024,
        "convnext_large": 1536,
    }

    def __init__(
        self,
        model_type: str,
        weights_path: str,
        num_classes: int,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_type = model_type
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes

        # Load backbone
        self.backbone = self._load_backbone(model_type, weights_path)

        # Get feature dimension
        self.feature_dim = self.FEATURE_DIMS.get(model_type, 384)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes),
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()

        self._init_classifier()

    def _load_backbone(self, model_type: str, weights_path: str) -> nn.Module:
        """Load DINOv3 backbone with pretrained weights."""
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available: {list(self.MODEL_REGISTRY.keys())}")

        model_fn = self.MODEL_REGISTRY[model_type]
        backbone = model_fn(pretrained=True, weights=weights_path)
        return backbone

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _init_classifier(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            Classification logits [B, num_classes]
        """
        # Extract features
        if self.freeze_backbone:
            with torch.no_grad():
                features = self._extract_features(x)
        else:
            features = self._extract_features(x)

        # Classify
        logits = self.classifier(features)
        return logits

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone."""
        output = self.backbone(x)

        if isinstance(output, dict):
            # DINOv3 returns dict with CLS token features
            if "x_norm_clstoken" in output:
                return output["x_norm_clstoken"]
            elif "x_clstoken" in output:
                return output["x_clstoken"]
            else:
                raise KeyError(f"Expected 'x_norm_clstoken' or 'x_clstoken' in output, "
                             f"got keys: {output.keys()}")
        else:
            # Direct tensor output
            return output

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features for visualization/analysis."""
        with torch.no_grad():
            return self._extract_features(x)

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        self.freeze_backbone = False
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone_layers(self, num_layers: int = 6):
        """Freeze only first N layers of backbone."""
        self.freeze_backbone = False

        # Freeze patch embed
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False

        # Freeze first num_layers blocks
        for i, block in enumerate(self.backbone.blocks):
            for param in block.parameters():
                param.requires_grad = i >= num_layers

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        weights_path: str,
        device: torch.device = None,
    ) -> "DINOv3Classifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

        model = cls(
            model_type=checkpoint.get("model_type", "vits16"),
            weights_path=weights_path,
            num_classes=checkpoint["num_classes"],
            freeze_backbone=checkpoint.get("freeze_backbone", True),
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.class_names = checkpoint.get("class_names", [])

        if device:
            model = model.to(device)

        return model

    def save_checkpoint(
        self,
        path: str,
        epoch: int = 0,
        optimizer_state: dict = None,
        metrics: dict = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "freeze_backbone": self.freeze_backbone,
            "epoch": epoch,
            "class_names": getattr(self, "class_names", []),
        }

        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
