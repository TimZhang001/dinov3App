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
    DINOv3 backbone with classification head.

    Args:
        model_type: Model architecture (vits16, vits16plus, vitb16, convnext_tiny, convnext_small, convnext_base, convnext_large)
        weights_path: Path to pretrained weights
        num_classes: Number of output classes
        freeze_backbone: Whether to freeze backbone weights
        hidden_dim: Hidden layer dimension in classifier (0 or None for no hidden layer)
        unfreeze_layers: Number of last backbone layers to unfreeze (0 = freeze all, -1 = unfreeze all)
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

    # Model depths (number of blocks/layers)
    MODEL_DEPTHS = {
        # ViT models
        "vits16": 12,
        "vits16plus": 12,
        "vitb16": 12,
        # ConvNeXt models (total blocks across 4 stages)
        "convnext_tiny": 12,  # 3+3+9+3 = 18 blocks, but 4 stages for partial unfreeze
        "convnext_small": 18,  # 3+3+27+3 = 36 blocks
        "convnext_base": 18,   # 3+3+27+3 = 36 blocks
        "convnext_large": 18,  # 3+3+27+3 = 36 blocks
    }

    def __init__(
        self,
        model_type: str,
        weights_path: str,
        num_classes: int,
        freeze_backbone: bool = True,
        hidden_dim: int = 0,
        unfreeze_layers: int = 0,
    ):
        super().__init__()

        self.model_type = model_type
        self.freeze_backbone = freeze_backbone
        self.unfreeze_layers = unfreeze_layers
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Load backbone
        self.backbone = self._load_backbone(model_type, weights_path)

        # Get feature dimension
        self.feature_dim = self.FEATURE_DIMS.get(model_type, 384)

        # Classification head with optional hidden layer
        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, num_classes),
            )

        # Freeze backbone based on settings
        if freeze_backbone and unfreeze_layers == 0:
            self._freeze_backbone()
        elif unfreeze_layers > 0:
            self._freeze_backbone_partial(unfreeze_layers)
        elif unfreeze_layers == -1:
            self.unfreeze_backbone()

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

    def _freeze_backbone_partial(self, num_layers: int):
        """
        Freeze backbone except for the last N layers/blocks.

        For ViT models: unfreeze last N transformer blocks.
        For ConvNeXt models: unfreeze last N stages (4 stages total).

        Args:
            num_layers: Number of last layers to unfreeze
        """
        self.freeze_backbone = False

        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Determine if ViT or ConvNeXt
        if hasattr(self.backbone, 'blocks'):
            # ViT model - unfreeze last N blocks
            total_blocks = len(self.backbone.blocks)
            start_idx = max(0, total_blocks - num_layers)

            # Unfreeze patch_embed if unfreezing all blocks
            if start_idx == 0:
                for param in self.backbone.patch_embed.parameters():
                    param.requires_grad = True

            # Unfreeze last N blocks
            for i in range(start_idx, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        elif hasattr(self.backbone, 'stages'):
            # ConvNeXt model - unfreeze last N stages
            total_stages = len(self.backbone.stages)
            start_stage = max(0, total_stages - num_layers)

            # Unfreeze last N stages and their downsample layers
            for i in range(start_stage, total_stages):
                for param in self.backbone.stages[i].parameters():
                    param.requires_grad = True
                # Also unfreeze the corresponding downsample layer
                for param in self.backbone.downsample_layers[i].parameters():
                    param.requires_grad = True

    def _init_classifier(self):
        """Initialize classifier weights."""
        if self.hidden_dim and self.hidden_dim > 0:
            # With hidden layer: classifier[1] and classifier[3] are Linear layers
            nn.init.xavier_uniform_(self.classifier[1].weight)
            nn.init.zeros_(self.classifier[1].bias)
            nn.init.xavier_uniform_(self.classifier[3].weight)
            nn.init.zeros_(self.classifier[3].bias)
        else:
            # Without hidden layer: classifier[1] is the Linear layer
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
        # Extract features (use no_grad only when backbone is fully frozen)
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
        self.unfreeze_layers = -1
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone_layers(self, num_layers: int = 6):
        """
        Freeze only first N layers of backbone (unfreeze last N layers).

        This is an alias for _freeze_backbone_partial for backward compatibility.

        Args:
            num_layers: Number of last layers to keep unfrozen
        """
        self._freeze_backbone_partial(num_layers)

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
            hidden_dim=checkpoint.get("hidden_dim", 0),
            unfreeze_layers=checkpoint.get("unfreeze_layers", 0),
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
            "hidden_dim": self.hidden_dim,
            "unfreeze_layers": self.unfreeze_layers,
            "epoch": epoch,
            "class_names": getattr(self, "class_names", []),
        }

        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state
        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)
