"""DINOv3 Classifier and Segmentation model definitions."""

from pathlib import Path
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinov3.hub.backbones import (
    dinov3_vits16,
    dinov3_vits16plus,
    dinov3_vitb16,
    dinov3_convnext_tiny,
    dinov3_convnext_small,
    dinov3_convnext_base,
    dinov3_convnext_large,
)

try:
    from dinov3.models.vision_transformer import vit_small, vit_base, vit_large
    from dinov3.models.convnext import convnext_tiny, convnext_small, convnext_base, convnext_large
except ImportError:
    print("警告: 无法导入dinov3分割模型，使用模拟实现")
    # 模拟实现用于测试
    def vit_small(patch_size=16, **kwargs):
        return nn.Identity()
    def vit_base(patch_size=16, **kwargs):
        return nn.Identity()
    def vit_large(patch_size=16, **kwargs):
        return nn.Identity()
    def convnext_tiny(**kwargs):
        return nn.Identity()
    def convnext_small(**kwargs):
        return nn.Identity()
    def convnext_base(**kwargs):
        return nn.Identity()
    def convnext_large(**kwargs):
        return nn.Identity()


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

        # Load state dict - handle torch.compile wrapper
        state_dict = checkpoint["model_state_dict"]

        # Check if state_dict is from a compiled model (keys start with _orig_mod.)
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            # Remove _orig_mod. prefix from all keys
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
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


class DINOv3Segmentation(nn.Module):
    """DINOv3分割模型类，用于人脸区域分割任务"""

    # 模型工厂
    MODEL_FACTORY = {
        'vits16': {'model_fn': vit_small, 'feature_dim': 384, 'patch_size': 16},
        'vits16plus': {'model_fn': vit_small, 'feature_dim': 384, 'patch_size': 16},
        'vitb16': {'model_fn': vit_base, 'feature_dim': 768, 'patch_size': 16},
        'convnext_tiny': {'model_fn': convnext_tiny, 'feature_dim': 768, 'patch_size': 32},
        'convnext_small': {'model_fn': convnext_small, 'feature_dim': 768, 'patch_size': 32},
        'convnext_base': {'model_fn': convnext_base, 'feature_dim': 1024, 'patch_size': 32},
        'convnext_large': {'model_fn': convnext_large, 'feature_dim': 1536, 'patch_size': 32},
    }

    def __init__(self, model_type: str = 'vits16', weights_path: str = None,
                 freeze_backbone: bool = True, unfreeze_layers: int = 0,
                 use_decoder: bool = True, decoder_channels: int = 256):
        """
        初始化DINOv3分割模型

        Args:
            model_type: 模型类型
            weights_path: 预训练权重路径
            freeze_backbone: 是否冻结backbone
            unfreeze_layers: 解冻的层数 (0=全部冻结, -1=全部解冻)
            use_decoder: 是否使用解码器
            decoder_channels: 解码器通道数
        """
        super().__init__()

        self.model_type = model_type
        self.use_decoder = use_decoder
        self.decoder_channels = decoder_channels

        # 获取模型配置
        if model_type not in self.MODEL_FACTORY:
            raise ValueError(f"不支持的模型类型: {model_type}，可选: {list(self.MODEL_FACTORY.keys())}")

        config = self.MODEL_FACTORY[model_type]
        self.feature_dim = config['feature_dim']
        self.patch_size = config['patch_size']

        # 创建backbone
        if 'vit' in model_type:
            self.backbone = config['model_fn'](patch_size=self.patch_size)
        else:
            self.backbone = config['model_fn']()

        # 添加特征提取头
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.feature_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True),
        )

        # 加载预训练权重
        if weights_path:
            self._load_weights(weights_path)

        # 冻结backbone
        self._freeze_backbone(freeze_backbone, unfreeze_layers)

        # 创建分割头
        self._create_segmentation_head()

        # 创建解码器
        if use_decoder:
            self._create_decoder()

    def _load_weights(self, weights_path: str):
        """加载预训练权重"""
        if not Path(weights_path).exists():
            print(f"警告: 权重文件不存在: {weights_path}")
            return

        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # 过滤掉不需要的键
            filtered_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('head.'):  # 排除分类头
                    filtered_dict[k] = v

            # 加载权重
            missing_keys, unexpected_keys = self.backbone.load_state_dict(filtered_dict, strict=False)
            if missing_keys:
                print(f"警告: 缺失的键: {missing_keys}")
            if unexpected_keys:
                print(f"警告: 意外的键: {unexpected_keys}")

            print(f"成功加载权重: {weights_path}")
        except Exception as e:
            print(f"加载权重失败: {e}")

    def _freeze_backbone(self, freeze: bool, unfreeze_layers: int):
        """冻结backbone参数"""
        if not freeze:
            return

        # 冻结所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 解冻指定层数
        if unfreeze_layers > 0:
            if 'vit' in self.model_type:
                # ViT模型：解冻最后几层
                total_blocks = len(self.backbone.blocks)
                for i in range(max(0, total_blocks - unfreeze_layers), total_blocks):
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = True
            else:
                # ConvNeXt模型：解冻最后几层
                total_stages = len(self.backbone.stages)
                for i in range(max(0, total_stages - unfreeze_layers), total_stages):
                    for param in self.backbone.stages[i].parameters():
                        param.requires_grad = True

    def _create_segmentation_head(self):
        """创建分割头"""
        if 'vit' in self.model_type:
            # ViT模型：使用特征图的上采样
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.decoder_channels, 3, padding=1),
                nn.BatchNorm2d(self.decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels, 1, 1),  # 输出单通道mask
            )
        else:
            # ConvNeXt模型：使用特征图的上采样
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(self.feature_dim, self.decoder_channels, 3, padding=1),
                nn.BatchNorm2d(self.decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels, 1, 1),  # 输出单通道mask
            )

    def _create_decoder(self):
        """创建解码器用于上采样"""
        if 'vit' in self.model_type:
            # ViT解码器：从patch特征恢复到原始分辨率
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_dim, self.decoder_channels,
                                 kernel_size=self.patch_size, stride=self.patch_size),
                nn.BatchNorm2d(self.decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels, self.decoder_channels // 2, 3, padding=1),
                nn.BatchNorm2d(self.decoder_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels // 2, 1, 1),
            )
        else:
            # ConvNeXt解码器：从特征图恢复到原始分辨率
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(self.feature_dim, self.decoder_channels,
                                 kernel_size=32, stride=32),  # ConvNeXt的stride是32
                nn.BatchNorm2d(self.decoder_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels, self.decoder_channels // 2, 3, padding=1),
                nn.BatchNorm2d(self.decoder_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.decoder_channels // 2, 1, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像张量 (B, C, H, W)

        Returns:
            分割mask张量 (B, 1, H, W)
        """
        # 使用特征提取器提取特征
        features = self.feature_extractor(x)

        # 使用解码器或分割头
        if self.use_decoder:
            mask = self.decoder(features)
        else:
            mask = self.segmentation_head(features)

        # 上采样到输入图像大小
        mask = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)

        return mask

    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim

    def get_patch_size(self) -> int:
        """获取patch大小"""
        return self.patch_size

    def unfreeze_backbone(self):
        """解冻整个backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """冻结整个backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False


class SegmentationLoss(nn.Module):
    """分割损失函数类"""

    def __init__(self, loss_type: str = 'bce', pos_weight: float = 1.0):
        """
        初始化损失函数

        Args:
            loss_type: 损失类型 ('bce', 'dice', 'focal')
            pos_weight: 正样本权重
        """
        super().__init__()
        self.loss_type = loss_type
        self.pos_weight = pos_weight

        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        elif loss_type == 'dice':
            self.criterion = self._dice_loss
        elif loss_type == 'focal':
            self.criterion = self._focal_loss
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice损失"""
        smooth = 1e-5
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor,
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal损失"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算损失

        Args:
            pred: 预测mask (B, 1, H, W)
            target: 真实mask (B, 1, H, W)

        Returns:
            损失值
        """
        return self.criterion(pred, target)
