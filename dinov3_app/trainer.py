"""Training pipeline for DINOv3 classifier and segmentation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
import numpy as np

from .config import Config
from .data import ClassificationDataset, SegmentationDataset
from .model import DINOv3Classifier, DINOv3Segmentation, SegmentationLoss


class ClassifierTrainer:
    """
    Training pipeline for DINOv3 linear classification.

    Args:
        config: Training configuration
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Components
        self.dataset: Optional[ClassificationDataset] = None
        self.model: Optional[DINOv3Classifier] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.writer: Optional[SummaryWriter] = None

        # State
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        # Setup
        self._setup()

    def _setup(self):
        """Initialize all components."""
        self._setup_directories()
        self._setup_dataset()
        self._setup_model()
        self._setup_optimization()
        self._setup_misc()

    def _setup_directories(self):
        """Create output directories."""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup TensorBoard
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Save config immediately after creating directories
        self.config.save(self.output_dir / "config.json")

    def _setup_dataset(self):
        """Initialize dataset and dataloaders."""
        self.dataset = ClassificationDataset(
            data_dir=self.config.data_dir,
            resize_size=self.config.resize_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        self.train_loader = self.dataset.get_dataloader("train", shuffle=True)
        self.val_loader = self.dataset.get_dataloader("val")

        # Update config with dataset info
        self.config.num_classes = self.dataset.num_classes
        self.config.class_names = self.dataset.class_names

    def _setup_model(self):
        """Initialize model."""
        self.model = DINOv3Classifier(
            model_type=self.config.model_type,
            weights_path=self.config.weights_path,
            num_classes=self.config.num_classes,
            freeze_backbone=self.config.freeze_backbone,
            hidden_dim=self.config.hidden_dim,
            unfreeze_layers=self.config.unfreeze_layers,
        )
        self.model = self.model.to(self.device)

        # Resume from checkpoint if specified
        if self.config.resume:
            self._resume_checkpoint()

    def _setup_optimization(self):
        """Initialize optimizer and scheduler."""
        # Different learning rates for backbone vs head
        # Case 1: Fully frozen backbone - only train classifier
        if self.config.freeze_backbone and self.config.unfreeze_layers == 0:
            params = self.model.classifier.parameters()
        # Case 2: Partial unfreeze or full unfreeze - train both backbone and classifier
        else:
            params = [
                {"params": self.model.backbone.parameters(), "lr": self.config.learning_rate * 0.1},
                {"params": self.model.classifier.parameters(), "lr": self.config.learning_rate},
            ]

        self.optimizer = optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
        elif self.config.lr_scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95,
            )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def _setup_misc(self):
        # Set random seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _resume_checkpoint(self):
        """Resume training from checkpoint."""
        checkpoint = torch.load(self.config.resume, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Resumed from epoch {self.current_epoch}")

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        global_step = self.current_epoch * len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Log to TensorBoard every 10 batches
            if self.writer and batch_idx % 10 == 0:
                step = global_step + batch_idx
                self.writer.add_scalar("train/batch_loss", loss.item(), step)
                self.writer.add_scalar("train/batch_acc", correct / total, step)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc="Validation")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{correct / total:.4f}",
            )

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, filename: str, metrics: Optional[Dict] = None):
        """Save checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "num_classes": self.config.num_classes,
            "model_type": self.config.model_type,
            "freeze_backbone": self.config.freeze_backbone,
            "hidden_dim": self.config.hidden_dim,
            "unfreeze_layers": self.config.unfreeze_layers,
            "class_names": self.config.class_names,
        }

        if metrics:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def train(self) -> Dict:
        """Run full training loop."""
        print(f"\n{'=' * 50}")
        print("Training Configuration")
        print(f"{'=' * 50}")
        print(f"Model: {self.config.model_type}")
        print(f"Device: {self.device}")
        print(f"Freeze backbone: {self.config.freeze_backbone}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Epochs: {self.config.epochs}")

        self.dataset.print_info()

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
                self.writer.add_scalar("epoch/train_acc", train_acc, epoch)
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
                self.writer.add_scalar("epoch/val_acc", val_acc, epoch)
                self.writer.add_scalar("epoch/learning_rate", self.optimizer.param_groups[0]["lr"], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint("best_model.pth", {
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                })
                print(f"  New best model saved! Val Acc: {val_acc:.4f}")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

        # Save final model
        self.save_checkpoint("final_model.pth")

        # Save config and history
        self.config.save(self.output_dir / "config.json")
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 50}")
        print("Training Completed!")
        print(f"{'=' * 50}")
        print(f"Total time: {elapsed / 60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"\nTensorBoard logs: {self.log_dir}")
        print(f"Run 'tensorboard --logdir={self.output_dir}' to view training curves")

        return self.history


class SegmentationTrainer:
    """DINOv3分割训练器类"""

    def __init__(self, config: Config):
        """
        初始化训练器

        Args:
            config: 配置对象
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # 初始化模型
        self.model = self._create_model()
        self.model.to(self.device)

        # 初始化数据集
        self.train_dataset = self._create_dataset('train')
        self.val_dataset = self._create_dataset('val')

        # 初始化数据加载器
        self.train_loader = self._create_dataloader(self.train_dataset, is_training=True)
        self.val_loader = self._create_dataloader(self.val_dataset, is_training=False)

        # 初始化损失函数
        self.criterion = SegmentationLoss(loss_type=config.loss_type, pos_weight=config.pos_weight)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'learning_rates': []
        }

        # 最佳模型指标
        self.best_val_dice = 0.0
        self.best_epoch = 0

    def _create_model(self) -> DINOv3Segmentation:
        """创建分割模型"""
        return DINOv3Segmentation(
            model_type=self.config.model_type,
            weights_path=self.config.weights_path,
            freeze_backbone=self.config.freeze_backbone,
            unfreeze_layers=self.config.unfreeze_layers,
            use_decoder=self.config.use_decoder,
            decoder_channels=self.config.decoder_channels
        )

    def _create_dataset(self, split: str) -> SegmentationDataset:
        """创建数据集"""
        return SegmentationDataset(
            data_dir=self.config.data_dir,
            split=split,
            resize=self.config.resize_size,
            weights_path=self.config.weights_path,
            face_detector_config=self.config.face_detector_config
        )

    def _create_dataloader(self, dataset: SegmentationDataset, is_training: bool) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=is_training,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=is_training
        )

    def _create_optimizer(self) -> optim.AdamW:
        """创建优化器"""
        # 分组参数：backbone和分割头使用不同的学习率
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},  # backbone使用较小学习率
            {'params': head_params, 'lr': self.config.learning_rate}
        ]

        return optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.lr_scheduler == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)
        elif self.config.lr_scheduler == 'step':
            return StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            return None

    def _compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算Dice分数"""
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)

        return dice.item()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            # 计算指标
            dice_score = self._compute_dice_score(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_score

            # 打印进度
            if batch_idx % self.config.print_freq == 0:
                print(f'Epoch [{epoch+1}/{self.config.epochs}], '
                      f'Step [{batch_idx+1}/{num_batches}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Dice: {dice_score:.4f}')

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return {'loss': avg_loss, 'dice': avg_dice}

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # 计算指标
                dice_score = self._compute_dice_score(outputs, masks)

                total_loss += loss.item()
                total_dice += dice_score

        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches

        return {'loss': avg_loss, 'dice': avg_dice}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_dice': self.best_val_dice,
            'config': self.config.__dict__
        }

        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f'保存最佳模型，Dice: {self.best_val_dice:.4f}')

        # 定期保存
        if (epoch + 1) % self.config.save_freq == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, epoch_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not Path(checkpoint_path).exists():
            print(f'检查点文件不存在: {checkpoint_path}')
            return

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_val_dice = checkpoint['best_val_dice']

        print(f'加载检查点: {checkpoint_path}, Epoch: {checkpoint["epoch"]}')

    def train(self):
        """开始训练"""
        print(f'开始训练，设备: {self.device}')
        print(f'训练集大小: {len(self.train_dataset)}')
        print(f'验证集大小: {len(self.val_dataset)}')
        print(f'模型参数量: {sum(p.numel() for p in self.model.parameters()):,}')
        print(f'可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

        # 加载检查点（如果指定）
        if self.config.resume:
            self._load_checkpoint(self.config.resume)

        # 训练循环
        for epoch in range(self.config.epochs):
            start_time = time.time()

            # 训练
            train_metrics = self._train_epoch(epoch)

            # 验证
            val_metrics = self._validate_epoch(epoch)

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_dice'].append(train_metrics['dice'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            # 检查是否为最佳模型
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                self.best_epoch = epoch

            # 保存检查点
            self._save_checkpoint(epoch, is_best)

            # 打印epoch结果
            epoch_time = time.time() - start_time
            print(f'Epoch [{epoch+1}/{self.config.epochs}] '
                  f'Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Dice: {train_metrics["dice"]:.4f}, '
                  f'Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Dice: {val_metrics["dice"]:.4f}, '
                  f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time: {epoch_time:.2f}s')

        # 保存训练历史
        self._save_history()

        print(f'训练完成！最佳验证Dice: {self.best_val_dice:.4f} (Epoch {self.best_epoch+1})')

    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / 'history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': self.config.model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_dim': self.model.get_feature_dim(),
            'patch_size': self.model.get_patch_size(),
            'use_decoder': self.model.use_decoder,
            'decoder_channels': self.model.decoder_channels
        }
