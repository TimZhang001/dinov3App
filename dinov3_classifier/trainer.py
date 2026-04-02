"""Training pipeline for DINOv3 classifier."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .data import ClassificationDataset
from .model import DINOv3Classifier


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
        )
        self.model = self.model.to(self.device)

        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device.type == 'cuda':
            self.model = torch.compile(self.model, mode='reduce-overhead')

        # Resume from checkpoint if specified
        if self.config.resume:
            self._resume_checkpoint()

    def _setup_optimization(self):
        """Initialize optimizer and scheduler."""
        # Different learning rates for backbone vs head
        if not self.config.freeze_backbone:
            params = [
                {"params": self.model.backbone.parameters(), "lr": self.config.learning_rate * 0.1},
                {"params": self.model.classifier.parameters(), "lr": self.config.learning_rate},
            ]
        else:
            params = self.model.classifier.parameters()

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
