"""Dataset and dataloader management."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from .config import IMAGENET_MEAN, IMAGENET_STD


class ClassificationDataset:
    """
    Classification dataset manager for train/val/test splits.

    Args:
        data_dir: Path to dataset root with train/val/test subdirectories
        resize_size: Image resize dimension
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
    """

    def __init__(
        self,
        data_dir: str,
        resize_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.resize_size = resize_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_transform = None
        self._val_transform = None

        # Datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @property
    def train_transform(self) -> v2.Compose:
        """Training transform with augmentation."""
        if self._train_transform is None:
            self._train_transform = v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(self.resize_size, antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandAugment(num_ops=2, magnitude=9),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return self._train_transform

    @property
    def val_transform(self) -> v2.Compose:
        """Validation/test transform without augmentation."""
        if self._val_transform is None:
            self._val_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((self.resize_size, self.resize_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return self._val_transform

    def _create_dataset(self, split: str, transform: Callable) -> ImageFolder:
        """Create dataset for a split."""
        split_path = self.data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        return ImageFolder(split_path, transform=transform)

    @property
    def train_dataset(self) -> ImageFolder:
        """Training dataset."""
        if self._train_dataset is None:
            self._train_dataset = self._create_dataset("train", self.train_transform)
        return self._train_dataset

    @property
    def val_dataset(self) -> ImageFolder:
        """Validation dataset."""
        if self._val_dataset is None:
            self._val_dataset = self._create_dataset("val", self.val_transform)
        return self._val_dataset

    @property
    def test_dataset(self) -> ImageFolder:
        """Test dataset."""
        if self._test_dataset is None:
            self._test_dataset = self._create_dataset("test", self.val_transform)
        return self._test_dataset

    @property
    def class_names(self) -> List[str]:
        """Get class names."""
        return self.train_dataset.classes

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        return len(self.class_names)

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> DataLoader:
        """
        Get dataloader for a split.

        Args:
            split: One of 'train', 'val', 'test'
            shuffle: Whether to shuffle data
            batch_size: Override default batch size
        """
        dataset_map = {
            "train": (self.train_dataset, True),
            "val": (self.val_dataset, False),
            "test": (self.test_dataset, False),
        }

        if split not in dataset_map:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'")

        dataset, default_shuffle = dataset_map[split]
        

        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle if shuffle else default_shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, val, test dataloaders."""
        return (
            self.get_dataloader("train"),
            self.get_dataloader("val"),
            self.get_dataloader("test"),
        )

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        from collections import Counter

        targets = self.train_dataset.targets
        class_counts = Counter(targets)
        total = len(targets)
        num_classes = len(class_counts)

        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)
            weights.append(total / (num_classes * count))

        return torch.tensor(weights, dtype=torch.float32)

    def print_info(self):
        """Print dataset information."""
        print(f"Dataset: {self.data_dir}")
        print(f"Classes ({self.num_classes}): {self.class_names}")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
