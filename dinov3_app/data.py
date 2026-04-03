"""Dataset and dataloader management."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from .config import IMAGENET_MEAN, IMAGENET_STD

# Face detection import with graceful fallback
try:
    import sys
    sys.path.append('/home/mi/Code/dinov3App/dms')
    from engine.face_algorithms.facedet.face_detection import FaceDetection
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FaceDetection = None
    FACE_DETECTION_AVAILABLE = False


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
                v2.RandomResizedCrop(self.resize_size, scale=(0.8, 1.0), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandAugment(num_ops=1, magnitude=9),  # Reduced from 2 to 1 for faster CPU processing
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
            pin_memory_device="cuda" if torch.cuda.is_available() else "",
            drop_last=(split == "train"),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
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


class SegmentationDataset(Dataset):
    """DINOv3分割数据集类，用于人脸区域分割任务"""

    def __init__(self, data_dir: str, split: str = 'train', resize: int = 256,
                 weights_path: str = None, face_detector_config: dict = None):
        """
        初始化分割数据集

        Args:
            data_dir: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            resize: 图像resize大小
            weights_path: DINOv3权重路径
            face_detector_config: 人脸检测器配置
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.resize = resize
        self.weights_path = weights_path

        # 设置人脸检测器
        if face_detector_config is None:
            face_detector_config = {
                'model_path': '/home/mi/Code/dinov3App/dms/onnx_model/cockpit-dms/facedet/facedet.onnx',
                'device': 'cuda',
                'conf_thres': 0.55,
                'iou_thres': 0.3
            }

        if FACE_DETECTION_AVAILABLE and FaceDetection is not None:
            self.face_detector = FaceDetection(**face_detector_config)
        else:
            self.face_detector = None
            print("Warning: FaceDetection not available, using dummy masks")

        # 设置图像变换
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((resize, resize), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # 设置mask变换
        self.mask_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((resize, resize), antialias=True),
            v2.ToDtype(torch.float32, scale=False),
        ])

        # 加载数据集
        self.image_paths = []
        self.mask_cache = {}  # 缓存生成的mask
        self._load_dataset()

    def _load_dataset(self):
        """加载数据集路径"""
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            raise ValueError(f"数据集目录不存在: {split_dir}")

        # 遍历所有类别目录
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                # 遍历类别下的子目录
                for subdir in class_dir.iterdir():
                    if subdir.is_dir():
                        # 查找所有图像文件
                        for img_file in subdir.glob('*.jpg'):
                            self.image_paths.append(img_file)
                        for img_file in subdir.glob('*.png'):
                            self.image_paths.append(img_file)

        print(f"加载 {self.split} 数据集: {len(self.image_paths)} 张图像")

    def _generate_face_mask(self, image_path: str) -> np.ndarray:
        """
        使用人脸检测器生成人脸区域mask

        Args:
            image_path: 图像路径

        Returns:
            人脸区域mask (H, W)，人脸区域为1，非人脸区域为0
        """
        # 检查缓存
        if image_path in self.mask_cache:
            return self.mask_cache[image_path]

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = image.shape[:2]

        # 如果人脸检测器不可用，返回全零mask
        if self.face_detector is None:
            return np.zeros((h, w), dtype=np.uint8)

        # 进行人脸检测
        result = self.face_detector.predict_image_data(image)

        # 创建mask
        mask = np.zeros((h, w), dtype=np.uint8)

        # 根据检测到的人脸框填充mask
        for face_box in result.get('face_boxes', []):
            bbox = face_box['bbox']  # [x1, y1, x2, y2]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w-1))
            y2 = max(0, min(y2, h-1))

            # 填充人脸区域
            mask[y1:y2+1, x1:x2+1] = 1

        # 缓存结果
        self.mask_cache[image_path] = mask

        return mask

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取数据集中的一个样本

        Args:
            idx: 样本索引

        Returns:
            tuple: (image_tensor, mask_tensor)
                - image_tensor: 预处理后的图像张量 (C, H, W)
                - mask_tensor: 人脸区域mask张量 (1, H, W)
        """
        image_path = self.image_paths[idx]

        # 读取图像
        image = Image.open(image_path).convert('RGB')

        # 生成人脸mask
        mask = self._generate_face_mask(str(image_path))
        mask_pil = Image.fromarray(mask * 255)  # 转换为PIL图像

        # 应用变换
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask_pil)

        # 确保mask是二值的
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        split_dir = self.data_dir / self.split
        class_names = []
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                class_names.append(class_dir.name)
        return class_names

    def get_num_classes(self) -> int:
        """获取类别数量"""
        return len(self.get_class_names())


class SegmentationTransform:
    """分割任务的数据变换类"""

    def __init__(self, resize: int = 256, is_training: bool = True):
        """
        初始化变换

        Args:
            resize: resize大小
            is_training: 是否为训练模式
        """
        self.resize = resize
        self.is_training = is_training

        # 图像变换
        if is_training:
            self.image_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((resize, resize), antialias=True),
                v2.RandomHorizontalFlip(0.5),
                v2.RandomRotation(10),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            self.image_transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((resize, resize), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        # mask变换
        self.mask_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((resize, resize), antialias=True),
            v2.ToDtype(torch.float32, scale=False),
        ])

    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用变换

        Args:
            image: PIL图像
            mask: numpy mask

        Returns:
            tuple: (image_tensor, mask_tensor)
        """
        # 转换mask为PIL图像
        mask_pil = Image.fromarray(mask * 255)

        # 应用变换
        image_tensor = self.image_transform(image)
        mask_tensor = self.mask_transform(mask_pil)

        # 确保mask是二值的
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
