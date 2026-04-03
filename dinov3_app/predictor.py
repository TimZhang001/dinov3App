"""Inference pipeline for DINOv3 classifier and segmentation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import cv2

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .config import IMAGENET_MEAN, IMAGENET_STD, Config
from .model import DINOv3Classifier, DINOv3Segmentation


class ClassifierPredictor:
    """
    Inference pipeline for trained classifier.

    Args:
        checkpoint_path: Path to model checkpoint
        weights_path: Path to pretrained backbone weights
        device: Device to run inference on
    """

    def __init__(
        self,
        checkpoint_path: str,
        weights_path: str,
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.weights_path = weights_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model: Optional[DINOv3Classifier] = None
        self.class_names: List[str] = []
        self.transform: Optional[v2.Compose] = None

        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        self.model = DINOv3Classifier.from_checkpoint(
            self.checkpoint_path,
            self.weights_path,
            self.device,
        )
        self.model.eval()
        self.class_names = getattr(self.model, "class_names", [])
        print(f"Loaded model with {len(self.class_names)} classes: {self.class_names}")

    def _get_transform(self, resize_size: int = 256) -> v2.Compose:
        """Get inference transform."""
        if self.transform is None:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((resize_size, resize_size), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        return self.transform

    def predict_single(
        self,
        image_path: str,
        top_k: int = 5,
        resize_size: int = 256,
    ) -> Dict:
        """
        Predict on a single image.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            resize_size: Image resize dimension

        Returns:
            Dictionary with prediction results
        """
        transform = self._get_transform(resize_size)

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)

        # Get top-k predictions
        top_k = min(top_k, len(self.class_names))
        top_probs, top_indices = probs.topk(top_k)

        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            predictions.append({
                "class": self.class_names[idx.item()],
                "class_index": int(idx.item()),
                "probability": float(prob.item()),
            })

        return {
            "image_path": str(image_path),
            "predictions": predictions,
            "predicted_class": predictions[0]["class"],
            "confidence": predictions[0]["probability"],
        }

    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        resize_size: int = 256,
        num_workers: int = 4,
    ) -> List[Dict]:
        """
        Predict on multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            resize_size: Image resize dimension
            num_workers: Number of data loading workers

        Returns:
            List of prediction dictionaries
        """
        transform = self._get_transform(resize_size)

        # Create dataset
        class ImageListDataset(Dataset):
            def __init__(self, paths, transform):
                self.paths = paths
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                image = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(image), str(self.paths[idx])

        dataset = ImageListDataset(image_paths, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        # Predict
        results = []
        with torch.no_grad():
            for images, paths in tqdm(loader, desc="Processing"):
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)

                for i, path in enumerate(paths):
                    top_prob, top_idx = probs[i].max(0)
                    results.append({
                        "image_path": path,
                        "predicted_class": self.class_names[top_idx.item()],
                        "class_index": int(top_idx.item()),
                        "confidence": float(top_prob.item()),
                    })

        return results

    def predict_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        batch_size: int = 32,
        output_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        Predict on all images in a directory.

        Args:
            directory: Path to directory containing images
            extensions: Allowed file extensions
            batch_size: Batch size for processing
            output_file: Optional path to save results as JSON

        Returns:
            List of prediction dictionaries
        """
        extensions = extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]

        # Find all images
        dir_path = Path(directory)
        image_paths = [
            str(p) for p in dir_path.rglob("*")
            if p.suffix.lower() in extensions
        ]

        print(f"Found {len(image_paths)} images in {directory}")

        if not image_paths:
            print("No images found!")
            return []

        # Predict
        results = self.predict_batch(image_paths, batch_size)

        # Save results
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")

        return results

    def predict_with_features(
        self,
        image_path: str,
        resize_size: int = 256,
    ) -> Dict:
        """
        Predict and return features for analysis.

        Args:
            image_path: Path to image file
            resize_size: Image resize dimension

        Returns:
            Dictionary with prediction and feature vector
        """
        transform = self._get_transform(resize_size)

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.get_features(image_tensor)
            logits = self.model.classifier(features)
            probs = torch.softmax(logits, dim=1)

        top_prob, top_idx = probs[0].max(0)

        return {
            "image_path": str(image_path),
            "predicted_class": self.class_names[top_idx.item()],
            "confidence": float(top_prob.item()),
            "features": features[0].cpu().numpy().tolist(),
        }

    def print_prediction(self, result: Dict):
        """Pretty print prediction result."""
        print(f"\nImage: {result['image_path']}")
        print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.4f})")

        if "predictions" in result:
            print("Top predictions:")
            for pred in result["predictions"]:
                print(f"  {pred['class']}: {pred['probability']:.4f}")


class SegmentationPredictor:
    """DINOv3分割推理器类，用于人脸区域分割推理"""

    def __init__(self, checkpoint_path: str, weights_path: str = None, device: str = 'auto'):
        """
        初始化推理器

        Args:
            checkpoint_path: 模型检查点路径
            weights_path: DINOv3权重路径
            device: 计算设备
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.weights_path = weights_path

        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 加载模型
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        # 推理配置
        self.batch_size = 1
        self.threshold = 0.5

    def _load_model(self) -> DINOv3Segmentation:
        """加载训练好的模型"""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")

        # 加载检查点
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config_dict = checkpoint['config']

        # 从检查点恢复配置
        config = Config()
        config.__dict__.update(config_dict)

        # 创建模型
        model = DINOv3Segmentation(
            model_type=config.model_type,
            weights_path=self.weights_path,
            freeze_backbone=False,  # 推理时不需要冻结
            use_decoder=config.use_decoder,
            decoder_channels=config.decoder_channels
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"成功加载模型: {self.checkpoint_path}")
        return model

    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: 输入图像（路径、numpy数组或PIL图像）

        Returns:
            预处理后的图像张量
        """
        # 读取图像
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("不支持的图像格式")

        # 应用变换
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # 添加batch维度

    def predict_single(self, image: Union[str, np.ndarray, Image.Image],
                      threshold: float = None) -> Dict[str, np.ndarray]:
        """
        对单张图像进行分割预测

        Args:
            image: 输入图像
            threshold: 分割阈值

        Returns:
            预测结果字典
        """
        if threshold is None:
            threshold = self.threshold

        # 预处理图像
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # 记录推理时间
        start_time = time.time()

        # 模型推理
        with torch.no_grad():
            output = self.model(image_tensor)

        inference_time = time.time() - start_time

        # 后处理
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # 移除batch和channel维度
        pred_binary = (pred_mask > threshold).astype(np.uint8)

        # 获取原始图像尺寸
        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_size = original_image.shape[:2]
        elif isinstance(image, np.ndarray):
            original_size = image.shape[:2]
        else:
            original_size = image.size[::-1]  # PIL图像的size是(width, height)

        # 调整mask到原始尺寸
        pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
        pred_binary_resized = cv2.resize(pred_binary, (original_size[1], original_size[0]))

        result = {
            'probability_mask': pred_mask_resized,
            'binary_mask': pred_binary_resized,
            'inference_time': inference_time,
            'threshold': threshold,
            'original_size': original_size
        }

        return result

    def predict_batch(self, images: List[Union[str, np.ndarray, Image.Image]],
                     threshold: float = None) -> List[Dict[str, np.ndarray]]:
        """
        对批量图像进行分割预测

        Args:
            images: 输入图像列表
            threshold: 分割阈值

        Returns:
            预测结果列表
        """
        if threshold is None:
            threshold = self.threshold

        results = []

        # 批量处理
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]

            # 预处理批次
            batch_tensors = []
            original_sizes = []

            for image in batch_images:
                image_tensor = self.preprocess_image(image)
                batch_tensors.append(image_tensor)

                # 获取原始尺寸
                if isinstance(image, str):
                    original_image = cv2.imread(image)
                    original_sizes.append(original_image.shape[:2])
                elif isinstance(image, np.ndarray):
                    original_sizes.append(image.shape[:2])
                else:
                    original_sizes.append(image.size[::-1])

            # 合并批次
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

            # 批量推理
            start_time = time.time()
            with torch.no_grad():
                batch_output = self.model(batch_tensor)
            inference_time = time.time() - start_time

            # 处理每个结果
            for j, (output, original_size) in enumerate(zip(batch_output, original_sizes)):
                pred_mask = torch.sigmoid(output).cpu().numpy()[0]  # 移除channel维度
                pred_binary = (pred_mask > threshold).astype(np.uint8)

                # 调整到原始尺寸
                pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
                pred_binary_resized = cv2.resize(pred_binary, (original_size[1], original_size[0]))

                result = {
                    'probability_mask': pred_mask_resized,
                    'binary_mask': pred_binary_resized,
                    'inference_time': inference_time / len(batch_images),  # 平均时间
                    'threshold': threshold,
                    'original_size': original_size
                }
                results.append(result)

        return results

    def predict_directory(self, input_dir: str, output_dir: str,
                         threshold: float = None, save_visualization: bool = True) -> Dict[str, any]:
        """
        对目录中的所有图像进行分割预测

        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
            threshold: 分割阈值
            save_visualization: 是否保存可视化结果

        Returns:
            处理结果统计
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 查找所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))

        if not image_files:
            raise ValueError(f"在目录 {input_dir} 中未找到图像文件")

        print(f"找到 {len(image_files)} 张图像")

        # 批量处理
        results = []
        total_time = 0

        for i, image_file in enumerate(image_files):
            try:
                # 预测
                result = self.predict_single(str(image_file), threshold)
                results.append(result)
                total_time += result['inference_time']

                # 保存结果
                self._save_prediction_result(
                    result, image_file, output_path, save_visualization
                )

                if (i + 1) % 10 == 0:
                    print(f"处理进度: {i + 1}/{len(image_files)}")

            except Exception as e:
                print(f"处理图像 {image_file} 时出错: {e}")
                continue

        # 统计结果
        stats = {
            'total_images': len(image_files),
            'processed_images': len(results),
            'total_time': total_time,
            'average_time': total_time / len(results) if results else 0,
            'threshold': threshold or self.threshold
        }

        # 保存统计信息
        stats_path = output_path / 'processing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"处理完成！共处理 {stats['processed_images']} 张图像")
        print(f"平均推理时间: {stats['average_time']:.4f} 秒")

        return stats

    def _save_prediction_result(self, result: Dict[str, np.ndarray],
                              image_file: Path, output_path: Path,
                              save_visualization: bool):
        """保存预测结果"""
        # 保存二值mask
        mask_path = output_path / f"{image_file.stem}_mask.png"
        cv2.imwrite(str(mask_path), result['binary_mask'] * 255)

        # 保存概率mask
        prob_path = output_path / f"{image_file.stem}_probability.png"
        prob_mask = (result['probability_mask'] * 255).astype(np.uint8)
        cv2.imwrite(str(prob_path), prob_mask)

        # 保存可视化结果
        if save_visualization:
            self._save_visualization(result, image_file, output_path)

    def _save_visualization(self, result: Dict[str, np.ndarray],
                          image_file: Path, output_path: Path):
        """保存可视化结果"""
        import matplotlib.pyplot as plt

        # 读取原始图像
        original_image = cv2.imread(str(image_file))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 概率mask
        axes[1].imshow(result['probability_mask'], cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Probability Mask')
        axes[1].axis('off')

        # 二值mask
        axes[2].imshow(result['binary_mask'], cmap='gray')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')

        plt.tight_layout()
        viz_path = output_path / f"{image_file.stem}_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

    def set_threshold(self, threshold: float):
        """设置分割阈值"""
        if not 0 <= threshold <= 1:
            raise ValueError("阈值必须在0和1之间")
        self.threshold = threshold

    def set_batch_size(self, batch_size: int):
        """设置批处理大小"""
        if batch_size < 1:
            raise ValueError("批处理大小必须大于0")
        self.batch_size = batch_size

    def get_model_info(self) -> Dict[str, any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': self.model.model_type,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'feature_dim': self.model.get_feature_dim(),
            'patch_size': self.model.get_patch_size(),
            'use_decoder': self.model.use_decoder,
            'device': str(self.device),
            'threshold': self.threshold,
            'batch_size': self.batch_size
        }
