"""Evaluation pipeline for DINOv3 classifier and segmentation."""

import datetime
import json
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from .data import ClassificationDataset, SegmentationDataset
from .model import DINOv3Classifier, DINOv3Segmentation
from .config import Config


class ResultManager:
    """管理验证结果的类，负责存储和处理路径、置信度等信息"""

    def __init__(self):
        self.results = []
        self.mistake_dict = {}

    def add_batch_result(self, paths: list, pred_prob: torch.Tensor) -> None:
        """添加批次结果

        Args:
            paths: 图像路径列表
            pred_prob: 预测概率张量，形状为 (batch_size, num_classes)
        """
        if pred_prob is None:
            return

        # 获取每个样本的预测置信度（最大概率值）
        confidences = torch.max(pred_prob, dim=1).values.cpu().numpy()

        # 存储路径和置信度信息
        for i, path in enumerate(paths):
            self.results.append({
                'path': path,
                'confidence': float(confidences[i])
            })

    def analyze_mistakes(self, targets: torch.Tensor, predictions: torch.Tensor,
                         class_names: List[str]) -> None:
        """分析错误分类样本

        Args:
            targets: 真实标签
            predictions: 预测结果
            class_names: 类别名称列表
        """
        # 初始化错误字典
        if not self.mistake_dict:
            for name in class_names:
                self.mistake_dict[name] = []

        # 处理每个样本的预测结果
        for i, (target, pred) in enumerate(zip(targets, predictions)):
            target_cls = class_names[int(target)]
            pred_cls = class_names[int(pred)]

            # 获取对应的路径和置信度信息
            if i < len(self.results):
                result_info = self.results[i]
                path = result_info['path']
                confidence = result_info['confidence']
            else:
                path = "unknown"
                confidence = 0.0

            if target_cls != pred_cls:
                # 存储错误样本的路径和置信度信息
                self.mistake_dict[pred_cls].append({
                    'path': path,
                    'confidence': confidence
                })

    def clear_results(self) -> None:
        """清空结果数据"""
        self.results.clear()

    def get_mistake_count(self) -> int:
        """获取错误样本总数"""
        return sum(len(results) for results in self.mistake_dict.values())

    def get_mistake_dict(self) -> dict:
        """获取错误字典"""
        return self.mistake_dict.copy()


class MistakeAnalyzer:
    """错误分析器，负责生成错误分析报告"""

    def __init__(self, result_manager: ResultManager):
        self.result_manager = result_manager

    @staticmethod
    def create_vscode_link(path: str) -> str:
        """创建VSCode可点击的链接"""
        encoded_path = urllib.parse.quote(path)
        return f"file:{encoded_path}"

    @staticmethod
    def create_markdown_link(path: str) -> str:
        """创建Markdown格式的超链接"""
        filename = Path(path).name
        return f"[{filename}]({MistakeAnalyzer.create_vscode_link(path)})"

    def save_mistakes_to_md(self, model_name: str, base_name: str, split: str,
                            save_dir: Path) -> str:
        """保存错误分析报告到Markdown文件

        Args:
            model_name: 模型名称
            base_name: 基础名称
            split: 数据集划分
            save_dir: 保存目录

        Returns:
            保存的文件路径
        """
        mistake_dict = self.result_manager.get_mistake_dict()
        total_mistakes = self.result_manager.get_mistake_count()

        # 创建保存目录
        save_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mistakes_{split}_{timestamp}.md"
        filepath = save_dir / filename

        # 写入markdown文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# 分类错误分析报告\n\n")
            f.write(f"**模型**: {model_name}\n\n")
            f.write(f"**数据集**: {base_name}\n\n")
            f.write(f"**划分**: {split}\n\n")
            f.write(f"**生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**总错误样本数**: {total_mistakes}\n\n")

            # 错误统计概览
            f.write("## 错误统计概览\n\n")
            f.write("| 预测类别 | 错误数量 | 占比 |\n")
            f.write("|----------|----------|------|\n")

            for pred_class, results in mistake_dict.items():
                if results:  # 只显示有错误的类别
                    percentage = (len(results) / total_mistakes * 100) if total_mistakes > 0 else 0
                    f.write(f"| {pred_class:<10} | {len(results):<8} | {percentage:>5.2f}% |\n")

            f.write("\n## 详细错误样本\n\n")
            f.write("> **提示**: 点击文件名可以在VSCode中打开图像文件\n\n")

            # 详细错误样本
            for pred_class, results in mistake_dict.items():
                if results:  # 只显示有错误的类别
                    f.write(f"### 预测为 {pred_class} 的错误样本（共 {len(results)} 个）\n\n")

                    # 按置信度降序排序，方便查看高置信度错误
                    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)

                    f.write("| 序号 | 文件名 | 置信度 |\n")
                    f.write("|------|--------|--------|\n")

                    for i, result in enumerate(sorted_results, 1):
                        path       = result['path']
                        confidence = result['confidence']

                        # path转化为绝对路径，确保链接正确
                        abs_path = str(Path(path).resolve())

                        # 获取文件名和创建超链接
                        md_link = self.create_markdown_link(abs_path)

                        f.write(f"| {i:<4} | {md_link} | {confidence:>7.4f} |\n")

                    f.write("\n")

        print(f"错误分析报告已保存到: {filepath}")
        return str(filepath)


class ClassifierEvaluator:
    """
    Evaluation pipeline for trained classifier.

    Args:
        checkpoint_path: Path to model checkpoint
        weights_path: Path to pretrained backbone weights
        output_dir: Directory to save evaluation results
        device: Device to run evaluation on
    """

    def __init__(
        self,
        checkpoint_path: str,
        weights_path: str,
        output_dir: str = "eval_results",
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.weights_path = weights_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model: Optional[DINOv3Classifier] = None
        self.class_names: List[str] = []
        self.model_type: str = "unknown"

        # 错误分析相关
        self.result_manager = ResultManager()
        self.mistake_analyzer = MistakeAnalyzer(self.result_manager)

    def load_model(self):
        """Load model from checkpoint."""
        self.model = DINOv3Classifier.from_checkpoint(
            self.checkpoint_path,
            self.weights_path,
            self.device,
        )
        self.model.eval()
        self.class_names = getattr(self.model, "class_names", [])
        self.model_type = getattr(self.model, "model_type", "unknown")
        print(f"Loaded model with {len(self.class_names)} classes")

    def evaluate(self, dataloader: DataLoader, split_name: str = "test") -> Dict:
        """
        Evaluate model on dataset.

        Args:
            dataloader: DataLoader for evaluation
            split_name: Name of the split for logging

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []

        # 清空之前的结果
        self.result_manager.clear_results()

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Evaluating {split_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            # 收集图像路径（从dataloader的dataset中获取）
            dataset = dataloader.dataset
            if hasattr(dataset, 'samples'):
                # ImageFolder stores (path, class_idx) tuples in samples
                all_paths = [sample[0] for sample in dataset.samples]
            elif hasattr(dataset, 'paths'):
                all_paths = dataset.paths
            else:
                all_paths = [f"sample_{i}" for i in range(len(dataset))]

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 添加批次结果到result_manager
        probs_tensor = torch.tensor(all_probs)
        self.result_manager.add_batch_result(all_paths, probs_tensor)

        # 分析错误样本
        targets_tensor = torch.tensor(all_labels)
        preds_tensor = torch.tensor(all_preds)
        self.result_manager.analyze_mistakes(targets_tensor, preds_tensor, self.class_names)

        # Calculate metrics
        results = self._compute_metrics(all_labels, all_preds, all_probs)
        results["predictions"] = all_preds.tolist()
        results["labels"] = all_labels.tolist()
        results["probabilities"] = all_probs.tolist()

        return results

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
    ) -> Dict:
        """Compute evaluation metrics."""
        accuracy = accuracy_score(labels, preds)
        precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
        recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

        report = classification_report(
            labels, preds,
            target_names=self.class_names if self.class_names else None,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(labels, preds)

        return {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        accuracy: float = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 10),
    ):
        """Plot and save confusion matrix."""
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if accuracy is not None:
            plt.title(f"Confusion Matrix (Top1 Acc: {accuracy:.2%})")
        else:
            plt.title("Confusion Matrix")
        plt.tight_layout()

        save_path = save_path or str(self.output_dir / "confusion_matrix.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    def plot_per_class_metrics(
        self,
        report: Dict,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 6),
    ):
        """Plot per-class precision, recall, and F1 scores."""
        metrics = {name: report[name] for name in self.class_names}

        x = np.arange(len(self.class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)
        precision_vals = [metrics[name]["precision"] for name in self.class_names]
        recall_vals = [metrics[name]["recall"] for name in self.class_names]
        f1_vals = [metrics[name]["f1-score"] for name in self.class_names]

        ax.bar(x - width, precision_vals, width, label="Precision", color="steelblue")
        ax.bar(x, recall_vals, width, label="Recall", color="darkorange")
        ax.bar(x + width, f1_vals, width, label="F1-score", color="forestgreen")

        ax.set_xlabel("Class")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Metrics")
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "per_class_metrics.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Per-class metrics saved to {save_path}")

    def run_full_evaluation(
        self,
        data_dir: str,
        split: str = "test",
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            data_dir: Path to dataset directory
            split: Dataset split to evaluate
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers

        Returns:
            Evaluation results dictionary
        """
        # Load model
        if self.model is None:
            self.load_model()

        # Load dataset
        dataset = ClassificationDataset(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if not self.class_names:
            self.class_names = dataset.class_names

        dataloader = dataset.get_dataloader(split, shuffle=False)

        print(f"\nEvaluating on {split} split ({len(dataloader.dataset)} samples)")

        # Evaluate
        results = self.evaluate(dataloader, split)

        # Print results
        print(f"\n{'=' * 50}")
        print("Evaluation Results")
        print(f"{'=' * 50}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Recall (macro): {results['recall_macro']:.4f}")
        print(f"F1-score (macro): {results['f1_macro']:.4f}")
        print(f"F1-score (weighted): {results['f1_weighted']:.4f}")

        print("\nPer-Class Metrics:")
        report = results["classification_report"]
        for name in self.class_names:
            metrics = report[name]
            print(f"  {name}: P={metrics['precision']:.4f}, "
                  f"R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

        # Save results
        metrics_save = {k: v for k, v in results.items()
                       if k not in ["predictions", "labels", "probabilities"]}
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics_save, f, indent=2)

        np.savez(
            self.output_dir / "predictions.npz",
            predictions=results["predictions"],
            labels=results["labels"],
            probabilities=results["probabilities"],
        )

        # Plot visualizations
        cm = np.array(results["confusion_matrix"])
        self.plot_confusion_matrix(cm, accuracy=results["accuracy"])
        self.plot_per_class_metrics(report)

        # Save mistake analysis report
        total_mistakes = self.result_manager.get_mistake_count()
        if total_mistakes > 0:
            dataset_name = Path(data_dir).name
            self.mistake_analyzer.save_mistakes_to_md(
                model_name=self.model_type,
                base_name=dataset_name,
                split=split,
                save_dir=self.output_dir,
            )
        else:
            print("No misclassifications found - no mistake report generated")

        print(f"\nResults saved to {self.output_dir}")

        return results


class SegmentationEvaluator:
    """DINOv3分割评估器类"""

    def __init__(self, checkpoint_path: str, weights_path: str = None, device: str = 'auto'):
        """
        初始化评估器

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

        # 评估指标
        self.metrics = {}

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
            freeze_backbone=False,  # 评估时不需要冻结
            use_decoder=config.use_decoder,
            decoder_channels=config.decoder_channels
        )

        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"成功加载模型: {self.checkpoint_path}")
        return model

    def _compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        计算分割指标

        Args:
            pred: 预测mask (B, 1, H, W)
            target: 真实mask (B, 1, H, W)

        Returns:
            指标字典
        """
        # 转换为numpy数组
        pred_np = torch.sigmoid(pred).cpu().numpy()
        target_np = target.cpu().numpy()

        # 二值化预测结果
        pred_binary = (pred_np > 0.5).astype(np.float32)

        # 计算各种指标
        metrics = {}

        # 计算每个样本的指标
        batch_size = pred_np.shape[0]
        dice_scores = []
        iou_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for i in range(batch_size):
            pred_i = pred_binary[i, 0].flatten()
            target_i = target_np[i, 0].flatten()

            # 计算混淆矩阵
            tn, fp, fn, tp = confusion_matrix(target_i, pred_i, labels=[0, 1]).ravel()

            # Dice系数
            dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            dice_scores.append(dice)

            # IoU (交并比)
            iou = tp / (tp + fp + fn + 1e-8)
            iou_scores.append(iou)

            # 精确率
            precision = tp / (tp + fp + 1e-8)
            precision_scores.append(precision)

            # 召回率
            recall = tp / (tp + fn + 1e-8)
            recall_scores.append(recall)

            # F1分数
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(f1)

        # 计算平均指标
        metrics['dice'] = np.mean(dice_scores)
        metrics['iou'] = np.mean(iou_scores)
        metrics['precision'] = np.mean(precision_scores)
        metrics['recall'] = np.mean(recall_scores)
        metrics['f1'] = np.mean(f1_scores)

        # 计算像素准确率
        pred_flat = pred_binary.flatten()
        target_flat = target_np.flatten()
        pixel_accuracy = np.mean(pred_flat == target_flat)
        metrics['pixel_accuracy'] = pixel_accuracy

        return metrics

    def evaluate(self, data_dir: str, split: str = 'test', batch_size: int = 32,
                 num_workers: int = 4, output_dir: str = None) -> Dict[str, float]:
        """
        评估模型性能

        Args:
            data_dir: 数据集目录
            split: 数据集划分
            batch_size: 批次大小
            num_workers: 数据加载线程数
            output_dir: 输出目录

        Returns:
            评估指标字典
        """
        print(f"开始评估模型，数据集: {data_dir}, 划分: {split}")

        # 创建数据集
        dataset = SegmentationDataset(
            data_dir=data_dir,
            split=split,
            resize=256,
            weights_path=self.weights_path
        )

        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        # 评估模型
        all_predictions = []
        all_targets = []
        all_images = []

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(dataloader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # 前向传播
                outputs = self.model(images)

                # 收集结果
                all_predictions.append(outputs.cpu())
                all_targets.append(masks.cpu())
                all_images.append(images.cpu())

                if batch_idx % 10 == 0:
                    print(f"评估进度: {batch_idx+1}/{len(dataloader)}")

        # 合并所有批次
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_images = torch.cat(all_images, dim=0)

        # 计算指标
        self.metrics = self._compute_metrics(all_predictions, all_targets)

        # 保存结果
        if output_dir:
            self._save_evaluation_results(output_dir, all_predictions, all_targets, all_images)

        # 打印结果
        self._print_metrics()

        return self.metrics

    def _save_evaluation_results(self, output_dir: str, predictions: torch.Tensor,
                               targets: torch.Tensor, images: torch.Tensor):
        """保存评估结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存指标
        metrics_path = output_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        # 保存可视化结果
        self._save_visualizations(output_path, predictions, targets, images)

        # 保存混淆矩阵
        self._save_confusion_matrix(output_path, predictions, targets)

        print(f"评估结果已保存到: {output_path}")

    def _save_visualizations(self, output_path: Path, predictions: torch.Tensor,
                           targets: torch.Tensor, images: torch.Tensor):
        """保存可视化结果"""
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # 选择前10个样本进行可视化
        num_samples = min(10, predictions.shape[0])

        for i in range(num_samples):
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            # 原始图像
            img = images[i].permute(1, 2, 0).numpy()
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            img = img.astype(np.uint8)
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            # 真实mask
            target_mask = targets[i, 0].numpy()
            axes[1].imshow(target_mask, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            # 预测mask
            pred_mask = torch.sigmoid(predictions[i, 0]).numpy()
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            # 二值化预测
            pred_binary = (pred_mask > 0.5).astype(np.float32)
            axes[3].imshow(pred_binary, cmap='gray')
            axes[3].set_title('Binary Prediction')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig(viz_dir / f'sample_{i:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _save_confusion_matrix(self, output_path: Path, predictions: torch.Tensor,
                             targets: torch.Tensor):
        """保存混淆矩阵"""
        # 转换为numpy数组
        pred_np = torch.sigmoid(predictions).cpu().numpy()
        target_np = targets.cpu().numpy()

        # 二值化预测结果
        pred_binary = (pred_np > 0.5).astype(np.float32)

        # 计算混淆矩阵
        pred_flat = pred_binary.flatten()
        target_flat = target_np.flatten()

        cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])

        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Background', 'Face'],
                   yticklabels=['Background', 'Face'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _print_metrics(self):
        """打印评估指标"""
        print("\n" + "="*50)
        print("分割评估结果")
        print("="*50)
        print(f"Dice系数: {self.metrics['dice']:.4f}")
        print(f"IoU: {self.metrics['iou']:.4f}")
        print(f"精确率: {self.metrics['precision']:.4f}")
        print(f"召回率: {self.metrics['recall']:.4f}")
        print(f"F1分数: {self.metrics['f1']:.4f}")
        print(f"像素准确率: {self.metrics['pixel_accuracy']:.4f}")
        print("="*50)

    def predict_single_image(self, image_path: str, output_path: str = None) -> Dict[str, np.ndarray]:
        """
        对单张图像进行预测

        Args:
            image_path: 输入图像路径
            output_path: 输出路径

        Returns:
            预测结果字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 预处理
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # 图像变换
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        image_tensor = transform(image_pil).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)

        # 后处理
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_binary = (pred_mask > 0.5).astype(np.uint8)

        # 调整到原始尺寸
        original_size = image.shape[:2]
        pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
        pred_binary_resized = cv2.resize(pred_binary, (original_size[1], original_size[0]))

        result = {
            'probability_mask': pred_mask_resized,
            'binary_mask': pred_binary_resized,
            'original_image': image
        }

        # 保存结果
        if output_path:
            self._save_prediction_result(result, output_path)

        return result

    def _save_prediction_result(self, result: Dict[str, np.ndarray], output_path: str):
        """保存预测结果"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(cv2.cvtColor(result['original_image'], cv2.COLOR_BGR2RGB))
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
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def get_metrics(self) -> Dict[str, float]:
        """获取评估指标"""
        return self.metrics

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
        }
