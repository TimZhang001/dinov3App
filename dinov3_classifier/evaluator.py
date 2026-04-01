"""Evaluation pipeline for DINOv3 classifier."""

import json
from pathlib import Path
from typing import Dict, List, Optional

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
from tqdm import tqdm

from .config import Config
from .data import ClassificationDataset
from .model import DINOv3Classifier


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

    def load_model(self):
        """Load model from checkpoint."""
        self.model = DINOv3Classifier.from_checkpoint(
            self.checkpoint_path,
            self.weights_path,
            self.device,
        )
        self.model.eval()
        self.class_names = getattr(self.model, "class_names", [])
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

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

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
        self.plot_confusion_matrix(cm)
        self.plot_per_class_metrics(report)

        print(f"\nResults saved to {self.output_dir}")

        return results
