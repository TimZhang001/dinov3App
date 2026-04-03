#!/usr/bin/env python3
"""
DINOv3 Classification and Segmentation Framework
Object-oriented implementation for training, evaluation, and inference.
"""

from .trainer import ClassifierTrainer, SegmentationTrainer
from .evaluator import ClassifierEvaluator, SegmentationEvaluator
from .predictor import ClassifierPredictor, SegmentationPredictor
from .model import DINOv3Classifier, DINOv3Segmentation, SegmentationLoss
from .data import ClassificationDataset, SegmentationDataset, SegmentationTransform
from .config import Config

__all__ = [
    # 分类模块
    "ClassifierTrainer",
    "ClassifierEvaluator",
    "ClassifierPredictor",
    "DINOv3Classifier",
    "ClassificationDataset",
    "Config",

    # 分割模块
    "SegmentationTrainer",
    "SegmentationEvaluator",
    "SegmentationPredictor",
    "DINOv3Segmentation",
    "SegmentationLoss",
    "SegmentationDataset",
    "SegmentationTransform",
]
