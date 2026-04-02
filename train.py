#!/usr/bin/env python3
"""
Main training script for DINOv3 classification.
"""

import argparse
from datetime import datetime
from pathlib import Path

from dinov3_classifier import Config
from dinov3_classifier.trainer import ClassifierTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Classification Training")

    # Data
    parser.add_argument("--data-dir", type=str, default="dataset/classification/phone26/phone_dataset",
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs/classifier",
                        help="Base output directory for checkpoints and logs")

    # Model
    parser.add_argument("--model-type", type=str, default="convnext_tiny",
                        choices=["vits16", "vits16plus", "vitb16",
                                "convnext_tiny", "convnext_small",
                                "convnext_base", "convnext_large"],
                        help="Model architecture type")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select)")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Unfreeze backbone for fine-tuning")

    # Training
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--resize", type=int, default=192,
                        help="Image resize size")

    # Optimization
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "exponential"],
                        help="Learning rate scheduler")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")

    return parser.parse_args()


def generate_output_dir(base_dir: str, model_type: str, data_dir: str) -> str:
    """Generate unique output directory based on model, dataset and timestamp."""
    # Extract dataset name from path
    dataset_name = Path(data_dir).name

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory name: model_dataset_timestamp
    dir_name = f"{model_type}_{dataset_name}_{timestamp}"

    return str(Path(base_dir) / dir_name)


def main():
    args = parse_args()

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        from dinov3_classifier.config import PRETRAINED_WEIGHTS
        weights_path = PRETRAINED_WEIGHTS.get(args.model_type)
        if weights_path is None:
            raise ValueError(f"No default weights for model type: {args.model_type}")

    # Generate unique output directory
    output_dir = generate_output_dir(
        base_dir=args.output_dir,
        model_type=args.model_type,
        data_dir=args.data_dir,
    )

    # Create config with timestamp
    config = Config(
        data_dir=args.data_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        model_type=args.model_type,
        freeze_backbone=not args.unfreeze_backbone,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        resize_size=args.resize,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
        save_every=args.save_every,
    )

    # Train
    trainer = ClassifierTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
