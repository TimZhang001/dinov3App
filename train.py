#!/usr/bin/env python3
"""
Unified training script for DINOv3 classification and segmentation.
"""

import argparse
from datetime import datetime
from pathlib import Path

from dinov3_app import Config
from dinov3_app.trainer import ClassifierTrainer, SegmentationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Training Script")

    # Task type
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Task type: classification or segmentation")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Base output directory for checkpoints and logs")

    # Model
    parser.add_argument("--model-type", type=str, default="vits16",
                        choices=["vits16", "vits16plus", "vitb16",
                                "convnext_tiny", "convnext_small",
                                "convnext_base", "convnext_large"],
                        help="Model architecture type")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select)")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Number of last backbone layers to unfreeze (0 = freeze all, -1 = unfreeze all)")
    parser.add_argument("--freeze-backbone", action="store_true", default=True,
                        help="Freeze backbone weights")

    # Classification-specific
    parser.add_argument("--hidden-dim", type=int, default=0,
                        help="Hidden layer dimension in classifier (0 = no hidden layer)")

    # Segmentation-specific
    parser.add_argument("--use-decoder", action="store_true", default=True,
                        help="Use decoder for segmentation")
    parser.add_argument("--decoder-channels", type=int, default=256,
                        help="Decoder channels for segmentation")
    parser.add_argument("--loss-type", type=str, default="bce",
                        choices=["bce", "dice", "focal"],
                        help="Loss function type for segmentation")
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Positive sample weight for segmentation loss")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--resize", type=int, default=256,
                        help="Image resize size")

    # Optimization
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="Weight decay")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "exponential"],
                        help="Learning rate scheduler")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping value (0 = disabled)")

    # Logging
    parser.add_argument("--print-freq", type=int, default=10,
                        help="Print frequency")
    parser.add_argument("--save-freq", type=int, default=10,
                        help="Save checkpoint frequency")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")

    return parser.parse_args()


def generate_output_dir(base_dir: str, task: str, model_type: str, data_dir: str) -> str:
    """Generate unique output directory based on task, model, dataset and timestamp."""
    dataset_name = Path(data_dir).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{task}_{model_type}_{dataset_name}_{timestamp}"
    return str(Path(base_dir) / dir_name)


def get_default_weights(model_type: str) -> str:
    """Get default pretrained weights path for model type."""
    from dinov3_app.config import PRETRAINED_WEIGHTS
    weights = PRETRAINED_WEIGHTS.get(model_type)
    if weights is None:
        raise ValueError(f"No default weights for model type: {model_type}")
    return weights


def get_default_face_detector_config() -> dict:
    """Get default face detector configuration for segmentation."""
    return {
        'model_path': '/home/mi/Code/dinov3App/dms/onnx_model/face_detection.onnx',
        'device': 'cuda',
        'conf_thres': 0.55,
        'iou_thres': 0.3
    }


def main():
    args = parse_args()

    # Determine weights path
    weights_path = args.weights or get_default_weights(args.model_type)

    # Generate unique output directory
    output_dir = generate_output_dir(
        base_dir=args.output_dir,
        task=args.task,
        model_type=args.model_type,
        data_dir=args.data_dir,
    )

    # Create config
    config = Config(
        task=args.task,
        data_dir=args.data_dir,
        weights_path=weights_path,
        output_dir=output_dir,
        model_type=args.model_type,
        freeze_backbone=args.freeze_backbone,
        hidden_dim=args.hidden_dim,
        unfreeze_layers=args.unfreeze_layers,
        use_decoder=args.use_decoder,
        decoder_channels=args.decoder_channels,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        resize_size=args.resize,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        grad_clip=args.grad_clip,
        print_freq=args.print_freq,
        save_freq=args.save_freq,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        resume=args.resume,
    )

    # Add face detector config for segmentation
    if args.task == "segmentation":
        config.face_detector_config = get_default_face_detector_config()

    # Print configuration
    print("=" * 50)
    print(f"DINOv3 {args.task.upper()} Training")
    print("=" * 50)
    print(f"Dataset: {config.data_dir}")
    print(f"Output: {config.output_dir}")
    print(f"Model: {config.model_type}")
    print(f"Weights: {config.weights_path}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print("=" * 50)

    # Create trainer and train
    if args.task == "classification":
        trainer = ClassifierTrainer(config)
    else:
        trainer = SegmentationTrainer(config)
        model_info = trainer.get_model_info()
        print(f"Total params: {model_info['total_params']:,}")
        print(f"Trainable params: {model_info['trainable_params']:,}")
        print(f"Feature dim: {model_info['feature_dim']}")
        print(f"Patch size: {model_info['patch_size']}")
        print("=" * 50)

    trainer.train()


if __name__ == "__main__":
    main()
