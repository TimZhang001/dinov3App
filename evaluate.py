#!/usr/bin/env python3
"""
Unified evaluation script for DINOv3 classification and segmentation.
"""

import argparse
from pathlib import Path

from dinov3_app.config import PRETRAINED_WEIGHTS
from dinov3_app.evaluator import ClassifierEvaluator, SegmentationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Evaluation Script")

    # Task type
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Task type: classification or segmentation")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select from checkpoint)")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")

    # Output
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")

    # Evaluation
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")

    # System
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def get_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """Extract model type from checkpoint."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("model_type", "vits16")


def main():
    args = parse_args()

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        model_type = get_model_type_from_checkpoint(args.checkpoint)
        weights_path = PRETRAINED_WEIGHTS.get(model_type)
        print(f"Auto-selected weights for {model_type}: {weights_path}")

    # Print configuration
    print("=" * 50)
    print(f"DINOv3 {args.task.upper()} Evaluation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Weights: {weights_path}")
    print(f"Dataset: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 50)

    # Create evaluator and run
    if args.task == "classification":
        evaluator = ClassifierEvaluator(
            checkpoint_path=args.checkpoint,
            weights_path=weights_path,
            output_dir=args.output_dir,
            device=args.device,
        )
        evaluator.run_full_evaluation(
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        evaluator = SegmentationEvaluator(
            checkpoint_path=args.checkpoint,
            weights_path=weights_path,
            device=args.device,
        )
        # Print model info
        model_info = evaluator.get_model_info()
        print(f"Model type: {model_info['model_type']}")
        print(f"Total params: {model_info['total_params']:,}")
        print(f"Feature dim: {model_info['feature_dim']}")
        print("=" * 50)

        metrics = evaluator.evaluate(
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
        )

        # Print results
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        print("=" * 50)
        print(f"\nEvaluation complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
