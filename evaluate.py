#!/usr/bin/env python3
"""
Evaluation script for trained DINOv3 classifier.
"""

import argparse

from dinov3_classifier.config import PRETRAINED_WEIGHTS
from dinov3_classifier.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Classification Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select from checkpoint)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model_type = checkpoint.get("model_type", "vits16")
        weights_path = PRETRAINED_WEIGHTS.get(model_type)
        print(f"Auto-selected weights for {model_type}: {weights_path}")

    # Evaluate
    evaluator = Evaluator(
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


if __name__ == "__main__":
    main()
