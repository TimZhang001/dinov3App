#!/usr/bin/env python3
"""
Inference script for trained DINOv3 classifier.
"""

import argparse
import json

from dinov3_classifier.config import PRETRAINED_WEIGHTS
from dinov3_classifier.predictor import ClassifierPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Classification Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for inference")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to directory of images")
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Output JSON file")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k predictions for single image")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for directory processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--resize", type=int, default=256,
                        help="Image resize size")

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image and not args.image_dir:
        print("Error: Either --image or --image-dir must be specified")
        return

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model_type = checkpoint.get("model_type", "vits16")
        weights_path = PRETRAINED_WEIGHTS.get(model_type)
        print(f"Auto-selected weights for {model_type}: {weights_path}")

    # Create predictor
    predictor = ClassifierPredictor(
        checkpoint_path=args.checkpoint,
        weights_path=weights_path,
        device=args.device,
    )

    # Run inference
    if args.image:
        # Single image
        result = predictor.predict_single(
            args.image,
            top_k=args.top_k,
            resize_size=args.resize,
        )
        results = [result]
        predictor.print_prediction(result)
    else:
        # Directory of images
        results = predictor.predict_directory(
            args.image_dir,
            batch_size=args.batch_size,
            output_file=args.output,
        )

    # Save results for single image
    if args.image:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
