#!/usr/bin/env python3
"""
Unified inference script for DINOv3 classification and segmentation.
"""

import argparse
import json
from pathlib import Path

from dinov3_app.config import PRETRAINED_WEIGHTS
from dinov3_app.predictor import ClassifierPredictor, SegmentationPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="DINOv3 Inference Script")

    # Task type
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "segmentation"],
                        help="Task type: classification or segmentation")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to pretrained weights (default: auto-select from checkpoint)")

    # Input
    parser.add_argument("--image", type=str, default=None,
                        help="Path to single image for inference")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to directory of images")

    # Output
    parser.add_argument("--output", type=str, default="predictions.json",
                        help="Output file for classification / Output directory for segmentation")
    parser.add_argument("--output-dir", type=str, default="segmentation_results",
                        help="Output directory for segmentation batch processing")

    # Classification-specific
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k predictions for single image classification")

    # Segmentation-specific
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Segmentation threshold")
    parser.add_argument("--save-visualization", action="store_true", default=True,
                        help="Save visualization results for segmentation")

    # Common
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for directory processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--resize", type=int, default=256,
                        help="Image resize size")

    return parser.parse_args()


def get_model_type_from_checkpoint(checkpoint_path: str) -> str:
    """Extract model type from checkpoint."""
    import torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("model_type", "vits16")


def run_classification(args):
    """Run classification inference."""
    if not args.image and not args.image_dir:
        print("Error: Either --image or --image-dir must be specified")
        return

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        model_type = get_model_type_from_checkpoint(args.checkpoint)
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
        predictor.print_prediction(result)
        results = [result]
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


def run_segmentation(args):
    """Run segmentation inference."""
    if args.image is None and args.image_dir is None:
        print("Error: Either --image or --image-dir must be specified")
        return

    if args.image is not None and args.image_dir is not None:
        print("Error: Cannot specify both --image and --image-dir")
        return

    # Determine weights path
    weights_path = args.weights
    if weights_path is None:
        model_type = get_model_type_from_checkpoint(args.checkpoint)
        weights_path = PRETRAINED_WEIGHTS.get(model_type)
        print(f"Auto-selected weights for {model_type}: {weights_path}")

    # Create predictor
    predictor = SegmentationPredictor(
        checkpoint_path=args.checkpoint,
        weights_path=weights_path,
        device=args.device,
    )
    predictor.set_threshold(args.threshold)
    predictor.set_batch_size(args.batch_size)

    # Print model info
    model_info = predictor.get_model_info()
    print("=" * 50)
    print("DINOv3 Segmentation Inference")
    print("=" * 50)
    print(f"Model type: {model_info['model_type']}")
    print(f"Total params: {model_info['total_params']:,}")
    print(f"Feature dim: {model_info['feature_dim']}")
    print(f"Threshold: {args.threshold}")
    print("=" * 50)

    if args.image:
        # Single image
        print(f"Processing: {args.image}")
        result = predictor.predict_single(args.image, args.threshold)

        # Save result
        output_path = Path(args.output) if args.output else Path(args.image).with_name(
            f"{Path(args.image).stem}_segmentation.png"
        )
        predictor._save_prediction_result(
            result, Path(args.image), output_path.parent,
            args.save_visualization
        )

        print(f"Inference time: {result['inference_time']:.4f}s")
        print(f"Face region ratio: {result['binary_mask'].mean():.4f}")
        print(f"Result saved to: {output_path}")
    else:
        # Directory
        print(f"Processing directory: {args.image_dir}")
        stats = predictor.predict_directory(
            input_dir=args.image_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            save_visualization=args.save_visualization,
        )

        print("\n" + "=" * 50)
        print("Processing Statistics")
        print("=" * 50)
        print(f"Total images: {stats['total_images']}")
        print(f"Processed: {stats['processed_images']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Avg time: {stats['average_time']:.4f}s/image")
        print("=" * 50)


def main():
    args = parse_args()

    if args.task == "classification":
        run_classification(args)
    else:
        run_segmentation(args)


if __name__ == "__main__":
    main()
