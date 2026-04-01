"""Inference pipeline for DINOv3 classifier."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm

from .config import IMAGENET_MEAN, IMAGENET_STD
from .model import DINOv3Classifier


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
