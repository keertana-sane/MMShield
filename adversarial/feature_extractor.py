"""
feature_extractor.py

CNN feature extraction module for the Patch Integrity Module.

This module extracts deep embeddings from candidate patch images using a
pretrained EfficientNet-B0 backbone (PyTorch). Extraction runs in batches
on GPU when available, falling back to CPU otherwise. Resulting embeddings
are saved as .npy files, one per input image (or optionally stacked into
a single array), for downstream fusion with handcrafted features.

Note on the multi-dataset pipeline:
    receipt_features.py calls CNNFeatureExtractor.extract_batch()
    directly and fuses embeddings in-memory — it does NOT go through
    process_directory() here. process_directory() and this module's
    standalone main() exist only as an independent debugging/inspection
    entry point (e.g. to precompute and inspect raw .npy embeddings for
    a specific dataset/split's candidate patches) and are not part of
    the dataset_builder.py pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import PATHS, IMAGE, CNN, LOGGING, ensure_directories


logger = logging.getLogger("feature_extractor")


def _configure_logging() -> None:
    """Configure module-level logging according to LOGGING settings."""
    level = getattr(logging, LOGGING.log_level.upper(), logging.INFO)
    handlers: List[logging.Handler] = []

    if LOGGING.log_to_console:
        handlers.append(logging.StreamHandler())

    if LOGGING.log_to_file:
        PATHS.logs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(PATHS.log_file))

    logging.basicConfig(
        level=level,
        format=LOGGING.log_format,
        datefmt=LOGGING.date_format,
        handlers=handlers,
        force=True,
    )


class _ImagePathDataset(Dataset):
    """
    Minimal PyTorch Dataset that loads and preprocesses images from a list
    of file paths for batched CNN inference.

    Attributes:
        image_paths: List of image file paths to load.
        transform: torchvision transform pipeline applied to each image.
    """

    def __init__(self, image_paths: List[Path], transform: transforms.Compose) -> None:
        """
        Initialize the dataset.

        Args:
            image_paths: List of image file paths.
            transform: torchvision transform pipeline to apply.
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load and transform a single image.

        Args:
            idx: Index into `image_paths`.

        Returns:
            Tuple[torch.Tensor, str]: The transformed image tensor and its
            source filename (stem).
        """
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        return tensor, path.stem


class CNNFeatureExtractor:
    """
    Extracts CNN embeddings from images using a pretrained EfficientNet-B0
    backbone, with the final classification layer removed so the model
    outputs pooled feature embeddings rather than class logits.

    Attributes:
        device: The torch device used for inference ("cuda" or "cpu").
        model: The EfficientNet-B0 feature-extraction backbone.
        transform: The preprocessing transform pipeline applied to input
            images before inference.
    """

    def __init__(self, device: Optional[str] = None) -> None:
        """
        Initialize the CNNFeatureExtractor and load the pretrained
        backbone onto the appropriate device.

        Args:
            device: Explicit device string ("cuda" or "cpu"). If None, the
                device is auto-selected based on CNN.device_preference and
                availability.
        """
        self.device = torch.device(device) if device else self._select_device()
        self.model = self._build_model().to(self.device)
        self.model.eval()
        self.transform = self._build_transform()

        logger.info(
            "CNNFeatureExtractor initialized with backbone=%s on device=%s",
            CNN.backbone,
            self.device,
        )

    @staticmethod
    def _select_device() -> torch.device:
        for preferred in CNN.device_preference:

            if preferred == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")

            if preferred == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")

            if preferred == "cpu":
                return torch.device("cpu")

        return torch.device("cpu")

    @staticmethod
    def _build_model() -> nn.Module:
        """
        Construct the EfficientNet-B0 backbone with the classification head
        replaced by an identity layer, so forward passes return pooled
        feature embeddings.

        Returns:
            nn.Module: The feature-extraction backbone.

        Raises:
            ValueError: If an unsupported backbone name is configured.
        """
        if CNN.backbone != "efficientnet_b0":
            raise ValueError(
                f"Unsupported backbone '{CNN.backbone}'. "
                f"Only 'efficientnet_b0' is currently implemented."
            )

        weights = (
            models.EfficientNet_B0_Weights.IMAGENET1K_V1 if CNN.pretrained else None
        )
        backbone = models.efficientnet_b0(weights=weights)
        # Replace the classifier with an identity layer to expose the
        # pooled 1280-dim embedding produced after global average pooling.
        backbone.classifier = nn.Identity()
        return backbone

    @staticmethod
    def _build_transform() -> transforms.Compose:
        """
        Build the preprocessing transform pipeline matching the CNN's
        expected input size and ImageNet normalization statistics.

        Returns:
            transforms.Compose: The preprocessing pipeline.
        """
        return transforms.Compose(
            [
                transforms.Resize(IMAGE.cnn_input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(CNN.normalization_mean),
                    std=list(CNN.normalization_std),
                ),
            ]
        )

    @torch.no_grad()
    def extract_single(self, image_path: Path) -> np.ndarray:
        """
        Extract a CNN embedding for a single image.

        Args:
            image_path: Path to the input image.

        Returns:
            np.ndarray: 1-D float32 embedding vector of length
            CNN.embedding_dim.

        Raises:
            FileNotFoundError: If the image path does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def extract_batch(
        self, image_paths: List[Path]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract CNN embeddings for a batch of images using a DataLoader
        for efficient batched inference.

        Args:
            image_paths: List of image file paths.

        Returns:
            Tuple[np.ndarray, List[str]]: A 2-D array of shape
            (N, embedding_dim) and the corresponding list of image stems
            (in the same order as the array rows).
        """
        if not image_paths:
            return np.empty((0, CNN.embedding_dim), dtype=np.float32), []

        dataset = _ImagePathDataset(image_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=CNN.batch_size,
            shuffle=False,
            num_workers=CNN.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        all_embeddings: List[np.ndarray] = []
        all_ids: List[str] = []

        for batch_tensors, batch_ids in loader:
            batch_tensors = batch_tensors.to(self.device)
            embeddings = self.model(batch_tensors)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
            all_ids.extend(batch_ids)

        stacked = np.concatenate(all_embeddings, axis=0)
        return stacked, all_ids

    def process_directory(
        self, image_dir: Path, output_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract CNN embeddings for every valid image in a directory and
        save both the stacked embedding array and per-image .npy files.

        Args:
            image_dir: Directory containing input images.
            output_dir: Directory to write .npy embedding files. Defaults
                to PATHS.features_dir / "cnn_embeddings".

        Returns:
            Tuple[np.ndarray, List[str]]: Stacked embeddings array of
            shape (N, embedding_dim) and the list of corresponding image
            stems.

        Raises:
            FileNotFoundError: If `image_dir` does not exist.
        """
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        output_dir = output_dir or (PATHS.features_dir / "cnn_embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE.valid_extensions
        )
        logger.info("Extracting CNN features for %d images in %s", len(image_paths), image_dir)

        embeddings, ids = self.extract_batch(image_paths)

        for embedding, image_id in zip(embeddings, ids):
            np.save(output_dir / f"{image_id}.npy", embedding)

        stacked_path = output_dir / "all_embeddings.npy"
        ids_path = output_dir / "all_embeddings_ids.npy"
        np.save(stacked_path, embeddings)
        np.save(ids_path, np.array(ids))

        logger.info(
            "Saved %d embeddings (dim=%d) to %s", len(ids), CNN.embedding_dim, output_dir
        )
        return embeddings, ids


def main() -> None:
    """Entry point for running CNN feature extraction standalone over a
    specific dataset/split's candidate patches directory (debugging /
    inspection only — not used by the dataset_builder.py pipeline,
    which calls extract_batch() directly via receipt_features.py)."""
    import argparse

    _configure_logging()
    ensure_directories()

    parser = argparse.ArgumentParser(
        description="Precompute and save CNN embeddings for a directory of candidate patches."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sroie",
        choices=["sroie", "cord", "funsd"],
        help="Dataset whose candidate patches to extract embeddings for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split's candidate patches to process (default: train).",
    )
    args = parser.parse_args()

    image_dir = PATHS.candidates_patches_dir / args.dataset / args.split
    output_dir = PATHS.features_dir / "cnn_embeddings" / args.dataset / args.split

    extractor = CNNFeatureExtractor()
    extractor.process_directory(image_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()