"""
receipt_features.py

Feature fusion module for the Patch Integrity Module.

Combines deep CNN embeddings (from feature_extractor.py) with handcrafted
visual statistics -- local entropy, edge density, LBP texture histogram,
gradient statistics, and color statistics -- computed directly from each
candidate patch image, into a single flat feature vector per patch. The
resulting vectors are the input representation used for classifier
training in train.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from config import PATHS, IMAGE, HANDCRAFTED, LOGGING, ensure_directories
from feature_extractor import CNNFeatureExtractor


logger = logging.getLogger("receipt_features")


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


class ReceiptFeatureFuser:
    """
    Computes handcrafted visual statistics for candidate patch images and
    fuses them with pretrained CNN embeddings into a single feature
    vector per patch, suitable for classical ML classifiers.

    Handcrafted feature groups:
        - Local entropy statistics (mean, std, max)
        - Edge density statistics (mean, std, density ratio)
        - LBP texture histogram (normalized bin counts)
        - Gradient magnitude/orientation statistics
        - Color statistics per channel, in RGB and HSV color spaces

    Attributes:
        cnn_extractor: A CNNFeatureExtractor instance used to compute deep
            embeddings for each patch.
    """

    def __init__(self, cnn_extractor: Optional[CNNFeatureExtractor] = None) -> None:
        """
        Initialize the ReceiptFeatureFuser.

        Args:
            cnn_extractor: An optional pre-constructed CNNFeatureExtractor.
                If None, a new one is instantiated (loading the pretrained
                backbone onto the best available device).
        """
        self.cnn_extractor = cnn_extractor or CNNFeatureExtractor()

    # ------------------------------------------------------------------
    # Handcrafted feature groups
    # ------------------------------------------------------------------

    def compute_entropy_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Compute local Shannon entropy statistics for a grayscale patch.

        Args:
            gray: Grayscale image array (uint8).

        Returns:
            Dict[str, float]: Dictionary of entropy summary statistics.
        """
        window_radius = max(1, HANDCRAFTED.entropy_window_size // 2)
        entropy_map = rank_entropy(gray, disk(window_radius)).astype(np.float32)

        return {
            "entropy_mean": float(entropy_map.mean()),
            "entropy_std": float(entropy_map.std()),
            "entropy_max": float(entropy_map.max()),
        }

    def compute_edge_density_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Compute edge density statistics for a grayscale patch using Canny
        edge detection.

        Args:
            gray: Grayscale image array (uint8).

        Returns:
            Dict[str, float]: Dictionary of edge density summary
            statistics.
        """
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

        return {
            "edge_density_mean": float(edges.astype(np.float32).mean() / 255.0),
            "edge_density_ratio": edge_ratio,
        }

    def compute_lbp_histogram_features(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Compute a normalized Local Binary Pattern histogram capturing the
        local texture distribution of a grayscale patch.

        Args:
            gray: Grayscale image array (uint8).

        Returns:
            Dict[str, float]: Dictionary mapping `lbp_bin_{i}` to the
            normalized frequency of that LBP bin.
        """
        lbp = local_binary_pattern(
            gray,
            P=HANDCRAFTED.lbp_num_points,
            R=HANDCRAFTED.lbp_radius,
            method="uniform",
        )

        n_bins = HANDCRAFTED.lbp_histogram_bins
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, HANDCRAFTED.lbp_num_points + 2)
        )
        hist = hist.astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum

        return {f"lbp_bin_{i}": float(v) for i, v in enumerate(hist)}

    def compute_gradient_statistics(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Compute gradient magnitude and orientation statistics for a
        grayscale patch using Sobel operators.

        Args:
            gray: Grayscale image array (uint8).

        Returns:
            Dict[str, float]: Dictionary of gradient summary statistics,
            including a coarse orientation histogram.
        """
        gray_f = gray.astype(np.float32)
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(grad_x, grad_y)
        orientation = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        features: Dict[str, float] = {
            "gradient_magnitude_mean": float(magnitude.mean()),
            "gradient_magnitude_std": float(magnitude.std()),
            "gradient_magnitude_max": float(magnitude.max()),
        }

        n_bins = HANDCRAFTED.gradient_bins
        hist, _ = np.histogram(
            orientation.ravel(),
            bins=n_bins,
            range=(0, 360),
            weights=magnitude.ravel(),
        )
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist = hist / hist_sum

        for i, v in enumerate(hist):
            features[f"gradient_orientation_bin_{i}"] = float(v)

        return features

    def compute_color_statistics(self, image_bgr: np.ndarray) -> Dict[str, float]:
        """
        Compute per-channel color statistics (mean, std) in both RGB and
        HSV color spaces.

        Args:
            image_bgr: BGR image array (uint8).

        Returns:
            Dict[str, float]: Dictionary of per-channel color statistics.
        """
        features: Dict[str, float] = {}

        if "RGB" in HANDCRAFTED.color_spaces:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            for idx, channel_name in enumerate(("r", "g", "b")):
                channel = rgb[:, :, idx].astype(np.float32)
                features[f"color_rgb_{channel_name}_mean"] = float(channel.mean())
                features[f"color_rgb_{channel_name}_std"] = float(channel.std())

        if "HSV" in HANDCRAFTED.color_spaces:
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            for idx, channel_name in enumerate(("h", "s", "v")):
                channel = hsv[:, :, idx].astype(np.float32)
                features[f"color_hsv_{channel_name}_mean"] = float(channel.mean())
                features[f"color_hsv_{channel_name}_std"] = float(channel.std())

        return features

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def compute_handcrafted_features(self, image_path: Path) -> Dict[str, float]:
        """
        Compute the full set of handcrafted features for a single patch
        image.

        Args:
            image_path: Path to the patch image.

        Returns:
            Dict[str, float]: Combined dictionary of all handcrafted
            feature groups.

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the image cannot be decoded.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        features: Dict[str, float] = {}
        features.update(self.compute_entropy_features(gray))
        features.update(self.compute_edge_density_features(gray))
        features.update(self.compute_lbp_histogram_features(gray))
        features.update(self.compute_gradient_statistics(gray))
        features.update(self.compute_color_statistics(image_bgr))
        return features

    def fuse_features(
        self, image_path: Path, cnn_embedding: np.ndarray
    ) -> Dict[str, float]:
        """
        Fuse handcrafted features and a precomputed CNN embedding into a
        single flat feature dictionary for one patch image.

        Args:
            image_path: Path to the patch image.
            cnn_embedding: 1-D CNN embedding vector for this patch.

        Returns:
            Dict[str, float]: Combined feature dictionary, with CNN
            embedding dimensions prefixed `cnn_{i}`.
        """
        handcrafted = self.compute_handcrafted_features(image_path)
        cnn_features = {f"cnn_{i}": float(v) for i, v in enumerate(cnn_embedding)}

        combined = {"image_id": image_path.stem}
        combined.update(cnn_features)
        combined.update(handcrafted)
        return combined

    def process_directory(
        self, image_dir: Path, output_csv: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Compute fused features for every valid image in a directory and
        save the result as a CSV file.

        Args:
            image_dir: Directory containing candidate patch images.
            output_csv: Path to write the combined features CSV. Defaults
                to PATHS.features_dir / "receipt_features.csv".

        Returns:
            pd.DataFrame: The fused feature table, one row per image.

        Raises:
            FileNotFoundError: If `image_dir` does not exist.
        """
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        ensure_directories()
        output_csv = output_csv or (PATHS.features_dir / "receipt_features.csv")

        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE.valid_extensions
        )
        logger.info("Fusing features for %d images in %s", len(image_paths), image_dir)

        if not image_paths:
            logger.warning("No images found in %s; nothing to fuse.", image_dir)
            empty_df = pd.DataFrame()
            empty_df.to_csv(output_csv, index=False)
            return empty_df

        cnn_embeddings, cnn_ids = self.cnn_extractor.extract_batch(image_paths)
        embedding_lookup = dict(zip(cnn_ids, cnn_embeddings))

        rows: List[Dict[str, float]] = []
        for image_path in image_paths:
            embedding = embedding_lookup.get(image_path.stem)
            if embedding is None:
                logger.warning(
                    "No CNN embedding found for %s; skipping.", image_path.name
                )
                continue
            try:
                fused = self.fuse_features(image_path, embedding)
                rows.append(fused)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping image %s: %s", image_path, exc)

        df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

        logger.info(
            "Fused feature table written to %s (%d rows, %d columns)",
            output_csv,
            df.shape[0],
            df.shape[1],
        )
        return df


def main() -> None:
    """Entry point for running feature fusion standalone over the
    generated candidate patches directory."""
    _configure_logging()
    ensure_directories()
    fuser = ReceiptFeatureFuser()
    fuser.process_directory(PATHS.candidates_patches_dir)


if __name__ == "__main__":
    main()
