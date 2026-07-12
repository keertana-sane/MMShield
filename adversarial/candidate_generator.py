"""
candidate_generator.py

Candidate region proposal module for the Patch Integrity Module.

Given a single image (no access to any "clean" reference), this module
computes a set of low-level visual statistics maps (edge density, local
entropy, LBP texture response, gradient magnitude, visual saliency),
fuses them into a single threat score map, extracts connected components
above a threshold, and ranks them to produce the top-K most visually
suspicious candidate regions. These candidate regions are cropped and
passed downstream to feature extraction and classification.

This module performs no attack generation and no reference-image
comparison; it only analyzes intrinsic visual statistics of a single
input image.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk

from config import PATHS, IMAGE, CANDIDATE_GEN, NAMING, LOGGING, ensure_directories


logger = logging.getLogger("candidate_generator")


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


@dataclass
class CandidateRegion:
    """
    A single ranked candidate region identified within an image.

    Attributes:
        image_id: Filename (or identifier) of the source image.
        candidate_rank: 1-indexed rank of this candidate by threat score
            (1 = most suspicious).
        threat_score: Mean fused threat score within the region, in [0, 1].
        bbox_x: Top-left x coordinate of the bounding box (pixels).
        bbox_y: Top-left y coordinate of the bounding box (pixels).
        bbox_width: Width of the bounding box (pixels).
        bbox_height: Height of the bounding box (pixels).
        patch_filename: Filename of the cropped candidate patch image
            written to disk.
    """

    image_id: str
    candidate_rank: int
    threat_score: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    patch_filename: str


class CandidateGenerator:
    """
    Locates visually suspicious regions within a single image by fusing
    multiple low-level statistical score maps into a unified threat score
    map, then extracting and ranking connected components.

    Pipeline: load -> preprocess -> {edge density, local entropy, LBP
    texture, gradient magnitude, saliency} -> weighted fusion -> threshold
    -> connected components -> bounding boxes -> rank -> top-K crops.

    Attributes:
        output_patches_dir: Directory where cropped candidate patches are
            saved.
        candidates_csv_path: CSV file path where candidate metadata is
            written.
    """

    def __init__(
        self,
        output_patches_dir: Optional[Path] = None,
        candidates_csv_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize the CandidateGenerator.

        Args:
            output_patches_dir: Directory to save cropped candidate
                patches. Defaults to PATHS.candidates_patches_dir.
            candidates_csv_path: CSV path to save candidate metadata.
                Defaults to PATHS.candidates_csv.
        """
        self.output_patches_dir = output_patches_dir or PATHS.candidates_patches_dir
        self.candidates_csv_path = candidates_csv_path or PATHS.candidates_csv

        weight_sum = sum(CANDIDATE_GEN.score_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=1e-3):
            logger.warning(
                "Candidate score weights sum to %.3f, not 1.0. "
                "Scores will be normalized after fusion regardless.",
                weight_sum,
            )

    # ------------------------------------------------------------------
    # Loading and preprocessing
    # ------------------------------------------------------------------

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image from disk as a BGR numpy array and resize it to the
        configured target size.

        Args:
            image_path: Path to the image file.

        Returns:
            np.ndarray: BGR image array of shape (H, W, 3).

        Raises:
            FileNotFoundError: If the image path does not exist.
            ValueError: If the image cannot be decoded.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {image_path}")

        image = cv2.resize(image, IMAGE.target_size, interpolation=cv2.INTER_AREA)
        return image

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess a BGR image into a denoised grayscale image used as the
        basis for all downstream statistical score maps.

        Args:
            image: BGR image array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (grayscale image, denoised
            grayscale image), both uint8 arrays of shape (H, W).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        return gray, denoised

    # ------------------------------------------------------------------
    # Individual score maps
    # ------------------------------------------------------------------

    def compute_edge_density_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a local edge density map using Canny edge detection
        followed by local averaging.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Float32 edge density map normalized to [0, 1].
        """
        edges = cv2.Canny(
            gray,
            CANDIDATE_GEN.canny_threshold_low,
            CANDIDATE_GEN.canny_threshold_high,
        )
        edges_float = edges.astype(np.float32) / 255.0

        kernel_size = CANDIDATE_GEN.entropy_window_size
        density = cv2.boxFilter(
            edges_float, ddepth=-1, ksize=(kernel_size, kernel_size)
        )
        return self._normalize(density)

    def compute_local_entropy_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a local Shannon entropy map, highlighting regions with
        unusually high or low local information content.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Float32 entropy map normalized to [0, 1].
        """
        window_radius = max(1, CANDIDATE_GEN.entropy_window_size // 2)
        entropy_map = rank_entropy(gray, disk(window_radius))
        return self._normalize(entropy_map.astype(np.float32))

    def compute_lbp_texture_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a local texture irregularity map using Local Binary
        Patterns (LBP). The local variance of the LBP response highlights
        texture discontinuities characteristic of composited regions.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Float32 LBP texture variance map normalized to
            [0, 1].
        """
        lbp = local_binary_pattern(
            gray,
            P=CANDIDATE_GEN.lbp_num_points,
            R=CANDIDATE_GEN.lbp_radius,
            method=CANDIDATE_GEN.lbp_method,
        ).astype(np.float32)

        kernel_size = CANDIDATE_GEN.entropy_window_size
        mean = cv2.boxFilter(lbp, ddepth=-1, ksize=(kernel_size, kernel_size))
        mean_sq = cv2.boxFilter(lbp**2, ddepth=-1, ksize=(kernel_size, kernel_size))
        variance = np.clip(mean_sq - mean**2, a_min=0, a_max=None)
        return self._normalize(variance)

    def compute_gradient_magnitude_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Compute a Sobel gradient magnitude map, highlighting sharp
        boundaries such as patch edges.

        Args:
            gray: Grayscale image array.

        Returns:
            np.ndarray: Float32 gradient magnitude map normalized to
            [0, 1].
        """
        gray_f = gray.astype(np.float32)
        grad_x = cv2.Sobel(
            gray_f, cv2.CV_32F, 1, 0, ksize=CANDIDATE_GEN.sobel_kernel_size
        )
        grad_y = cv2.Sobel(
            gray_f, cv2.CV_32F, 0, 1, ksize=CANDIDATE_GEN.sobel_kernel_size
        )
        magnitude = cv2.magnitude(grad_x, grad_y)
        return self._normalize(magnitude)

    def compute_saliency_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a visual saliency map using OpenCV's spectral residual
        saliency detector, highlighting regions that visually "pop out"
        from their surroundings.

        Args:
            image: BGR image array.

        Returns:
            np.ndarray: Float32 saliency map normalized to [0, 1].
        """
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency_detector.computeSaliency(image)

        if not success:
            logger.warning("Saliency computation failed; returning a zero map.")
            return np.zeros(image.shape[:2], dtype=np.float32)

        return self._normalize(saliency_map.astype(np.float32))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """
        Min-max normalize an array to the [0, 1] range.

        Args:
            arr: Input array.

        Returns:
            np.ndarray: Normalized float32 array.
        """
        arr = arr.astype(np.float32)
        min_val, max_val = float(arr.min()), float(arr.max())
        if max_val - min_val < 1e-6:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)

    # ------------------------------------------------------------------
    # Fusion and region extraction
    # ------------------------------------------------------------------

    def compute_threat_score_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute and fuse all individual score maps into a single threat
        score map using the configured weights.

        Args:
            image: BGR image array.

        Returns:
            np.ndarray: Float32 fused threat score map normalized to
            [0, 1], same spatial dimensions as the input image.
        """
        gray, denoised = self.preprocess(image)

        edge_map = self.compute_edge_density_map(denoised)
        entropy_map = self.compute_local_entropy_map(denoised)
        lbp_map = self.compute_lbp_texture_map(denoised)
        gradient_map = self.compute_gradient_magnitude_map(denoised)
        saliency_map = self.compute_saliency_map(image)

        weights = CANDIDATE_GEN.score_weights
        fused = (
            weights.get("edge_density", 0.2) * edge_map
            + weights.get("entropy", 0.2) * entropy_map
            + weights.get("lbp_texture", 0.2) * lbp_map
            + weights.get("gradient_magnitude", 0.2) * gradient_map
            + weights.get("saliency", 0.2) * saliency_map
        )
        return self._normalize(fused)

    def extract_connected_components(
        self, threat_map: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Threshold the threat score map and extract connected component
        bounding boxes.

        Args:
            threat_map: Fused threat score map, normalized to [0, 1].

        Returns:
            List[Tuple[int, int, int, int, float]]: List of
            (x, y, width, height, mean_score) tuples, one per valid
            connected component.
        """
        binary_mask = (threat_map >= CANDIDATE_GEN.threat_score_threshold).astype(
            np.uint8
        )

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        components = []
        # Label 0 is the background; skip it.
        for label_idx in range(1, min(num_labels, CANDIDATE_GEN.max_components_considered + 1)):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])

            if area < CANDIDATE_GEN.min_candidate_area:
                continue

            region_mask = labels[y : y + h, x : x + w] == label_idx
            region_scores = threat_map[y : y + h, x : x + w][region_mask]
            mean_score = float(region_scores.mean()) if region_scores.size > 0 else 0.0

            components.append((x, y, w, h, mean_score))

        return components

    def rank_candidates(
        self, components: List[Tuple[int, int, int, int, float]]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Rank candidate components by mean threat score, descending, and
        keep only the top-K.

        Args:
            components: List of (x, y, width, height, mean_score) tuples.

        Returns:
            List[Tuple[int, int, int, int, float]]: Top-K ranked
            candidates, highest score first.
        """
        ranked = sorted(components, key=lambda c: c[4], reverse=True)
        return ranked[: CANDIDATE_GEN.top_k_candidates]

    def _crop_with_padding(
        self, image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """
        Crop a padded region from the image, clamped to image bounds.

        Args:
            image: Source BGR image array.
            x: Bounding box top-left x.
            y: Bounding box top-left y.
            w: Bounding box width.
            h: Bounding box height.

        Returns:
            np.ndarray: Cropped BGR image patch.
        """
        pad = CANDIDATE_GEN.candidate_padding
        img_h, img_w = image.shape[:2]

        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(img_w, x + w + pad)
        y1 = min(img_h, y + h + pad)

        return image[y0:y1, x0:x1].copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_image(self, image_path: Path) -> List[CandidateRegion]:
        """
        Run the full candidate generation pipeline on a single image:
        load, compute the fused threat score map, extract and rank
        connected components, crop and save the top-K candidate patches.

        Args:
            image_path: Path to the input image.

        Returns:
            List[CandidateRegion]: Ranked candidate regions (up to
            top_k_candidates), possibly empty if no region exceeded the
            threat score threshold.
        """
        self.output_patches_dir.mkdir(parents=True, exist_ok=True)

        image = self.load_image(image_path)
        threat_map = self.compute_threat_score_map(image)
        components = self.extract_connected_components(threat_map)
        top_components = self.rank_candidates(components)

        image_id = image_path.name
        stem = image_path.stem
        candidate_regions: List[CandidateRegion] = []

        for rank, (x, y, w, h, score) in enumerate(top_components, start=1):
            patch_crop = self._crop_with_padding(image, x, y, w, h)
            patch_filename = f"{stem}_candidate{rank:02d}.png"
            patch_path = self.output_patches_dir / patch_filename
            cv2.imwrite(str(patch_path), patch_crop)

            candidate_regions.append(
                CandidateRegion(
                    image_id=image_id,
                    candidate_rank=rank,
                    threat_score=round(score, 4),
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=w,
                    bbox_height=h,
                    patch_filename=patch_filename,
                )
            )

        if not candidate_regions:
            logger.info("No candidates above threshold for image: %s", image_id)
        else:
            logger.info(
                "Generated %d candidate(s) for image: %s (top score: %.4f)",
                len(candidate_regions),
                image_id,
                candidate_regions[0].threat_score,
            )

        return candidate_regions

    def process_directory(self, image_dir: Path) -> List[CandidateRegion]:
        """
        Run candidate generation over every valid image file in a
        directory and save all resulting metadata to the configured CSV.

        Args:
            image_dir: Directory containing input images.

        Returns:
            List[CandidateRegion]: All candidate regions generated across
            the directory.

        Raises:
            FileNotFoundError: If `image_dir` does not exist.
        """
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_paths = sorted(
            p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE.valid_extensions
        )
        logger.info("Processing %d images from %s", len(image_paths), image_dir)

        all_candidates: List[CandidateRegion] = []
        for image_path in image_paths:
            try:
                candidates = self.process_image(image_path)
                all_candidates.extend(candidates)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Skipping image %s: %s", image_path, exc)

        self.save_candidates(all_candidates)
        return all_candidates

    def save_candidates(self, candidates: List[CandidateRegion]) -> None:
        """
        Write candidate region metadata to the configured CSV path.

        Args:
            candidates: List of CandidateRegion instances to serialize.
        """
        self.candidates_csv_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            NAMING.image_id_column,
            NAMING.candidate_rank_column,
            NAMING.threat_score_column,
            *NAMING.bbox_columns,
            "patch_filename",
        ]

        with self.candidates_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates:
                row = asdict(candidate)
                writer.writerow(
                    {
                        NAMING.image_id_column: row["image_id"],
                        NAMING.candidate_rank_column: row["candidate_rank"],
                        NAMING.threat_score_column: row["threat_score"],
                        NAMING.bbox_columns[0]: row["bbox_x"],
                        NAMING.bbox_columns[1]: row["bbox_y"],
                        NAMING.bbox_columns[2]: row["bbox_width"],
                        NAMING.bbox_columns[3]: row["bbox_height"],
                        "patch_filename": row["patch_filename"],
                    }
                )

        logger.info(
            "Candidate metadata written to %s (%d rows)",
            self.candidates_csv_path,
            len(candidates),
        )


def main() -> None:
    """Entry point for running candidate generation standalone over the
    generated attacks directory."""
    _configure_logging()
    ensure_directories()
    generator = CandidateGenerator()
    generator.process_directory(PATHS.generated_attacks_images_dir)


if __name__ == "__main__":
    main()
