"""
attack_generator.py

Synthetic patch dataset generator for the Patch Integrity Module.

This module composites transparent PNG patches onto clean base images
using randomized geometric and photometric transformations (location,
size, rotation, opacity, brightness, contrast, perspective warp,
gaussian blur). It produces a labeled dataset of clean/attacked image
pairs along with metadata describing exactly how each patch was applied.

Multi-dataset + split support:
    Base images are now sourced from config.get_dataset_images(dataset,
    split) instead of being hardcoded to PATHS.sroie_dir. This lets the
    same generator be pointed at SROIE, CORD, or FUNSD, and at either
    the train or test split, without code changes. Output images are
    written to PATHS.generated_attacks_images_dir / dataset / split,
    and metadata to
    PATHS.generated_attacks_dir / f"{dataset}_{split}_metadata.csv",
    so train and test attack sets never share a folder or overwrite
    each other's metadata (mirroring the typographic module's
    train/test separation).

IMPORTANT: This module does NOT implement TRAP, SmoothPrompt, DPATCH, or any
adversarial optimization algorithm. It performs synthetic, randomized visual
modification for the purpose of training and evaluating a visual integrity
verification classifier (the downstream candidate/feature/classifier
pipeline). No gradient-based or optimization-based attack search is
performed anywhere in this file.
"""

from __future__ import annotations
from tqdm import tqdm

import csv
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from config import (
    PATHS,
    SEED,
    IMAGE,
    PATCH_GEN,
    NAMING,
    LOGGING,
    get_dataset_images,
    ensure_directories,
)


logger = logging.getLogger("attack_generator")


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
class PatchRecord:
    """
    A single metadata record describing one synthetic patch application.

    Attributes:
        clean_image: Filename of the original, unmodified base image.
        attacked_image: Filename of the generated patched image.
        patch: Filename of the patch asset used.
        x: Horizontal top-left coordinate (pixels) where the patch was
            composited onto the base image.
        y: Vertical top-left coordinate (pixels) where the patch was
            composited onto the base image.
        width: Width (pixels) of the patch after scaling.
        height: Height (pixels) of the patch after scaling.
        rotation: Rotation angle (degrees) applied to the patch.
        opacity: Alpha blending opacity applied to the patch (0-1).
        brightness: Brightness enhancement factor applied to the patch.
        contrast: Contrast enhancement factor applied to the patch.
        blur: Gaussian blur kernel size applied to the patch (0 = none).
        severity: Categorical severity bucket derived from the combined
            transformation strength ("low", "medium", "high").
        attack_type: Descriptive label for the category of synthetic patch.
    """

    clean_image: str
    attacked_image: str
    patch: str
    x: int
    y: int
    width: int
    height: int
    rotation: float
    opacity: float
    brightness: float
    contrast: float
    blur: int
    severity: str
    attack_type: str


class PatchDatasetGenerator:
    """
    Generates a synthetic dataset of patched (positive) images from a
    directory of clean base images and a library of transparent PNG
    patches, for a specific dataset + split.

    The generator applies randomized geometric transformations (scale,
    rotation, perspective warp) and photometric transformations (opacity,
    brightness, contrast, gaussian blur) to each patch before compositing
    it onto a randomly chosen location on the base image. Every generated
    image is accompanied by a metadata record capturing the exact
    transformation parameters used, enabling full reproducibility and
    ground-truth bounding boxes for downstream training and evaluation.

    Attributes:
        dataset: Which dataset this generator draws base images from
            (e.g. "sroie", "cord", "funsd").
        split: Which split ("train" or "test") this generator draws
            base images from.
        base_image_dir: Directory containing clean base images.
        patch_library_dir: Directory containing transparent PNG patch
            assets.
        output_dir: Directory where attacked images are written.
        metadata_path: CSV file path where metadata records are written.
        rng: A seeded `random.Random` instance for reproducible sampling.
    """

    def __init__(
        self,
        dataset: str = "sroie",
        split: str = "train",
        base_image_dir: Optional[Path] = None,
        patch_library_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
        seed: int = SEED.seed,
    ) -> None:
        """
        Initialize the PatchDatasetGenerator.

        Args:
            dataset: Which dataset to draw clean base images from. Must
                be a key in config.DATASETS ("sroie", "cord", "funsd").
            split: "train" or "test". Selects which clean-image split to
                read from config.DATASETS[dataset].
            base_image_dir: Directory of clean base images. Defaults to
                get_dataset_images(dataset, split). Overriding this
                bypasses the dataset/split registry entirely.
            patch_library_dir: Directory of transparent PNG patches.
                Defaults to PATHS.patch_library_dir.
            output_dir: Directory to write attacked images. Defaults to
                PATHS.generated_attacks_images_dir / dataset / split.
            metadata_path: CSV path to write metadata. Defaults to
                PATHS.generated_attacks_dir /
                f"{dataset}_{split}_metadata.csv".
            seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.split = split

        self.base_image_dir = base_image_dir or get_dataset_images(dataset, split)
        self.patch_library_dir = patch_library_dir or PATHS.patch_library_dir
        self.output_dir = (
            output_dir
            or PATHS.generated_attacks_images_dir / dataset / split
        )
        self.metadata_path = (
            metadata_path
            or PATHS.generated_attacks_dir / f"{dataset}_{split}_metadata.csv"
        )

        self.rng = random.Random(seed)
        np.random.seed(seed)

        self._patch_file_cache: Optional[List[Path]] = None

    def _discover_base_images(self) -> List[Path]:
        if not self.base_image_dir.exists():
            raise FileNotFoundError(
                f"Base image directory not found: {self.base_image_dir}"
            )

        images = sorted(
            p
            for p in self.base_image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE.valid_extensions
        )

        logger.info(
            "Discovered %d base images in %s",
            len(images),
            self.base_image_dir,
        )

        return images

    def _discover_patch_files(self) -> List[Path]:
        """
        Scan `patch_library_dir` for transparent PNG patch assets.

        Returns:
            List[Path]: Sorted list of patch file paths.

        Raises:
            FileNotFoundError: If the patch library directory does not
                exist or contains no patch files.
        """
        if self._patch_file_cache is not None:
            return self._patch_file_cache

        if not self.patch_library_dir.exists():
            raise FileNotFoundError(
                f"Patch library directory not found: {self.patch_library_dir}"
            )

        patches = sorted(
            self.patch_library_dir.glob(f"*{PATCH_GEN.patch_file_extension}")
        )
        if not patches:
            raise FileNotFoundError(
                f"No patch files with extension "
                f"'{PATCH_GEN.patch_file_extension}' found in "
                f"{self.patch_library_dir}"
            )

        logger.info("Discovered %d patch assets in %s", len(patches), self.patch_library_dir)
        self._patch_file_cache = patches
        return patches

    def load_patch(self, patch_path: Path) -> Image.Image:
        """
        Load a transparent PNG patch asset from disk.

        Args:
            patch_path: Path to the patch PNG file.

        Returns:
            Image.Image: The patch image in RGBA mode.

        Raises:
            FileNotFoundError: If the patch file does not exist.
            ValueError: If the patch file cannot be opened as an image.
        """
        if not patch_path.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        try:
            patch = Image.open(patch_path).convert("RGBA")
        except Exception as exc:
            raise ValueError(f"Failed to load patch image {patch_path}: {exc}") from exc

        return patch

    def random_transform(
        self, patch: Image.Image, base_size: Tuple[int, int]
    ) -> Tuple[Image.Image, dict]:
        """
        Apply a randomized set of geometric and photometric transformations
        to a patch image.

        Transformations applied, in order: scale (relative to base image
        size), brightness, contrast, gaussian blur, opacity, perspective
        warp, and rotation.

        Args:
            patch: The source patch image (RGBA).
            base_size: (width, height) of the base image the patch will be
                composited onto, used to compute relative patch scale.

        Returns:
            Tuple[Image.Image, dict]: The transformed patch (RGBA) and a
            dictionary of the transformation parameters applied.
        """
        base_w, base_h = base_size
        shorter_edge = min(base_w, base_h)

        # --- Scale ---
        scale_fraction = self.rng.uniform(*PATCH_GEN.patch_scale_range)
        target_dim = max(8, int(shorter_edge * scale_fraction))
        aspect_ratio = patch.height / patch.width if patch.width > 0 else 1.0
        new_w = target_dim
        new_h = max(8, int(target_dim * aspect_ratio))
        patch = patch.resize((new_w, new_h), resample=Image.LANCZOS)

        # --- Brightness ---
        brightness = self.rng.uniform(*PATCH_GEN.brightness_range)
        patch = ImageEnhance.Brightness(patch).enhance(brightness)

        # --- Contrast ---
        contrast = self.rng.uniform(*PATCH_GEN.contrast_range)
        patch = ImageEnhance.Contrast(patch).enhance(contrast)

        # --- Gaussian blur ---
        blur_kernel = self.rng.choice(PATCH_GEN.gaussian_blur_kernel_choices)
        if blur_kernel > 0:
            patch = patch.filter(ImageFilter.GaussianBlur(radius=blur_kernel / 2.0))

        # --- Opacity ---
        opacity = self.rng.uniform(*PATCH_GEN.opacity_range)
        patch = self._apply_opacity(patch, opacity)

        # --- Perspective warp ---
        warp_strength = self.rng.uniform(*PATCH_GEN.perspective_warp_range)
        patch = self._apply_perspective_warp(patch, warp_strength)

        # --- Rotation (last, so warped canvas rotates as a whole) ---
        rotation = self.rng.uniform(*PATCH_GEN.rotation_range_degrees)
        patch = patch.rotate(rotation, expand=True, resample=Image.BICUBIC)

        params = {
            "width": patch.width,
            "height": patch.height,
            "rotation": round(rotation, 2),
            "opacity": round(opacity, 3),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "blur": int(blur_kernel),
            "perspective_warp": round(warp_strength, 3),
        }
        return patch, params

    @staticmethod
    def _apply_opacity(patch: Image.Image, opacity: float) -> Image.Image:
        """
        Scale the alpha channel of an RGBA patch by a given opacity factor.

        Args:
            patch: RGBA patch image.
            opacity: Opacity multiplier in [0, 1].

        Returns:
            Image.Image: The patch with adjusted alpha channel.
        """
        r, g, b, a = patch.split()
        a = a.point(lambda px: int(px * opacity))
        return Image.merge("RGBA", (r, g, b, a))

    @staticmethod
    def _apply_perspective_warp(patch: Image.Image, strength: float) -> Image.Image:
        """
        Apply a random perspective warp to a patch by jittering its corner
        points and resampling via PIL's PERSPECTIVE transform.

        Args:
            patch: RGBA patch image.
            strength: Maximum corner jitter as a fraction of the patch's
                width/height. A strength of 0 leaves the patch unchanged.

        Returns:
            Image.Image: The perspective-warped patch.
        """
        if strength <= 0:
            return patch

        w, h = patch.size
        if w < 2 or h < 2:
            return patch

        def jitter(max_x: float, max_y: float) -> Tuple[float, float]:
            return (
                random.uniform(-max_x, max_x),
                random.uniform(-max_y, max_y),
            )

        max_dx = w * strength
        max_dy = h * strength

        # Original corners.
        src_corners = [(0, 0), (w, 0), (w, h), (0, h)]
        # Jittered destination corners.
        dst_corners = []
        for cx, cy in src_corners:
            jx, jy = jitter(max_dx, max_dy)
            dst_corners.append((cx + jx, cy + jy))

        coeffs = PatchDatasetGenerator._find_perspective_coeffs(dst_corners, src_corners)
        try:
            warped = patch.transform(
                (w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC
            )
        except Exception:
            # If the perspective solve is degenerate, fall back to the
            # unwarped patch rather than failing dataset generation.
            return patch
        return warped

    @staticmethod
    def _find_perspective_coeffs(
        source_coords: List[Tuple[float, float]],
        target_coords: List[Tuple[float, float]],
    ) -> List[float]:
        """
        Solve for the 8 coefficients of a perspective transform mapping
        `target_coords` -> `source_coords`, compatible with
        PIL's Image.transform(..., Image.PERSPECTIVE, coeffs).

        Args:
            source_coords: Four (x, y) points in the source image.
            target_coords: Four (x, y) points in the target image.

        Returns:
            List[float]: The 8 perspective transform coefficients.
        """
        matrix = []
        for (sx, sy), (tx, ty) in zip(source_coords, target_coords):
            matrix.append([tx, ty, 1, 0, 0, 0, -sx * tx, -sx * ty])
            matrix.append([0, 0, 0, tx, ty, 1, -sy * tx, -sy * ty])

        a = np.array(matrix, dtype=np.float64)
        b = np.array(source_coords, dtype=np.float64).reshape(8)

        result = np.linalg.solve(a, b)
        return result.tolist()

    def apply_patch(
        self, base_image: Image.Image, patch: Image.Image
    ) -> Tuple[Image.Image, int, int]:
        """
        Composite a (pre-transformed) patch onto a randomly chosen location
        on the base image using alpha blending.

        Args:
            base_image: The clean base image (RGB).
            patch: The transformed patch image (RGBA).

        Returns:
            Tuple[Image.Image, int, int]: The resulting composited image
            (RGB) and the (x, y) top-left coordinates where the patch was
            placed.
        """
        base_w, base_h = base_image.size
        patch_w, patch_h = patch.size

        # Clamp patch size to fit within the base image if necessary.
        if patch_w >= base_w or patch_h >= base_h:
            scale = min((base_w - 2) / patch_w, (base_h - 2) / patch_h)
            scale = max(scale, 0.05)
            patch = patch.resize(
                (max(1, int(patch_w * scale)), max(1, int(patch_h * scale))),
                resample=Image.LANCZOS,
            )
            patch_w, patch_h = patch.size

        max_x = max(0, base_w - patch_w)
        max_y = max(0, base_h - patch_h)
        x = self.rng.randint(0, max_x)
        y = self.rng.randint(0, max_y)

        composited = base_image.convert("RGBA")
        composited.alpha_composite(patch, dest=(x, y))
        composited = composited.convert("RGB")

        return composited, x, y

    @staticmethod
    def _compute_severity(params: dict) -> str:
        """
        Derive a categorical severity label from the combined strength of
        the applied transformations.

        The severity score aggregates normalized deviations of opacity,
        rotation, blur, and perspective warp from their "neutral" values,
        then buckets the result into low/medium/high.

        Args:
            params: Transformation parameter dictionary produced by
                `random_transform`.

        Returns:
            str: One of "low", "medium", "high".
        """
        opacity_score = params["opacity"]
        rotation_score = min(abs(params["rotation"]) / 30.0, 1.0)
        blur_score = min(params["blur"] / 7.0, 1.0)
        warp_score = min(params["perspective_warp"] / 0.15, 1.0)

        combined = np.mean([opacity_score, rotation_score, blur_score, warp_score])

        if combined < 0.33:
            return "low"
        elif combined < 0.66:
            return "medium"
        return "high"

    def process_dataset(self) -> List[PatchRecord]:
        """
        Run the full synthetic patch dataset generation pipeline: for every
        clean base image, generate `PATCH_GEN.patched_images_per_receipt`
        attacked variants using randomly sampled patches and
        transformations, save the resulting images, and collect metadata
        records.

        Returns:
            List[PatchRecord]: All metadata records generated in this run.
        """
        ensure_directories()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        base_images = self._discover_base_images()
        patch_files = self._discover_patch_files()

        if not base_images:
            logger.warning("No base images found; nothing to generate.")
            return []

        records: List[PatchRecord] = []

        for base_path in tqdm(
            base_images,
            desc=f"Generating Patch Dataset ({self.dataset}/{self.split})",
        ):
            try:
                base_image = Image.open(base_path).convert("RGB")
            except Exception as exc:
                logger.warning("Skipping unreadable base image %s: %s", base_path, exc)
                continue

            base_image = base_image.resize(IMAGE.target_size, resample=Image.LANCZOS)

            for variant_idx in range(PATCH_GEN.patched_images_per_receipt):
                patch_path = self.rng.choice(patch_files)
                attack_type = self.rng.choice(PATCH_GEN.attack_type_labels)

                try:
                    raw_patch = self.load_patch(patch_path)
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning("Skipping patch %s: %s", patch_path, exc)
                    continue

                transformed_patch, params = self.random_transform(
                    raw_patch, base_image.size
                )
                attacked_image, x, y = self.apply_patch(base_image, transformed_patch)

                severity = self._compute_severity(params)

                attacked_filename = (
                    f"{base_path.stem}_attack{variant_idx:02d}{IMAGE.valid_extensions[0]}"
                )
                attacked_path = self.output_dir / attacked_filename
                attacked_image.save(attacked_path, quality=IMAGE.jpeg_quality)

                record = PatchRecord(
                    clean_image=base_path.name,
                    attacked_image=attacked_filename,
                    patch=patch_path.name,
                    x=x,
                    y=y,
                    width=params["width"],
                    height=params["height"],
                    rotation=params["rotation"],
                    opacity=params["opacity"],
                    brightness=params["brightness"],
                    contrast=params["contrast"],
                    blur=params["blur"],
                    severity=severity,
                    attack_type=attack_type,
                )
                records.append(record)

            logger.info("Generated variants for base image: %s", base_path.name)

        self.save_metadata(records)
        logger.info(
            "Dataset generation complete. %d attacked images written to %s",
            len(records),
            self.output_dir,
        )
        return records

    def save_metadata(self, records: List[PatchRecord]) -> None:
        """
        Write metadata records to the configured CSV path.

        Args:
            records: List of PatchRecord instances to serialize.
        """
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            NAMING.clean_image_column,
            NAMING.attacked_image_column,
            NAMING.patch_column,
            NAMING.x_column,
            NAMING.y_column,
            NAMING.width_column,
            NAMING.height_column,
            NAMING.rotation_column,
            NAMING.opacity_column,
            NAMING.brightness_column,
            NAMING.contrast_column,
            NAMING.blur_column,
            NAMING.severity_column,
            NAMING.attack_type_column,
        ]

        with self.metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = asdict(record)
                writer.writerow(
                    {
                        NAMING.clean_image_column: row["clean_image"],
                        NAMING.attacked_image_column: row["attacked_image"],
                        NAMING.patch_column: row["patch"],
                        NAMING.x_column: row["x"],
                        NAMING.y_column: row["y"],
                        NAMING.width_column: row["width"],
                        NAMING.height_column: row["height"],
                        NAMING.rotation_column: row["rotation"],
                        NAMING.opacity_column: row["opacity"],
                        NAMING.brightness_column: row["brightness"],
                        NAMING.contrast_column: row["contrast"],
                        NAMING.blur_column: row["blur"],
                        NAMING.severity_column: row["severity"],
                        NAMING.attack_type_column: row["attack_type"],
                    }
                )

        logger.info("Metadata written to %s (%d rows)", self.metadata_path, len(records))


def main() -> None:
    """Entry point for running the synthetic patch dataset generator standalone."""
    import argparse

    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Generate a synthetic adversarial-patch dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sroie",
        choices=["sroie", "cord", "funsd"],
        help="Dataset to generate patch attacks for.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which split's clean images to attack (default: train).",
    )
    args = parser.parse_args()

    generator = PatchDatasetGenerator(dataset=args.dataset, split=args.split)
    generator.process_dataset()


if __name__ == "__main__":
    main()