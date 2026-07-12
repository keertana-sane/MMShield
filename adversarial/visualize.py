"""
visualize.py

Visualization module for the Patch Integrity Module.

Generates human-readable diagnostic outputs from prediction results:
annotated images with bounding boxes and threat scores, threat score
heatmaps overlaid on the source image, and composite top-3 candidate
region visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from config import PATHS, IMAGE, VISUALIZATION, LOGGING, ensure_directories
from candidate_generator import CandidateGenerator
from predict import ImagePrediction, CandidatePrediction


logger = logging.getLogger("visualize")


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


_COLORMAP_LOOKUP = {
    "JET": cv2.COLORMAP_JET,
    "HOT": cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "TURBO": cv2.COLORMAP_TURBO,
}


class Visualizer:
    """
    Produces diagnostic visualizations for Patch Integrity Module
    predictions: annotated bounding-box overlays, threat score heatmaps,
    and composite top-3 candidate region panels.

    Attributes:
        candidate_generator: A CandidateGenerator instance, reused to
            recompute the fused threat score map for heatmap generation.
        annotated_dir: Directory where annotated images are saved.
        heatmaps_dir: Directory where heatmap images are saved.
    """

    def __init__(
        self,
        candidate_generator: Optional[CandidateGenerator] = None,
        annotated_dir: Optional[Path] = None,
        heatmaps_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            candidate_generator: Optional shared CandidateGenerator
                instance. If None, a new one is created.
            annotated_dir: Directory to save annotated images. Defaults
                to PATHS.annotated_dir.
            heatmaps_dir: Directory to save heatmap images. Defaults to
                PATHS.heatmaps_dir.
        """
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.annotated_dir = annotated_dir or PATHS.annotated_dir
        self.heatmaps_dir = heatmaps_dir or PATHS.heatmaps_dir

    def _load_and_resize(self, image_path: Path) -> np.ndarray:
        """
        Load an image and resize it to the configured target size, so
        that visualization coordinates align with pipeline bounding boxes.

        Args:
            image_path: Path to the source image.

        Returns:
            np.ndarray: BGR image array.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return cv2.resize(image, IMAGE.target_size, interpolation=cv2.INTER_AREA)

    def annotate_image(
        self, image_path: Path, prediction: ImagePrediction, output_path: Optional[Path] = None
    ) -> Path:
        """
        Draw bounding boxes and threat scores for every candidate region
        onto a copy of the source image, and save the annotated result.

        Args:
            image_path: Path to the source image.
            prediction: The ImagePrediction result for this image.
            output_path: Destination path. Defaults to
                `annotated_dir / f"{image_path.stem}_annotated.png"`.

        Returns:
            Path: The path the annotated image was saved to.
        """
        image = self._load_and_resize(image_path)
        color = VISUALIZATION.bounding_box_color[::-1]  # RGB -> BGR
        thickness = VISUALIZATION.bounding_box_thickness

        for candidate in prediction.candidates:
            top_left = (candidate.bbox_x, candidate.bbox_y)
            bottom_right = (
                candidate.bbox_x + candidate.bbox_width,
                candidate.bbox_y + candidate.bbox_height,
            )
            box_color = color if candidate.is_patch else (128, 128, 128)
            cv2.rectangle(image, top_left, bottom_right, box_color, thickness)

            label = f"#{candidate.candidate_rank} {candidate.threat_probability:.2f}"
            self._draw_label(image, label, (candidate.bbox_x, candidate.bbox_y - 6))

        banner = f"Prediction: {prediction.prediction.upper()} (confidence={prediction.confidence:.2f})"
        self._draw_label(image, banner, (10, 24), scale_multiplier=1.2)

        output_path = output_path or (
            self.annotated_dir / f"{image_path.stem}_annotated.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

        logger.info("Annotated image saved to %s", output_path)
        return output_path

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple,
        scale_multiplier: float = 1.0,
    ) -> None:
        """
        Draw a text label with a filled background rectangle for
        readability at an arbitrary position on an image, in place.

        Args:
            image: BGR image array to draw on (modified in place).
            text: Label text.
            origin: (x, y) bottom-left origin for the text.
            scale_multiplier: Multiplier applied to the configured base
                font scale.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = VISUALIZATION.font_scale * scale_multiplier
        thickness = 1

        x, y = origin
        y = max(y, 12)

        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        bg_color = VISUALIZATION.text_background_color[::-1]
        text_color = VISUALIZATION.text_color[::-1]

        cv2.rectangle(
            image,
            (x, y - text_h - baseline),
            (x + text_w, y + baseline),
            bg_color,
            thickness=-1,
        )
        cv2.putText(image, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

    def generate_heatmap(
        self, image_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """
        Compute the fused threat score map for an image and overlay it as
        a color heatmap on top of the original image.

        Args:
            image_path: Path to the source image.
            output_path: Destination path. Defaults to
                `heatmaps_dir / f"{image_path.stem}_heatmap.png"`.

        Returns:
            Path: The path the heatmap image was saved to.
        """
        image = self.candidate_generator.load_image(image_path)
        threat_map = self.candidate_generator.compute_threat_score_map(image)

        heatmap_uint8 = (threat_map * 255).astype(np.uint8)
        colormap = _COLORMAP_LOOKUP.get(
            VISUALIZATION.heatmap_colormap.upper(), cv2.COLORMAP_JET
        )
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        alpha = VISUALIZATION.heatmap_alpha
        overlay = cv2.addWeighted(colored_heatmap, alpha, image, 1 - alpha, 0)

        output_path = output_path or (
            self.heatmaps_dir / f"{image_path.stem}_heatmap.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)

        logger.info("Heatmap saved to %s", output_path)
        return output_path

    def visualize_top_candidates(
        self, image_path: Path, prediction: ImagePrediction, output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Build a composite panel image showing the source image alongside
        crops of its top-ranked candidate regions, each labeled with its
        threat probability.

        Args:
            image_path: Path to the source image.
            prediction: The ImagePrediction result for this image.
            output_path: Destination path. Defaults to
                `annotated_dir / f"{image_path.stem}_top_candidates.png"`.

        Returns:
            Optional[Path]: The path the panel was saved to, or None if
            there were no candidates to visualize.
        """
        if not prediction.candidates:
            logger.info("No candidates to visualize for %s", image_path.name)
            return None

        source_image = self._load_and_resize(image_path)
        panel_height = 200
        panels: List[np.ndarray] = [
            cv2.resize(source_image, (panel_height, panel_height))
        ]

        sorted_candidates: List[CandidatePrediction] = sorted(
            prediction.candidates, key=lambda c: c.candidate_rank
        )

        for candidate in sorted_candidates:
            crop = source_image[
                candidate.bbox_y : candidate.bbox_y + candidate.bbox_height,
                candidate.bbox_x : candidate.bbox_x + candidate.bbox_width,
            ]
            if crop.size == 0:
                continue

            crop_resized = cv2.resize(crop, (panel_height, panel_height))
            label = f"#{candidate.candidate_rank}: {candidate.threat_probability:.2f}"
            self._draw_label(crop_resized, label, (5, 20))
            panels.append(crop_resized)

        composite = np.hstack(panels)

        output_path = output_path or (
            self.annotated_dir / f"{image_path.stem}_top_candidates.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), composite)

        logger.info("Top-candidate panel saved to %s", output_path)
        return output_path

    def visualize_prediction(
        self, image_path: Path, prediction: ImagePrediction
    ) -> dict:
        """
        Generate all diagnostic visualizations (annotated image, heatmap,
        top-candidate panel) for a single prediction in one call.

        Args:
            image_path: Path to the source image.
            prediction: The ImagePrediction result for this image.

        Returns:
            dict: Mapping of visualization type to output file path (or
            None if that visualization was skipped).
        """
        ensure_directories()
        return {
            "annotated": self.annotate_image(image_path, prediction),
            "heatmap": self.generate_heatmap(image_path),
            "top_candidates": self.visualize_top_candidates(image_path, prediction),
        }


def main() -> None:
    """Entry point for running visualization standalone: predicts on and
    visualizes every image in the generated attacks directory."""
    _configure_logging()
    ensure_directories()

    from predict import PatchPredictor  # local import to avoid unused import if unused

    predictor = PatchPredictor()
    visualizer = Visualizer(candidate_generator=predictor.candidate_generator)

    image_paths = sorted(
    p
    for p in PATHS.generated_attacks_images_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:100]

    for image_path in image_paths:
        prediction = predictor.predict(image_path)
        visualizer.visualize_prediction(image_path, prediction)


if __name__ == "__main__":
    main()
