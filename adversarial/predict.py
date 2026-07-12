"""
predict.py

Inference module for the Patch Integrity Module.

Given a single receipt image, runs the full detection pipeline: candidate
region proposal, feature extraction (CNN + handcrafted, fused), and
classification, returning a per-candidate threat probability along with
an overall image-level prediction, bounding boxes, and confidence.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import faulthandler
faulthandler.enable()

import logging
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

# --- macOS / Apple Silicon (M-series) segfault workaround -----------------
# OpenCV and PyTorch each spin up their own internal thread pools. When
# calls to each library are interleaved in a tight loop (as predict()
# does below: OpenCV candidate generation -> PyTorch CNN embedding,
# repeated per candidate), the two thread pools can fight over CPU
# affinity and segfault with no Python traceback. Pinning both to
# single-threaded/no-pool avoids this.
cv2.setNumThreads(0)
torch.set_num_threads(1)
# ---------------------------------------------------------------------------

from config import PATHS, NAMING, EVALUATION, LOGGING, ensure_directories
from candidate_generator import CandidateGenerator, CandidateRegion
from feature_extractor import CNNFeatureExtractor
from receipt_features import ReceiptFeatureFuser


logger = logging.getLogger("predict")


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
class CandidatePrediction:
    """
    Prediction result for a single candidate region within an image.

    Attributes:
        candidate_rank: The candidate's rank from candidate generation
            (1 = most visually suspicious).
        bbox_x: Bounding box top-left x coordinate (pixels).
        bbox_y: Bounding box top-left y coordinate (pixels).
        bbox_width: Bounding box width (pixels).
        bbox_height: Bounding box height (pixels).
        threat_probability: Classifier-predicted probability that this
            candidate region is a synthetic patch.
        is_patch: Boolean decision from thresholding `threat_probability`.
    """

    candidate_rank: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    threat_probability: float
    is_patch: bool


@dataclass
class ImagePrediction:
    """
    Aggregated prediction result for a single input image.

    Attributes:
        image_path: Path to the input image.
        prediction: Overall image-level decision ("attacked" or "clean").
        confidence: Confidence associated with the overall decision,
            equal to the maximum candidate threat probability.
        candidates: Per-candidate prediction results.
    """

    image_path: str
    prediction: str
    confidence: float
    candidates: List[CandidatePrediction]


class PatchPredictor:
    """
    End-to-end inference pipeline: candidate proposal -> feature
    extraction -> classification -> threat probability, for a single
    receipt image at a time.

    Attributes:
        model: The trained classifier loaded from disk.
        scaler: The StandardScaler fit during training.
        feature_columns: The exact ordered list of feature column names
            expected by the model.
        candidate_generator: A CandidateGenerator instance.
        cnn_extractor: A CNNFeatureExtractor instance (shared with the
            feature fuser to avoid loading the backbone twice).
        feature_fuser: A ReceiptFeatureFuser instance.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initialize the PatchPredictor by loading the trained model and
        constructing the upstream pipeline components.

        Args:
            model_path: Path to the pickled model payload saved by
                train.py. Defaults to PATHS.best_model_path.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = model_path or PATHS.best_model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run train.py first."
            )

        with model_path.open("rb") as f:
            payload = pickle.load(f)

        self.model_name: str = payload["model_name"]
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_columns: List[str] = payload["feature_columns"]

        logger.debug("Constructing CandidateGenerator")
        self.candidate_generator = CandidateGenerator()

        logger.debug("Constructing CNNFeatureExtractor")
        self.cnn_extractor = CNNFeatureExtractor()

        logger.debug("Constructing ReceiptFeatureFuser")
        self.feature_fuser = ReceiptFeatureFuser(cnn_extractor=self.cnn_extractor)

        logger.info(
            "PatchPredictor initialized with model=%s (%d features)",
            self.model_name,
            len(self.feature_columns),
        )

    def _vectorize(self, feature_row: Dict[str, float]) -> np.ndarray:
        """
        Convert a fused feature dictionary into a model-ready, scaled
        feature vector, aligned to the training feature column order.

        Args:
            feature_row: Dictionary of fused features for one candidate,
                as produced by ReceiptFeatureFuser.fuse_features.

        Returns:
            np.ndarray: A (1, n_features) scaled feature array.
        """
        vector = np.array(
            [[feature_row.get(col, 0.0) for col in self.feature_columns]],
            dtype=np.float32,
        )
        return self.scaler.transform(vector)

    def predict(self, image_path: Path) -> ImagePrediction:
        """
        Run the full inference pipeline on a single receipt image.

        Args:
            image_path: Path to the input receipt image.

        Returns:
            ImagePrediction: The aggregated image-level and per-candidate
            prediction results.

        Raises:
            FileNotFoundError: If the image path does not exist.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.debug("Generating candidates for %s", image_path.name)
        candidates: List[CandidateRegion] = self.candidate_generator.process_image(
            image_path
        )
        logger.debug("Got %d candidates for %s", len(candidates), image_path.name)

        candidate_predictions: List[CandidatePrediction] = []

        for candidate in candidates:
            patch_path = self.candidate_generator.output_patches_dir / candidate.patch_filename
            if not patch_path.exists():
                logger.warning(
                    "Candidate patch file missing on disk: %s; skipping.", patch_path
                )
                continue

            logger.debug("Extracting CNN embedding for %s", patch_path.name)
            cnn_embedding = self.cnn_extractor.extract_single(patch_path)

            logger.debug("Fusing features for %s", patch_path.name)
            fused = self.feature_fuser.fuse_features(patch_path, cnn_embedding)

            feature_vector = self._vectorize(fused)

            if hasattr(self.model, "predict_proba"):
                probability = float(self.model.predict_proba(feature_vector)[0, 1])
            else:
                # Fall back to a decision function mapped through a
                # logistic squashing function if predict_proba is
                # unavailable.
                raw_score = float(self.model.decision_function(feature_vector)[0])
                probability = float(1.0 / (1.0 + np.exp(-raw_score)))

            is_patch = probability >= EVALUATION.decision_threshold

            candidate_predictions.append(
                CandidatePrediction(
                    candidate_rank=candidate.candidate_rank,
                    bbox_x=candidate.bbox_x,
                    bbox_y=candidate.bbox_y,
                    bbox_width=candidate.bbox_width,
                    bbox_height=candidate.bbox_height,
                    threat_probability=round(probability, 4),
                    is_patch=is_patch,
                )
            )

        if candidate_predictions:
            max_confidence = max(c.threat_probability for c in candidate_predictions)
            overall_prediction = (
                "attacked"
                if any(c.is_patch for c in candidate_predictions)
                else "clean"
            )
        else:
            max_confidence = 0.0
            overall_prediction = "clean"

        result = ImagePrediction(
            image_path=str(image_path),
            prediction=overall_prediction,
            confidence=round(max_confidence, 4),
            candidates=candidate_predictions,
        )

        logger.info(
            "Prediction for %s: %s (confidence=%.4f, %d candidates)",
            image_path.name,
            result.prediction,
            result.confidence,
            len(candidate_predictions),
        )
        return result

    def predict_batch(self, image_paths: List[Path]) -> List[ImagePrediction]:
        """
        Run inference on a list of images sequentially.

        Args:
            image_paths: List of paths to input receipt images.

        Returns:
            List[ImagePrediction]: One prediction result per input image
            that was successfully processed.
        """
        results: List[ImagePrediction] = []
        for image_path in image_paths:
            try:
                results.append(self.predict(image_path))
            except FileNotFoundError as exc:
                logger.warning("Skipping %s: %s", image_path, exc)
        return results

    @staticmethod
    def to_dict(prediction: ImagePrediction) -> dict:
        """
        Convert an ImagePrediction (with nested CandidatePrediction
        instances) into a plain, JSON-serializable dictionary.

        Args:
            prediction: The ImagePrediction to convert.

        Returns:
            dict: A nested dictionary representation.
        """
        result = asdict(prediction)
        return result


def main() -> None:
    """
    Entry point for running inference on a single test image.
    """
    _configure_logging()
    ensure_directories()

    predictor = PatchPredictor()

    image_paths = sorted(
    p
    for p in PATHS.generated_attacks_images_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:100]

    predictions = predictor.predict_batch(image_paths)

    for prediction in predictions:
        logger.info(
            "%s -> %s (confidence=%.4f)",
            Path(prediction.image_path).name,
            prediction.prediction,
            prediction.confidence,
        )

        for candidate in prediction.candidates:
            logger.info(
                "  Candidate %d | Probability = %.4f | Patch = %s | "
                "BBox = (%d, %d, %d, %d)",
                candidate.candidate_rank,
                candidate.threat_probability,
                "YES" if candidate.is_patch else "NO",
                candidate.bbox_x,
                candidate.bbox_y,
                candidate.bbox_width,
                candidate.bbox_height,
            )


if __name__ == "__main__":
    main()