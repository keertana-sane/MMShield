"""
Typographic Attack Predictor

Loads the trained model and runs the full inference pipeline (OCR ->
typography features -> semantic features -> receipt-level
aggregation -> classification) on a single receipt image, reporting
whether it is predicted SAFE or an ATTACK, with a probability score.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from config import FEATURE_OUTPUT

from ocr import OCRExtractor
from typography import TypographyAnalyzer
from semantic import SemanticAnalyzer
from receipt_features import ReceiptFeatureBuilder


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["TypographicPredictor"]

MODEL_PATH: Path = FEATURE_OUTPUT / "typographic_model.pkl"


class TypographicPredictor:
    """
    Runs end-to-end typographic attack prediction on a single receipt
    image.
    """

    def __init__(self) -> None:

        if not MODEL_PATH.is_file():

            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run train.py first."
            )

        logger.info("Loading model...")

        self.model = joblib.load(MODEL_PATH)

        self.ocr = OCRExtractor()

        self.typography = TypographyAnalyzer()

        self.semantic = SemanticAnalyzer()

        self.receipt_builder = ReceiptFeatureBuilder()

    ###################################################

    def extract_receipt_features(self, image_path) -> dict:
        """
        Runs OCR + typography + semantic extraction and aggregates
        the result into a single receipt-level feature dict.

        Args:
            image_path: Path (str or Path) to the receipt image.

        Returns:
            The aggregated receipt-level feature dict.

        Raises:
            FileNotFoundError: If image_path does not exist (raised
                by OCRExtractor).
            ValueError: If no text regions were detected in the image,
                meaning no feature vector can be built.
        """

        regions = self.ocr.extract_text_regions(image_path)

        region_features = []

        for region in regions:

            features = self.typography.extract_features(region)

            semantic_features = self.semantic.extract_features(
                region["text"]
            )

            features.update(semantic_features)

            region_features.append(features)

        receipt = self.receipt_builder.aggregate(region_features)

        if receipt is None:

            raise ValueError(
                f"No text regions detected in '{image_path}'. Cannot "
                "build a feature vector for prediction."
            )

        return receipt

    ###################################################

    def predict(self, image_path) -> dict:
        """
        Predicts whether a receipt image contains a typographic
        prompt-injection attack.

        Args:
            image_path: Path (str or Path) to the receipt image.

        Returns:
            Dict with "prediction" ("SAFE" or "ATTACK") and
            "probability" (float, probability of the ATTACK class).
        """

        receipt = self.extract_receipt_features(image_path)

        X = pd.DataFrame([receipt])

        prediction = self.model.predict(X)[0]

        probability = self.model.predict_proba(X)[0][1]

        label = "ATTACK" if prediction == 1 else "SAFE"

        logger.info("=" * 30)

        logger.info("Prediction : %s", label)

        logger.info("Attack Probability : %.3f", probability)

        logger.info("=" * 30)

        return {
            "prediction": label,
            "probability": float(probability),
        }


##########################################################

if __name__ == "__main__":

    TEST_IMAGE = input("\nEnter receipt image path: ").strip()

    predictor = TypographicPredictor()

    predictor.predict(TEST_IMAGE)