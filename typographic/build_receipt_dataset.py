"""
Receipt Dataset Builder

Builds the receipt-level training dataset (one row per receipt) by
running OCR + typography + semantic feature extraction over every
clean and attacked receipt image, aggregating each receipt's regions
into a single fixed-length feature vector via ReceiptFeatureBuilder,
and writing the result to receipt_dataset.csv. This is the dataset
actually consumed by train.py, evaluate.py, and predict.py.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import TRAIN_IMAGES
from config import ATTACK_OUTPUT
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

__all__ = ["ReceiptDatasetBuilder"]


class ReceiptDatasetBuilder:
    """
    Creates one feature vector per receipt, across clean (label=0) and
    attacked (label=1) images.
    """

    def __init__(self, dataset: str = "sroie") -> None:
        """
        Args:
            dataset: Which dataset's attack images to pull, matching
                the subfolder convention attack_generator.py writes
                to (ATTACK_OUTPUT / dataset). Clean images always come
                from TRAIN_IMAGES (SROIE), since that's the only
                dataset with a defined "clean training split" in this
                project; dataset only controls which attack subfolder
                is read.
        """

        logger.info("Initializing modules...")

        self.dataset = dataset

        self.ocr = OCRExtractor()

        self.typography = TypographyAnalyzer()

        self.semantic = SemanticAnalyzer()

        self.receipt_builder = ReceiptFeatureBuilder()

        self.rows: list[dict[str, Any]] = []

    ######################################################

    def process_receipt(self, image_path: Path, label: int) -> bool:
        """
        Extracts, aggregates, and appends one receipt's feature
        vector to self.rows.

        Args:
            image_path: Path to the receipt image.
            label: 0 for clean, 1 for attacked.

        Returns:
            True if a feature vector was appended, False if the
            receipt was skipped (no OCR regions detected, or an error
            occurred during processing).
        """

        logger.info("Processing: %s", image_path.name)

        try:

            regions = self.ocr.extract_text_regions(image_path)

            region_features = []

            for region in regions:

                typography_features = self.typography.extract_features(
                    region
                )

                semantic_features = self.semantic.extract_features(
                    region["text"]
                )

                typography_features.update(semantic_features)

                region_features.append(typography_features)

            receipt_features = self.receipt_builder.aggregate(
                region_features
            )

            if receipt_features is None:

                logger.warning(
                    "No text regions detected in '%s'; skipping.",
                    image_path.name,
                )

                return False

            receipt_features["image_name"] = image_path.name

            receipt_features["label"] = label

            self.rows.append(receipt_features)

            return True

        except Exception as exc:

            logger.error(
                "Skipping '%s' due to error: %s",
                image_path.name,
                exc,
            )

            return False

    ######################################################

    def build_dataset(
        self,
        max_clean: Optional[int] = 10,
        max_attack: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Processes clean and attacked receipt images, builds the
        receipt-level dataset, writes it to CSV, and returns it.

        Args:
            max_clean: Maximum number of clean images to process
                (None = all).
            max_attack: Maximum number of attacked images to process
                (None = all).

        Returns:
            The resulting DataFrame (also written to
            FEATURE_OUTPUT/receipt_dataset.csv).
        """

        clean_images = sorted(TRAIN_IMAGES.glob("*.jpg"))

        if not clean_images:

            logger.warning(
                "No clean images found under %s.", TRAIN_IMAGES
            )

        if max_clean is not None:

            clean_images = clean_images[:max_clean]

        attack_folder = ATTACK_OUTPUT / self.dataset

        attack_images = sorted(attack_folder.glob("*.jpg"))

        if not attack_images:

            logger.warning(
                "No attack images found under %s.", attack_folder
            )

        if max_attack is not None:

            attack_images = attack_images[:max_attack]

        logger.info("Processing CLEAN receipts...")

        skipped_clean = 0

        for i, image in enumerate(clean_images, 1):

            logger.info("[Clean %d/%d]", i, len(clean_images))

            if not self.process_receipt(image, label=0):

                skipped_clean += 1

        logger.info("Processing ATTACK receipts...")

        skipped_attack = 0

        for i, image in enumerate(attack_images, 1):

            logger.info("[Attack %d/%d]", i, len(attack_images))

            if not self.process_receipt(image, label=1):

                skipped_attack += 1

        df = pd.DataFrame(self.rows)

        output_path = FEATURE_OUTPUT / "receipt_dataset.csv"

        df.to_csv(output_path, index=False)

        logger.info("Receipt dataset created successfully.")

        logger.info("Saved to: %s", output_path)

        logger.info("Total receipts: %d", len(df))

        if skipped_clean or skipped_attack:

            logger.warning(
                "Skipped %d clean and %d attack image(s) "
                "(errors or zero detected regions).",
                skipped_clean,
                skipped_attack,
            )

        if not df.empty:

            logger.info(
                "Label distribution:\n%s",
                df["label"].value_counts().to_string(),
            )

        else:

            logger.warning(
                "Resulting dataset is empty — no receipts were "
                "successfully processed."
            )

        return df


##########################################################

if __name__ == "__main__":

    builder = ReceiptDatasetBuilder(dataset="sroie")

    builder.build_dataset(

        max_clean=200,

        max_attack=None

    )