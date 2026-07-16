"""
Receipt Dataset Builder

Builds the receipt-level training dataset (one row per receipt) by
running OCR + typography + semantic feature extraction over every
clean and attacked receipt image, aggregating each receipt's regions
into a single fixed-length feature vector via ReceiptFeatureBuilder,
and writing the result to CSV. This is the dataset actually consumed
by train.py, evaluate.py, and predict.py.

Multi-dataset support:
    Clean images and the split (train/test) they come from are now
    sourced from config.DATASETS[dataset]["clean_<split>"], instead
    of being hardcoded to a single SROIE folder. This lets the same
    builder be pointed at SROIE, CORD, or FUNSD without code changes.
    Use build_multi_dataset() to build a merged training set across
    several datasets, and per-dataset test sets for evaluation.

    config.py is expected to define:

        DATASETS: dict[str, dict] = {
            "sroie": {
                "clean_train": Path(...),
                "clean_test": Path(...),
                "image_glob": "*.jpg",
            },
            "cord": {...},
            "funsd": {...},
        }

    NOTE: attack images are still sourced from
    ATTACK_OUTPUT / dataset regardless of split (unchanged from the
    original implementation). attack_generator.py does not currently
    produce a train/test split of attack images, so evaluation splits
    may see attack images that overlap with what training used. This
    is a known limitation to revisit when attack_generator.py is
    updated — not solved here to avoid redesigning a file outside
    this pass's scope.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import DATASETS
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

__all__ = ["ReceiptDatasetBuilder", "build_multi_dataset"]


class ReceiptDatasetBuilder:
    """
    Creates one feature vector per receipt, across clean (label=0) and
    attacked (label=1) images, for a single dataset + split.
    """

    def __init__(self, dataset: str = "sroie", split: str = "train") -> None:
        """
        Args:
            dataset: Which dataset to build from. Must be a key in
                config.DATASETS (currently: "sroie", "cord", "funsd").
                Both clean images and the attack subfolder
                (ATTACK_OUTPUT / dataset) are sourced from this
                dataset.
            split: "train" or "test". Selects which clean-image split
                to read from config.DATASETS[dataset].
        """

        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available: "
                f"{sorted(DATASETS)}"
            )

        if split not in ("train", "test"):
            raise ValueError(
                f"split must be 'train' or 'test', got '{split}'"
            )

        logger.info("Initializing modules...")

        self.dataset = dataset

        self.split = split

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

            receipt_features["dataset"] = self.dataset

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
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Processes clean and attacked receipt images for this
        dataset + split, builds the receipt-level dataset, optionally
        writes it to CSV, and returns it.

        Args:
            max_clean: Maximum number of clean images to process
                (None = all).
            max_attack: Maximum number of attacked images to process
                (None = all).
            save: If True, writes the result to
                FEATURE_OUTPUT / f"receipt_dataset_{dataset}_{split}.csv".
                Set False when combining via build_multi_dataset, so
                only the final merged file gets written.

        Returns:
            The resulting DataFrame.
        """

        dataset_cfg = DATASETS[self.dataset]

        image_glob = dataset_cfg.get("image_glob", "*.jpg")

        clean_dir = dataset_cfg[f"clean_{self.split}"]

        clean_images = sorted(Path(clean_dir).glob(image_glob))

        if not clean_images:

            logger.warning(
                "No clean images found under %s.", clean_dir
            )

        if max_clean is not None:

            clean_images = clean_images[:max_clean]

        attack_folder = ATTACK_OUTPUT / self.dataset / self.split

        attack_images = sorted(attack_folder.glob("*.jpg"))

        if not attack_images:

            logger.warning(
                "No attack images found under %s.", attack_folder
            )

        if max_attack is not None:

            attack_images = attack_images[:max_attack]

        logger.info(
            "Processing CLEAN receipts (%s / %s)...",
            self.dataset,
            self.split,
        )

        skipped_clean = 0

        for i, image in enumerate(clean_images, 1):

            logger.info("[Clean %d/%d]", i, len(clean_images))

            if not self.process_receipt(image, label=0):

                skipped_clean += 1

        logger.info(
            "Processing ATTACK receipts (%s)...", self.dataset
        )

        skipped_attack = 0

        for i, image in enumerate(attack_images, 1):

            logger.info("[Attack %d/%d]", i, len(attack_images))

            if not self.process_receipt(image, label=1):

                skipped_attack += 1

        df = pd.DataFrame(self.rows)

        if save:

            output_path = (
                FEATURE_OUTPUT
                / f"receipt_dataset_{self.dataset}_{self.split}.csv"
            )

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


def build_multi_dataset(
    datasets: list[str],
    split: str,
    max_clean: Optional[int] = None,
    max_attack: Optional[int] = None,
) -> pd.DataFrame:
    """
    Builds and merges receipt-level datasets across multiple datasets
    for a given split, writing the combined result to
    FEATURE_OUTPUT / f"receipt_dataset_combined_{split}.csv".

    Each dataset is processed by its own ReceiptDatasetBuilder, and
    rows carry a "dataset" column so per-dataset performance can be
    broken out later in evaluate.py / evaluate_external.py.

    Args:
        datasets: Dataset names to include, e.g. ["sroie", "cord",
            "funsd"]. Each must be a key in config.DATASETS.
        split: "train" or "test".
        max_clean: Per-dataset cap on clean images (None = all).
        max_attack: Per-dataset cap on attack images (None = all).

    Returns:
        The combined DataFrame.
    """

    frames = []

    for name in datasets:

        logger.info("=== Building %s / %s ===", name, split)

        builder = ReceiptDatasetBuilder(dataset=name, split=split)

        df = builder.build_dataset(
            max_clean=max_clean,
            max_attack=max_attack,
            save=False,
        )

        frames.append(df)

    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )

    output_path = FEATURE_OUTPUT / f"receipt_dataset_combined_{split}.csv"

    combined.to_csv(output_path, index=False)

    logger.info("Combined dataset saved to: %s", output_path)

    logger.info("Total receipts: %d", len(combined))

    if not combined.empty:

        logger.info(
            "Label distribution:\n%s",
            combined["label"].value_counts().to_string(),
        )

        logger.info(
            "Per-dataset counts:\n%s",
            combined["dataset"].value_counts().to_string(),
        )

    return combined


##########################################################

if __name__ == "__main__":

    build_multi_dataset(
        datasets=["sroie", "cord", "funsd"],
        split="train",
        max_clean=None,
        max_attack=None,
    )

    build_multi_dataset(
        datasets=["sroie", "cord", "funsd"],
        split="test",
        max_clean=None,
        max_attack=None,
    )