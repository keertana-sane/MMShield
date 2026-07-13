"""
MMShield Typographic Pipeline — Demo Script

Ad-hoc/demo script: runs the full extraction pipeline (OCR ->
typography -> semantic) on a single hardcoded test image and writes
the per-region feature vectors to a CSV via FeatureVectorBuilder.
Not part of the automated dataset-building pipeline (see
build_receipt_dataset.py for that); this is a quick manual sanity
check / exploration tool.
"""

import logging

from config import TRAIN_IMAGES
from ocr import OCRExtractor
from typography import TypographyAnalyzer
from semantic import SemanticAnalyzer
from feature_vector import FeatureVectorBuilder

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

TEST_IMAGE = "X00016469612.jpg"


def main() -> None:
    """Runs the extraction pipeline on TEST_IMAGE and prints a preview."""

    image_path = TRAIN_IMAGES / TEST_IMAGE

    if not image_path.is_file():

        raise FileNotFoundError(
            f"Test image not found: {image_path}. Check TRAIN_IMAGES "
            "in config.py and that TEST_IMAGE exists in that folder."
        )

    logger.info("Reading: %s", image_path.name)

    ocr = OCRExtractor()

    typography = TypographyAnalyzer()

    semantic = SemanticAnalyzer()

    builder = FeatureVectorBuilder()

    regions = ocr.extract_text_regions(image_path)

    logger.info("Detected %d text regions", len(regions))

    if not regions:

        logger.warning(
            "No text regions detected in '%s'; resulting CSV will be empty.",
            image_path.name,
        )

    for region in regions:

        features = typography.extract_features(region)

        semantic_features = semantic.extract_features(region["text"])

        features.update(semantic_features)

        builder.add_region(features)

    df = builder.save_csv()

    logger.info("Preview:\n%s", df.head().to_string())


if __name__ == "__main__":

    main()