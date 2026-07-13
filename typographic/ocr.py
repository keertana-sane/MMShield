"""
OCR Module

Thin, defensive wrapper around PaddleOCR responsible for extracting
text regions (text, confidence, bounding polygon) from financial
document images. This is the foundation of the MMShield typographic
pipeline: every downstream feature-extraction module (typography.py,
semantic.py) and every dataset-building script (attack_generator.py,
build_receipt_dataset.py, dataset_builder.py) consumes the exact
region dict contract documented below.
"""

import logging
from pathlib import Path
from typing import Any, Union

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["OCRExtractor"]

# Required keys in PaddleOCR's per-page result dict. If a future
# PaddleOCR version renames these, we want a clear error instead of a
# silent KeyError deep in a batch job.
_REQUIRED_PAGE_KEYS = ("rec_texts", "rec_scores", "dt_polys")


class OCRExtractor:
    """
    OCR module for MMShield.

    Responsible only for extracting text regions from financial
    document images. Each extracted region is a dict with the
    following contract, relied upon by every downstream module:

        {
            "text": str,                        # recognized text
            "confidence": float,                # in [0.0, 1.0]
            "polygon": list[list[float]],        # 4 (x, y) corners
        }
    """

    def __init__(self) -> None:

        logger.info("Loading PaddleOCR...")

        try:

            self.ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to initialize PaddleOCR. Check that paddleocr "
                "and its model weights are correctly installed/"
                "downloaded. Original error: "
                f"{exc}"
            ) from exc

        logger.info("OCR Ready!")

    ####################################################

    def extract_text_regions(
        self,
        image_path: Union[str, Path]
    ) -> list[dict[str, Any]]:
        """
        Extracts all text regions from an image.

        Args:
            image_path: Path to the image file to run OCR on.

        Returns:
            A list of region dicts, each with keys "text" (str),
            "confidence" (float in [0, 1]), and "polygon"
            (list of [x, y] corner points). Returns an empty list if
            no text was detected (this is a legitimate outcome for a
            blank or heavily corrupted image, not an error).

        Raises:
            FileNotFoundError: If image_path does not exist or is not
                a file.
            RuntimeError: If OCR inference fails, or if PaddleOCR
                returns a result structure that doesn't match the
                expected contract (e.g. due to a library version
                change).
        """

        image_path = Path(image_path)

        if not image_path.is_file():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:

            results = self.ocr.predict(str(image_path))

        except Exception as exc:

            raise RuntimeError(
                f"PaddleOCR inference failed on '{image_path}': {exc}"
            ) from exc

        if not results:

            logger.warning(
                "PaddleOCR returned no pages for '%s'. Treating as "
                "zero detected regions.",
                image_path.name,
            )

            return []

        page = results[0]

        missing_keys = [
            key for key in _REQUIRED_PAGE_KEYS if key not in page
        ]

        if missing_keys:

            raise RuntimeError(
                f"PaddleOCR result for '{image_path}' is missing "
                f"expected key(s) {missing_keys}. This likely means "
                "the installed paddleocr version has changed its "
                "output format."
            )

        texts = page["rec_texts"]
        scores = page["rec_scores"]
        polygons = page["dt_polys"]

        if not (len(texts) == len(scores) == len(polygons)):

            logger.warning(
                "Mismatched region counts for '%s': texts=%d, "
                "scores=%d, polygons=%d. Truncating to the shortest "
                "list to avoid misaligned regions.",
                image_path.name,
                len(texts),
                len(scores),
                len(polygons),
            )

        regions: list[dict[str, Any]] = []

        for text, score, polygon in zip(texts, scores, polygons):

            polygon_list = (
                polygon.tolist()
                if hasattr(polygon, "tolist")
                else list(polygon)
            )

            regions.append(
                {
                    "text": text,
                    "confidence": float(score),
                    "polygon": polygon_list,
                }
            )

        logger.info(
            "Detected %d text region(s) in '%s'.",
            len(regions),
            image_path.name,
        )

        return regions