"""
Typography Feature Extraction Module

Extracts typographic features (geometry, text statistics, character
composition, visual density) from a single OCR region. This is one of
the two core feature-extraction modules the classifier trains on
(alongside semantic.py); its output is combined per-region and later
aggregated per-receipt by receipt_features.py.

Known heuristic limitations (documented here for research
transparency / paper reviewers):
    - "area" is the bounding-box area (width * height) of the
      polygon, not the true polygon area. This is a standard, cheap
      proxy but will overestimate area for non-rectangular or
      rotated text regions.
    - "estimated_font_size" is the bounding-box height of the region,
      not a true font-size estimate. It correlates with font size for
      single-line regions but will overestimate it for multi-line
      regions.
    - "special_character_ratio" only recognizes ASCII punctuation
      (via string.punctuation). Non-ASCII symbols (e.g. currency
      glyphs outside ASCII) are not counted as special characters.
"""

import logging
import string
from typing import Any, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["TypographyAnalyzer"]

Polygon = list[list[float]]
Region = dict[str, Any]
FeatureDict = dict[str, Any]

_PUNCTUATION = set(string.punctuation)

_REQUIRED_REGION_KEYS = ("polygon", "text", "confidence")


class TypographyAnalyzer:
    """
    Extracts typographic features from OCR regions.
    """

    def __init__(self) -> None:
        pass

    # -------------------------------------------------
    # Geometry Features
    # -------------------------------------------------

    def geometry_features(self, polygon: Polygon) -> FeatureDict:
        """
        Computes bounding-box geometry features from a polygon.

        Args:
            polygon: List of [x, y] points describing the OCR
                region's bounding polygon (as returned by
                ocr.py / OCRExtractor).

        Returns:
            Dict with width, height, bounding-box area, center
            coordinates, and aspect ratio.

        Raises:
            ValueError: If polygon is empty or contains malformed
                points (missing x/y coordinates).
        """

        if not polygon:

            raise ValueError(
                "geometry_features received an empty polygon; "
                "cannot compute bounding-box geometry."
            )

        try:

            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]

        except (IndexError, TypeError, ValueError) as exc:

            raise ValueError(
                f"Malformed polygon point(s) in {polygon!r}: {exc}"
            ) from exc

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        # Bounding-box area, not true polygon area (see module docstring).
        area = width * height

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        aspect_ratio = width / height if height != 0 else 0.0

        if width == 0 or height == 0:

            logger.debug(
                "Degenerate polygon (zero width or height): %s",
                polygon,
            )

        return {
            "width": width,
            "height": height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "aspect_ratio": aspect_ratio,
        }

    # -------------------------------------------------
    # Text Statistics
    # -------------------------------------------------

    def text_statistics(self, text: Optional[str]) -> FeatureDict:
        """
        Computes word/character count statistics for OCR text.

        Args:
            text: The recognized text for a region. None is treated
                as an empty string.

        Returns:
            Dict with character_count, word_count, and
            avg_word_length.
        """

        text = text or ""

        words = text.split()

        character_count = len(text)

        word_count = len(words)

        avg_word_length = (
            sum(len(word) for word in words) / word_count
            if word_count > 0 else 0.0
        )

        return {
            "character_count": character_count,
            "word_count": word_count,
            "avg_word_length": round(avg_word_length, 2),
        }

    # -------------------------------------------------
    # Character Composition
    # -------------------------------------------------

    def character_statistics(self, text: Optional[str]) -> FeatureDict:
        """
        Computes character-composition ratios for OCR text, in a
        single pass over the string.

        Args:
            text: The recognized text for a region. None is treated
                as an empty string.

        Returns:
            Dict with alphabet_ratio, numeric_ratio, uppercase_ratio,
            whitespace_ratio, and special_character_ratio, each
            rounded to 3 decimal places.
        """

        text = text or ""

        total = len(text) or 1

        alphabet = numeric = uppercase = whitespace = special = 0

        for char in text:

            if char.isalpha():

                alphabet += 1

                if char.isupper():

                    uppercase += 1

            elif char.isdigit():

                numeric += 1

            elif char.isspace():

                whitespace += 1

            elif char in _PUNCTUATION:

                special += 1

        return {
            "alphabet_ratio": round(alphabet / total, 3),
            "numeric_ratio": round(numeric / total, 3),
            "uppercase_ratio": round(uppercase / total, 3),
            "whitespace_ratio": round(whitespace / total, 3),
            "special_character_ratio": round(special / total, 3),
        }

    # -------------------------------------------------
    # Visual Density
    # -------------------------------------------------

    def visual_density(self, text: Optional[str], area: float) -> float:
        """
        Computes character density (characters per unit bounding-box
        area). Higher density can indicate cramped/small text, which
        is a common typographic-attack signal (attackers often use
        small font sizes to stay visually inconspicuous).

        Args:
            text: The recognized text for a region.
            area: Bounding-box area of the region, as computed by
                geometry_features.

        Returns:
            Character density, rounded to 5 decimal places, or 0.0
            if area is 0 (degenerate region).
        """

        text = text or ""

        if area <= 0:
            return 0.0

        return round(len(text) / area, 5)

    # -------------------------------------------------
    # Final Feature Vector
    # -------------------------------------------------

    def extract_features(self, region: Region) -> FeatureDict:
        """
        Combines geometry, text-statistics, character-composition,
        and density features into a single per-region feature dict.

        Args:
            region: A region dict as produced by
                OCRExtractor.extract_text_regions, with at least
                "polygon", "text", and "confidence" keys.

        Returns:
            A dict containing the original text/confidence plus every
            computed typographic feature. Downstream code
            (receipt_features.py) relies on this exact key set.

        Raises:
            KeyError: If region is missing a required key.
            ValueError: If region["polygon"] is empty or malformed.
        """

        missing_keys = [
            key for key in _REQUIRED_REGION_KEYS if key not in region
        ]

        if missing_keys:

            raise KeyError(
                f"Region dict is missing required key(s) "
                f"{missing_keys}: {region!r}"
            )

        geometry = self.geometry_features(region["polygon"])

        statistics = self.text_statistics(region["text"])

        characters = self.character_statistics(region["text"])

        density = self.visual_density(
            region["text"],
            geometry["area"]
        )

        return {

            "text": region["text"],

            "confidence": region["confidence"],

            **geometry,

            **statistics,

            **characters,

            "character_density": density,

            "estimated_font_size": geometry["height"],
        }