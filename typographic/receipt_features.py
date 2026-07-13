"""
Receipt Feature Aggregation Module

Converts a variable-length list of per-region feature dicts (from
TypographyAnalyzer + SemanticAnalyzer) into a single, fixed-length
feature vector per receipt, via mean/max/min/std aggregation across
regions. This fixed-length vector is what train.py / evaluate.py /
predict.py actually consume as model input (X).
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["ReceiptFeatureBuilder"]

FeatureDict = dict[str, Any]

# Per-region numeric features to aggregate. Kept as an explicit,
# documented contract rather than inferred from dict keys, so a
# missing/renamed upstream feature fails loudly here instead of
# silently changing the receipt-level schema.
_NUMERIC_FEATURES: tuple[str, ...] = (
    "confidence",
    "width",
    "height",
    "area",
    "aspect_ratio",
    "character_count",
    "word_count",
    "avg_word_length",
    "alphabet_ratio",
    "numeric_ratio",
    "uppercase_ratio",
    "whitespace_ratio",
    "special_character_ratio",
    "character_density",
    "estimated_font_size",
    "financial_keyword_score",
    "attack_keyword_score",
)


class ReceiptFeatureBuilder:
    """
    Aggregates multiple per-region feature dicts into one fixed-length
    feature vector (dict) per receipt.
    """

    def __init__(
        self,
        numeric_features: Optional[tuple[str, ...]] = None
    ) -> None:
        """
        Args:
            numeric_features: Optional override of the feature names
                to aggregate. Defaults to the built-in contract of 17
                typography + semantic features.
        """

        self.numeric_features = (
            numeric_features
            if numeric_features is not None
            else _NUMERIC_FEATURES
        )

    ####################################################

    def aggregate(
        self,
        region_features: list[FeatureDict]
    ) -> Optional[FeatureDict]:
        """
        Aggregates a list of per-region feature dicts into a single
        receipt-level feature dict via mean/max/min/std per feature.

        Args:
            region_features: List of per-region feature dicts, each
                expected to contain every key in self.numeric_features.

        Returns:
            A dict with "num_regions" plus 4 aggregate columns
            (_mean, _max, _min, _std) per numeric feature, or None if
            region_features is empty (a receipt with zero detected
            text regions has no feature vector to build).

        Raises:
            KeyError: If any region dict is missing one of the
                expected numeric feature keys.
        """

        if len(region_features) == 0:

            logger.debug(
                "aggregate() received 0 regions; returning None "
                "(caller should skip this receipt)."
            )

            return None

        receipt: FeatureDict = {
            "num_regions": len(region_features)
        }

        for feature in self.numeric_features:

            try:

                values = [row[feature] for row in region_features]

            except KeyError as exc:

                raise KeyError(
                    f"Region feature dict is missing expected key "
                    f"'{feature}'. Every region must contain all of: "
                    f"{self.numeric_features}. Original error: {exc}"
                ) from exc

            receipt[f"{feature}_mean"] = float(np.mean(values))
            receipt[f"{feature}_max"] = float(np.max(values))
            receipt[f"{feature}_min"] = float(np.min(values))
            receipt[f"{feature}_std"] = float(np.std(values))

        return receipt