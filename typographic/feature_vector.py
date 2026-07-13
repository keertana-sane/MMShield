"""
Feature Vector Builder

Thin accumulator/writer used by exploratory/demo scripts (main.py) to
collect per-region feature dicts and dump them to a CSV under
FEATURE_OUTPUT. Not part of the main receipt-level dataset pipeline
(see receipt_features.py / build_receipt_dataset.py for that); this is
the region-level, single-image utility.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config import FEATURE_OUTPUT

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["FeatureVectorBuilder"]

FeatureDict = dict[str, Any]


class FeatureVectorBuilder:
    """
    Accumulates per-region feature dicts and writes them to a CSV.
    """

    def __init__(self) -> None:
        self.rows: list[FeatureDict] = []

    ####################################################

    def add_region(self, features: FeatureDict) -> None:
        """
        Adds one region's feature dict to the accumulator.

        Args:
            features: A feature dict, typically produced by
                TypographyAnalyzer.extract_features (optionally
                merged with SemanticAnalyzer.extract_features).

        Raises:
            TypeError: If features is not a dict.
        """

        if not isinstance(features, dict):

            raise TypeError(
                f"add_region expected a dict, got {type(features).__name__}"
            )

        self.rows.append(features)

    ####################################################

    def save_csv(
        self,
        filename: str = "typography_features.csv"
    ) -> pd.DataFrame:
        """
        Writes all accumulated rows to a CSV under FEATURE_OUTPUT.

        Args:
            filename: Output filename (relative to FEATURE_OUTPUT).

        Returns:
            The DataFrame that was written. Empty DataFrame (with no
            columns) if no rows were ever added.
        """

        if not self.rows:

            logger.warning(
                "No rows accumulated; writing an empty CSV to '%s'.",
                filename,
            )

        df = pd.DataFrame(self.rows)

        output_path: Path = FEATURE_OUTPUT / filename

        df.to_csv(output_path, index=False)

        logger.info(
            "Feature vector saved to '%s' (%d rows).",
            output_path,
            len(df),
        )

        return df