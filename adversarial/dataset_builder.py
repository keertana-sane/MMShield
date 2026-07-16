"""
dataset_builder.py

Dataset assembly module for the Patch Integrity Module.

Merges the fused feature table (receipt_features.py output), the
candidate region metadata (candidate_generator.py output), and the
ground-truth patch metadata (attack_generator.py output) into a single
labeled dataset for one dataset + split.

Labeling strategy: a candidate region is labeled positive (1) if it
sufficiently overlaps (by IoU) with the ground-truth patch bounding box
recorded for its source attacked image. Candidates from clean images, or
candidates that do not overlap any ground-truth patch, are labeled
negative (0).

Multi-dataset support:
    build_multi_dataset() merges labeled candidates across several
    datasets (SROIE, CORD, FUNSD) for a single official split
    (train or test) into one combined CSV, mirroring the typographic
    module's build_receipt_dataset.py. There is NO internal
    train/validation/test re-split here: the official per-dataset
    train/test boundaries (config.DATASETS) are preserved end to end.
    Model selection among candidate model types (random_forest,
    xgboost, svm) is expected to happen via TRAINING.cv_folds
    cross-validation on the combined train CSV in train.py, not via a
    held-out validation split carved out here.

    Per-dataset-per-split input files are read from the naming
    conventions established upstream:
        features:  PATHS.features_dir / f"receipt_features_{dataset}_{split}.csv"
        candidates: PATHS.candidates_dir / f"candidates_{dataset}_{split}.csv"
        attack metadata: PATHS.generated_attacks_dir / f"{dataset}_{split}_metadata.csv"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import PATHS, NAMING, SPLIT, LOGGING, ensure_directories


logger = logging.getLogger("dataset_builder")


def _configure_logging() -> None:
    """Configure module-level logging according to LOGGING settings."""
    level = getattr(logging, LOGGING.log_level.upper(), logging.INFO)
    handlers = []

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


class DatasetBuilder:
    """
    Merges features, candidate metadata, and ground-truth attack metadata
    into a single labeled dataset, for one dataset + split.

    Attributes:
        dataset: Dataset name (e.g. "sroie", "cord", "funsd").
        split: Split name ("train" or "test").
        features_csv: Path to the fused feature table CSV
            (receipt_features.py output) for this dataset + split.
        candidates_csv: Path to the candidate region metadata CSV
            (candidate_generator.py output) for this dataset + split.
        attack_metadata_csv: Path to the ground-truth patch metadata CSV
            (attack_generator.py output) for this dataset + split.
    """

    def __init__(
        self,
        dataset: str = "sroie",
        split: str = "train",
        features_csv: Optional[Path] = None,
        candidates_csv: Optional[Path] = None,
        attack_metadata_csv: Optional[Path] = None,
    ) -> None:
        """
        Initialize the DatasetBuilder.

        Args:
            dataset: Dataset name this builder assembles data for.
            split: Split name ("train" or "test") this builder assembles
                data for.
            features_csv: Path to the fused feature CSV. Defaults to
                PATHS.features_dir / f"receipt_features_{dataset}_{split}.csv".
            candidates_csv: Path to the candidate metadata CSV. Defaults
                to PATHS.candidates_dir / f"candidates_{dataset}_{split}.csv".
            attack_metadata_csv: Path to the attack metadata CSV. Defaults
                to PATHS.generated_attacks_dir / f"{dataset}_{split}_metadata.csv".
        """
        self.dataset = dataset
        self.split = split

        self.features_csv = features_csv or (
            PATHS.features_dir / f"receipt_features_{dataset}_{split}.csv"
        )
        self.candidates_csv = candidates_csv or (
            PATHS.candidates_dir / f"candidates_{dataset}_{split}.csv"
        )
        self.attack_metadata_csv = attack_metadata_csv or (
            PATHS.generated_attacks_dir / f"{dataset}_{split}_metadata.csv"
        )

    @staticmethod
    def _compute_iou(box_a: Tuple[float, float, float, float],
                      box_b: Tuple[float, float, float, float]) -> float:
        """
        Compute the Intersection-over-Union (IoU) of two axis-aligned
        bounding boxes.

        Args:
            box_a: (x, y, width, height) of the first box.
            box_b: (x, y, width, height) of the second box.

        Returns:
            float: IoU value in [0, 1].
        """
        ax0, ay0, aw, ah = box_a
        bx0, by0, bw, bh = box_b
        ax1, ay1 = ax0 + aw, ay0 + ah
        bx1, by1 = bx0 + bw, by0 + bh

        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)

        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h

        area_a = max(0.0, aw) * max(0.0, ah)
        area_b = max(0.0, bw) * max(0.0, bh)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _build_ground_truth_lookup(self) -> pd.DataFrame:
        """
        Load the attack metadata CSV and index it by attacked image
        filename for fast ground-truth bounding box lookup.

        Returns:
            pd.DataFrame: The attack metadata table, indexed by
            `attacked_image`.

        Raises:
            FileNotFoundError: If the attack metadata CSV does not exist.
        """
        if not self.attack_metadata_csv.exists():
            raise FileNotFoundError(
                f"Attack metadata CSV not found: {self.attack_metadata_csv}"
            )

        metadata = pd.read_csv(self.attack_metadata_csv)
        metadata = metadata.set_index(NAMING.attacked_image_column, drop=False)
        return metadata

    def assign_labels(
        self, candidates: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign a binary label to each candidate region based on IoU
        overlap with the ground-truth patch bounding box of its source
        image.

        Args:
            candidates: Candidate region metadata table (from
                candidate_generator.py), containing `image_id` and bbox
                columns.
            ground_truth: Ground-truth attack metadata table (from
                attack_generator.py), indexed by attacked image filename.

        Returns:
            pd.DataFrame: The candidates table with an added
            `label` column.
        """
        labels = []
        bbox_cols = list(NAMING.bbox_columns)

        for _, row in candidates.iterrows():
            image_id = row[NAMING.image_id_column]
            candidate_box = (
                row[bbox_cols[0]],
                row[bbox_cols[1]],
                row[bbox_cols[2]],
                row[bbox_cols[3]],
            )

            if image_id not in ground_truth.index:
                # No ground-truth record (e.g. a clean, unattacked image).
                labels.append(SPLIT.negative_label)
                continue

            gt_row = ground_truth.loc[image_id]
            # `.loc` may return a DataFrame if there are duplicate index
            # entries; normalize to a single row.
            if isinstance(gt_row, pd.DataFrame):
                gt_row = gt_row.iloc[0]

            gt_box = (
                float(gt_row[NAMING.x_column]),
                float(gt_row[NAMING.y_column]),
                float(gt_row[NAMING.width_column]),
                float(gt_row[NAMING.height_column]),
            )

            iou = self._compute_iou(candidate_box, gt_box)
            label = (
                SPLIT.positive_label
                if iou >= SPLIT.label_iou_threshold
                else SPLIT.negative_label
            )
            labels.append(label)

        result = candidates.copy()
        result[SPLIT.label_column] = labels
        return result

    def merge(self) -> pd.DataFrame:
        """
        Load and merge the feature table, candidate metadata, and
        ground-truth metadata into a single labeled dataset for this
        dataset + split.

        Returns:
            pd.DataFrame: The merged, labeled dataset, one row per
            candidate patch. Includes a "dataset" column identifying
            which source dataset each row came from, so per-dataset
            performance can be broken out later during evaluation.

        Raises:
            FileNotFoundError: If any required input CSV is missing.
        """
        if not self.features_csv.exists():
            raise FileNotFoundError(f"Features CSV not found: {self.features_csv}")
        if not self.candidates_csv.exists():
            raise FileNotFoundError(f"Candidates CSV not found: {self.candidates_csv}")

        features = pd.read_csv(self.features_csv)
        candidates = pd.read_csv(self.candidates_csv)
        ground_truth = self._build_ground_truth_lookup()

        labeled_candidates = self.assign_labels(candidates, ground_truth)

        # `receipt_features.py` writes an `image_id` column equal to the
        # candidate patch filename stem. `candidate_generator.py` writes a
        # `patch_filename` column with the same stem (plus extension).
        labeled_candidates["_patch_stem"] = labeled_candidates["patch_filename"].apply(
            lambda name: Path(str(name)).stem
        )

        merged = features.merge(
            labeled_candidates[
                ["_patch_stem", SPLIT.label_column, NAMING.image_id_column,
                 NAMING.candidate_rank_column, NAMING.threat_score_column]
            ],
            left_on="image_id",
            right_on="_patch_stem",
            how="inner",
        ).drop(columns=["_patch_stem"])

        merged["dataset"] = self.dataset

        logger.info(
            "Merged dataset (%s/%s): %d rows, %d positive, %d negative",
            self.dataset,
            self.split,
            len(merged),
            int((merged[SPLIT.label_column] == SPLIT.positive_label).sum()),
            int((merged[SPLIT.label_column] == SPLIT.negative_label).sum()),
        )
        return merged


def build_multi_dataset(datasets: List[str], split: str) -> pd.DataFrame:
    """
    Builds and merges labeled datasets across multiple datasets for a
    given official split, writing the combined result to
    PATHS.get_combined_dataset_csv(split).

    No train/validation/test re-split is performed here — the official
    per-dataset train/test boundaries (config.DATASETS) are preserved.
    Each dataset is processed by its own DatasetBuilder, and rows carry
    a "dataset" column so per-dataset performance can be broken out
    later in evaluate.py.

    Args:
        datasets: Dataset names to include, e.g. ["sroie", "cord",
            "funsd"]. Each must have upstream features/candidates/attack
            metadata CSVs already generated for this split.
        split: "train" or "test".

    Returns:
        pd.DataFrame: The combined, labeled dataset.
    """
    ensure_directories()

    frames = []

    for name in datasets:
        logger.info("=== Building %s / %s ===", name, split)

        builder = DatasetBuilder(dataset=name, split=split)

        try:
            df = builder.merge()
        except FileNotFoundError as exc:
            logger.warning(
                "Skipping %s/%s: %s", name, split, exc
            )
            continue

        frames.append(df)

    combined = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )

    output_path = PATHS.get_combined_dataset_csv(split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    logger.info("Combined dataset saved to: %s", output_path)
    logger.info("Total rows: %d", len(combined))

    if not combined.empty:
        logger.info(
            "Label distribution:\n%s",
            combined[SPLIT.label_column].value_counts().to_string(),
        )
        logger.info(
            "Per-dataset counts:\n%s",
            combined["dataset"].value_counts().to_string(),
        )

    return combined


def main() -> None:
    """Entry point for running dataset assembly standalone across all
    three training datasets, for both the train and test splits."""
    _configure_logging()

    build_multi_dataset(datasets=["sroie", "cord", "funsd"], split="train")
    build_multi_dataset(datasets=["sroie", "cord", "funsd"], split="test")


if __name__ == "__main__":
    main()