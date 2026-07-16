"""
evaluate_external.py

External benchmark evaluation module for the Patch Integrity Module.

Evaluates the already-trained patch-integrity model on PatchBenchmark,
an external, evaluation-only image set that is NEVER used for training
or model selection (see config.PATCH_BENCHMARK_IMAGES_DIR, deliberately
excluded from config.DATASETS).

This script performs NO retraining, NO attack generation, NO dataset
building, and NO train/validation/test splitting. It reuses the exact
same per-image inference pipeline as predict.py's PatchPredictor
(CandidateGenerator -> CNNFeatureExtractor -> ReceiptFeatureFuser ->
classifier), including PatchPredictor._aggregate_image_prediction() for
image-level aggregation, so PatchBenchmark results are directly
comparable to predict.py's real-world behavior:

    - A candidate region is "is_patch" if its threat_probability >=
      EVALUATION.decision_threshold.
    - An image is "attacked" if ANY candidate is_patch; confidence is
      the MAXIMUM threat_probability across ALL candidates for that
      image, regardless of the final clean/attacked verdict.

Labeling mode:
    If config.PATCH_BENCHMARK_LABELS_CSV exists (columns: image, label
    where label is 0=clean/1=patch), full classification metrics
    (accuracy, precision, recall, specificity, F1, ROC-AUC) and
    diagnostic plots are computed, exactly as in evaluate.py. If the
    labels file is absent, this script automatically switches to
    prediction-only mode and writes a CSV of
    (image, predicted_class, confidence) without raising an error.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

from config import PATHS, IMAGE, EVALUATION, LOGGING, ensure_directories
from predict import PatchPredictor, ImagePrediction


logger = logging.getLogger("evaluate_external")

RESULTS_DIR: Path = PATHS.outputs_dir / "patchbenchmark_results"
_CANDIDATE_PATCHES_DIR: Path = RESULTS_DIR / "candidate_patches"


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
class EvaluationMetrics:
    """
    Full set of classification metrics computed on PatchBenchmark
    (labeled mode only).

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision for the positive (patch) class.
        recall: Recall (sensitivity) for the positive (patch) class.
        specificity: True negative rate for the negative (clean) class.
        f1: F1 score for the positive (patch) class.
        roc_auc: Area under the ROC curve (NaN if only one class present).
    """

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    roc_auc: float

    def as_dict(self) -> dict:
        """Return the metrics as a plain dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


class ExternalBenchmarkEvaluator:
    """
    Evaluates the trained patch-integrity model on the external
    PatchBenchmark image set, using predict.py's PatchPredictor
    end-to-end (candidate generation, feature fusion, classification,
    and image-level aggregation), with no retraining and no dataset
    assembly.

    Attributes:
        predictor: A PatchPredictor instance, reused for every
            PatchBenchmark image, pointed at a dedicated candidate
            patches folder so its output never mixes with the training
            pipeline's per-dataset/split candidate folders.
        keep_candidate_patches: If False (default), the dedicated
            candidate patches folder is deleted after evaluation
            completes, since it can grow large for big benchmarks and
            serves no purpose after metrics/predictions are saved. Set
            True to retain the cropped candidate patches for manual
            debugging/inspection.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        keep_candidate_patches: bool = False,
    ) -> None:
        """
        Initialize the evaluator by constructing a PatchPredictor
        pointed at a dedicated candidate patches folder.

        Args:
            model_path: Path to the pickled model payload saved by
                train.py. Defaults to PATHS.best_model_path (handled by
                PatchPredictor itself).
            keep_candidate_patches: If True, do not delete the
                cropped candidate patch images after evaluation
                finishes. Defaults to False.

        Raises:
            FileNotFoundError: If the model file does not exist
                (raised by PatchPredictor).
        """
        self.keep_candidate_patches = keep_candidate_patches

        self.predictor = PatchPredictor(model_path=model_path)
        # Redirect candidate output to a PatchBenchmark-specific folder
        # so it never collides with the training pipeline's per
        # dataset/split candidate folders.
        self.predictor.candidate_generator.output_patches_dir = _CANDIDATE_PATCHES_DIR
        self.predictor.candidate_generator.candidates_csv_path = (
            RESULTS_DIR / "candidates_patchbenchmark.csv"
        )

        logger.info(
            "ExternalBenchmarkEvaluator initialized with model=%s",
            self.predictor.model_name,
        )

    # ------------------------------------------------------------------
    # Image discovery
    # ------------------------------------------------------------------

    def _discover_images(self, image_dir: Path) -> List[Path]:
        """
        Discover every valid image file in the PatchBenchmark directory.

        Args:
            image_dir: Directory containing benchmark images.

        Returns:
            List[Path]: Sorted list of image file paths.

        Raises:
            FileNotFoundError: If image_dir does not exist.
        """
        if not image_dir.exists():
            raise FileNotFoundError(
                f"PatchBenchmark image directory not found: {image_dir}"
            )

        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE.valid_extensions
        )

        if not images:
            logger.warning("No images found under %s.", image_dir)

        logger.info("Discovered %d PatchBenchmark image(s) in %s", len(images), image_dir)
        return images

    # ------------------------------------------------------------------
    # Labels loading
    # ------------------------------------------------------------------

    def _load_labels(self, labels_csv: Path) -> Optional[dict]:
        """
        Load ground-truth labels for PatchBenchmark, if available.

        Args:
            labels_csv: Path to a CSV with columns "image" and "label"
                (label: 0 = clean, 1 = patch).

        Returns:
            Optional[dict]: Mapping of image filename -> int label, or
            None if the labels file does not exist or is malformed.
        """
        if not labels_csv.exists():
            logger.info(
                "No labels file found at %s; running in prediction-only mode.",
                labels_csv,
            )
            return None

        df = pd.read_csv(labels_csv)
        if "image" not in df.columns or "label" not in df.columns:
            logger.warning(
                "Labels file %s is missing 'image'/'label' columns; "
                "falling back to prediction-only mode.",
                labels_csv,
            )
            return None

        return dict(zip(df["image"], df["label"].astype(int)))

    # ------------------------------------------------------------------
    # Metrics and plots (labeled mode)
    # ------------------------------------------------------------------

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
    ) -> EvaluationMetrics:
        """
        Compute the full suite of classification metrics.

        Args:
            y_true: Ground-truth labels (0=clean, 1=patch).
            y_pred: Predicted labels (0=clean, 1=patch).
            y_score: Positive-class confidence scores (max candidate
                threat_probability per image).

        Returns:
            EvaluationMetrics: The computed metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        roc_auc = (
            float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else float("nan")
        )

        metrics = EvaluationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            specificity=float(specificity),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=roc_auc,
        )

        logger.info(
            "PatchBenchmark metrics -> accuracy=%.4f precision=%.4f recall=%.4f "
            "specificity=%.4f f1=%.4f roc_auc=%s",
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.specificity,
            metrics.f1,
            f"{metrics.roc_auc:.4f}" if not np.isnan(metrics.roc_auc) else "N/A (single class)",
        )
        return metrics

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
    ) -> Path:
        """
        Compute and save a confusion matrix heatmap figure.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_path: Destination PNG path.

        Returns:
            Path: The path the figure was saved to.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Clean", "Patch"])
        ax.set_yticklabels(["Clean", "Patch"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("PatchBenchmark Confusion Matrix")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        logger.info("Confusion matrix figure saved to %s", output_path)
        return output_path

    def plot_roc_curve(
        self, y_true: np.ndarray, y_score: np.ndarray, output_path: Path
    ) -> Path:
        """
        Compute and save an ROC curve figure.

        Args:
            y_true: Ground-truth labels.
            y_score: Positive-class confidence scores.
            output_path: Destination PNG path.

        Returns:
            Path: The path the figure was saved to.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("PatchBenchmark ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        logger.info("ROC curve figure saved to %s", output_path)
        return output_path

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_score: np.ndarray, output_path: Path
    ) -> Path:
        """
        Compute and save a precision-recall curve figure.

        Args:
            y_true: Ground-truth labels.
            y_score: Positive-class confidence scores.
            output_path: Destination PNG path.

        Returns:
            Path: The path the figure was saved to.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        ax.plot(recall, precision, label="Precision-Recall curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("PatchBenchmark Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        logger.info("Precision-recall curve figure saved to %s", output_path)
        return output_path

    def save_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
    ) -> Path:
        """
        Compute and save a scikit-learn text classification report.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_path: Destination text file path.

        Returns:
            Path: The path the report was saved to.
        """
        report = classification_report(
            y_true, y_pred, target_names=["clean", "patch"], zero_division=0
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        logger.info("Classification report saved to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_candidate_patches(self) -> None:
        """
        Delete the dedicated PatchBenchmark candidate patches folder,
        unless self.keep_candidate_patches is True. Safe to call even
        if the folder was never created or already removed.
        """
        if self.keep_candidate_patches:
            logger.info(
                "keep_candidate_patches=True; retaining candidate patches at %s",
                _CANDIDATE_PATCHES_DIR,
            )
            return

        if _CANDIDATE_PATCHES_DIR.exists():
            shutil.rmtree(_CANDIDATE_PATCHES_DIR, ignore_errors=True)
            logger.info(
                "Deleted temporary candidate patches folder: %s",
                _CANDIDATE_PATCHES_DIR,
            )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Run the full PatchBenchmark evaluation: discover images, predict
        each one via PatchPredictor's full pipeline, and either compute
        labeled metrics + plots (if ground truth is available) or save
        a prediction-only CSV. Cleans up the temporary candidate
        patches folder afterward unless keep_candidate_patches was set.

        Returns:
            pd.DataFrame: Per-image results (always includes image_name,
            predicted_label, confidence, num_candidates; includes
            true_label as well when labels are available).
        """
        ensure_directories()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        try:
            images = self._discover_images(PATHS.patch_benchmark_images_dir)
            if not images:
                logger.warning("No PatchBenchmark images to evaluate.")
                return pd.DataFrame()

            labels = self._load_labels(PATHS.patch_benchmark_labels_csv)

            rows: List[dict] = []
            for i, image_path in enumerate(images, 1):
                logger.info("[%d/%d] Predicting: %s", i, len(images), image_path.name)
                try:
                    prediction: ImagePrediction = self.predictor.predict(image_path)
                except Exception as exc:
                    logger.error(
                        "Skipping '%s' due to error: %s", image_path.name, exc
                    )
                    continue

                rows.append(
                    {
                        "image_name": image_path.name,
                        "predicted_label": 1 if prediction.prediction == "attacked" else 0,
                        "confidence": prediction.confidence,
                        "num_candidates": len(prediction.candidates),
                    }
                )

            results_df = pd.DataFrame(rows)

            if labels is None:
                output_path = RESULTS_DIR / "patchbenchmark_predictions.csv"
                results_df.to_csv(output_path, index=False)
                logger.info(
                    "Prediction-only mode: results saved to %s (%d images)",
                    output_path,
                    len(results_df),
                )
                return results_df

            results_df["true_label"] = results_df["image_name"].map(labels)

            unlabeled_mask = results_df["true_label"].isna()
            if unlabeled_mask.any():
                logger.warning(
                    "%d image(s) had no matching entry in the labels file and "
                    "will be excluded from metric computation: %s",
                    int(unlabeled_mask.sum()),
                    ", ".join(results_df.loc[unlabeled_mask, "image_name"].tolist()),
                )

            labeled_df = results_df.loc[~unlabeled_mask].copy()
            labeled_df["true_label"] = labeled_df["true_label"].astype(int)

            if labeled_df.empty:
                logger.warning(
                    "No images could be matched to ground-truth labels; "
                    "falling back to prediction-only output."
                )
                output_path = RESULTS_DIR / "patchbenchmark_predictions.csv"
                results_df.to_csv(output_path, index=False)
                return results_df

            y_true = labeled_df["true_label"].to_numpy(dtype=np.int32)
            y_pred = labeled_df["predicted_label"].to_numpy(dtype=np.int32)
            y_score = labeled_df["confidence"].to_numpy(dtype=np.float32)

            metrics = self.compute_metrics(y_true, y_pred, y_score)

            self.plot_confusion_matrix(
                y_true, y_pred, RESULTS_DIR / "confusion_matrix.png"
            )
            if len(set(y_true)) > 1:
                self.plot_roc_curve(y_true, y_score, RESULTS_DIR / "roc_curve.png")
                self.plot_precision_recall_curve(
                    y_true, y_score, RESULTS_DIR / "pr_curve.png"
                )
            else:
                logger.warning(
                    "Only one class present in PatchBenchmark labels; "
                    "skipping ROC and precision-recall curves."
                )
            self.save_classification_report(
                y_true, y_pred, RESULTS_DIR / "classification_report.txt"
            )

            metrics_csv_path = RESULTS_DIR / "patchbenchmark_metrics.csv"
            pd.DataFrame([metrics.as_dict()]).to_csv(metrics_csv_path, index=False)
            logger.info("Metrics summary saved to %s", metrics_csv_path)

            predictions_csv_path = RESULTS_DIR / "patchbenchmark_predictions.csv"
            results_df.to_csv(predictions_csv_path, index=False)
            logger.info("Per-image predictions saved to %s", predictions_csv_path)

            return results_df
        finally:
            self._cleanup_candidate_patches()


def main() -> None:
    """Entry point for running PatchBenchmark evaluation standalone."""
    import argparse

    _configure_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate patch-integrity classifier on external PatchBenchmark set."
    )
    parser.add_argument(
        "--keep-candidate-patches",
        action="store_true",
        help="Retain cropped candidate patches directory for manual debugging/inspection.",
    )
    args = parser.parse_args()

    evaluator = ExternalBenchmarkEvaluator(
        keep_candidate_patches=args.keep_candidate_patches
    )
    evaluator.run()


if __name__ == "__main__":
    main()