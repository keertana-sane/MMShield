"""
evaluate.py

Evaluation module for the Patch Integrity Module.

Loads the trained best model and the OFFICIAL held-out test split
(patch_dataset_combined_test.csv, produced by dataset_builder.py's
build_multi_dataset(split="test")), computes a full suite of
classification metrics (accuracy, precision, recall, specificity, F1,
ROC-AUC), and generates and saves the confusion matrix, ROC curve,
precision-recall curve, and a text classification report.

This is the ONLY place in the patch module where held-out
generalization metrics are computed — train.py's cross-validation
results are for model selection only and must not be treated as a
substitute for this evaluation.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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

from config import PATHS, SPLIT, EVALUATION, LOGGING, ensure_directories


logger = logging.getLogger("evaluate")


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


@dataclass
class EvaluationMetrics:
    """
    Full set of classification metrics computed on the official test
    split.

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision for the positive (patch) class.
        recall: Recall (sensitivity) for the positive (patch) class.
        specificity: True negative rate for the negative (clean) class.
        f1: F1 score for the positive (patch) class.
        roc_auc: Area under the ROC curve.
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


class ModelEvaluator:
    """
    Evaluates a trained classifier on the official held-out test split,
    producing both scalar metrics and saved diagnostic figures.

    Attributes:
        model: The trained classifier loaded from disk.
        scaler: The StandardScaler fit during training (on the full
            training set, in train.py's fit_final_model).
        feature_columns: The ordered list of feature column names expected
            by the model.
    """

    NON_FEATURE_COLUMNS = (
        "image_id",
        "image_id_x",
        "image_id_y",
        "label",
        "candidate_rank",
        "threat_score",
        "dataset",
    )

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initialize the ModelEvaluator by loading the trained model
        payload.

        Args:
            model_path: Path to the pickled model payload saved by
                train.py. Defaults to PATHS.best_model_path.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        model_path = model_path or PATHS.best_model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run train.py first."
            )

        with model_path.open("rb") as f:
            payload = pickle.load(f)

        self.model_name: str = payload["model_name"]
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_columns = payload["feature_columns"]

    def _load_test_split(self, test_csv: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and scale the official test split feature matrix and label
        vector.

        Args:
            test_csv: Path to the test split CSV. Defaults to
                PATHS.get_combined_dataset_csv("test").

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scaled feature matrix, label
            vector).

        Raises:
            FileNotFoundError: If the test split CSV does not exist.
        """
        test_csv = test_csv or PATHS.get_combined_dataset_csv("test")
        if not test_csv.exists():
            raise FileNotFoundError(f"Test split not found: {test_csv}")

        df = pd.read_csv(test_csv)
        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Test split is missing expected columns: {missing}")

        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        y = df[SPLIT.label_column].to_numpy(dtype=np.int32)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hard predictions and positive-class scores for a feature
        matrix.

        Args:
            X: Scaled feature matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted labels, positive
            class scores).
        """
        y_pred = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(X)[:, 1]
        else:
            y_score = self.model.decision_function(X)

        return y_pred, y_score

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
    ) -> EvaluationMetrics:
        """
        Compute the full suite of classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_score: Positive-class scores/probabilities.

        Returns:
            EvaluationMetrics: The computed metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = EvaluationMetrics(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            specificity=float(specificity),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_true, y_score)),
        )

        logger.info(
            "Test metrics -> accuracy=%.4f precision=%.4f recall=%.4f "
            "specificity=%.4f f1=%.4f roc_auc=%.4f",
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.specificity,
            metrics.f1,
            metrics.roc_auc,
        )
        return metrics

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Optional[Path] = None
    ) -> Path:
        """
        Compute and save a confusion matrix heatmap figure.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_path: Destination PNG path. Defaults to
                PATHS.confusion_matrix_png.

        Returns:
            Path: The path the figure was saved to.
        """
        output_path = output_path or PATHS.confusion_matrix_png
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        im = ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Clean", "Patch"])
        ax.set_yticklabels(["Clean", "Patch"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

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
        self, y_true: np.ndarray, y_score: np.ndarray, output_path: Optional[Path] = None
    ) -> Path:
        """
        Compute and save an ROC curve figure.

        Args:
            y_true: Ground-truth labels.
            y_score: Positive-class scores/probabilities.
            output_path: Destination PNG path. Defaults to
                PATHS.roc_curve_png.

        Returns:
            Path: The path the figure was saved to.
        """
        output_path = output_path or PATHS.roc_curve_png
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        logger.info("ROC curve figure saved to %s", output_path)
        return output_path

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_score: np.ndarray, output_path: Optional[Path] = None
    ) -> Path:
        """
        Compute and save a precision-recall curve figure.

        Args:
            y_true: Ground-truth labels.
            y_score: Positive-class scores/probabilities.
            output_path: Destination PNG path. Defaults to
                PATHS.pr_curve_png.

        Returns:
            Path: The path the figure was saved to.
        """
        output_path = output_path or PATHS.pr_curve_png
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        fig, ax = plt.subplots(figsize=EVALUATION.figure_size, dpi=EVALUATION.figure_dpi)
        ax.plot(recall, precision, label="Precision-Recall curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)

        logger.info("Precision-recall curve figure saved to %s", output_path)
        return output_path

    def save_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray, output_path: Optional[Path] = None
    ) -> Path:
        """
        Compute and save a scikit-learn text classification report.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            output_path: Destination text file path. Defaults to
                PATHS.classification_report_txt.

        Returns:
            Path: The path the report was saved to.
        """
        output_path = output_path or PATHS.classification_report_txt
        report = classification_report(
            y_true, y_pred, target_names=["clean", "patch"], zero_division=0
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        logger.info("Classification report saved to %s", output_path)
        return output_path

    def run(self) -> EvaluationMetrics:
        """
        Run the full evaluation pipeline: load the official test split,
        generate predictions, compute metrics, and save all diagnostic
        figures and reports.

        Returns:
            EvaluationMetrics: The computed metrics on the test split.
        """
        ensure_directories()
        X_test, y_test = self._load_test_split()
        y_pred, y_score = self._predict(X_test)

        metrics = self.compute_metrics(y_test, y_pred, y_score)

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_score)
        self.plot_precision_recall_curve(y_test, y_score)
        self.save_classification_report(y_test, y_pred)

        return metrics


def main() -> None:
    """Entry point for running evaluation standalone on the official test split."""
    _configure_logging()
    evaluator = ModelEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()