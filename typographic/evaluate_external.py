"""
External Dataset Evaluator — Typographic Module

Evaluates the already-trained typographic Random Forest model (loaded
via predict.py's TypographicPredictor) on datasets it was not
validated against during training/evaluate.py's holdout split: CORD,
FUNSD, and FigStep (SafeBench). The model is trained only on SROIE
(see evaluate.py for the SROIE held-out split), so this measures
generalization to unseen document layouts (CORD, FUNSD) and to a
fully out-of-distribution, attack-only image set (FigStep).

This script performs NO feature extraction of its own and does NOT
retrain or modify the model. Every prediction is produced by calling
TypographicPredictor.predict(), exactly as predict.py does. This
script is only responsible for:
    - locating external dataset images and assigning ground-truth
      labels (0 = clean, 1 = attack)
    - calling the existing predictor once per image
    - aggregating predictions into accuracy/precision/recall/F1/
      ROC-AUC/confusion-matrix, per dataset and in one final summary

Dataset ground truth:
    CORD / FUNSD  : clean images = 0, generated attack images = 1
    FigStep (SafeBench) : every image is an attack -> label = 1

Outputs (all under PROJECT_ROOT / "results"):
    cord_metrics.csv, funsd_metrics.csv, figstep_metrics.csv
    cord_confusion_matrix.png, funsd_confusion_matrix.png,
    figstep_confusion_matrix.png
    final_results.csv  (Dataset,Accuracy,Precision,Recall,F1,ROC_AUC)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import PROJECT_ROOT
from config import ATTACK_OUTPUT
from config import get_dataset_images

from predict import TypographicPredictor


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["ExternalDataset", "ExternalEvaluator"]

RESULTS_DIR: Path = PROJECT_ROOT / "results"

_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
)

_SUMMARY_COLUMNS: tuple[str, ...] = (
    "Dataset",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
)


@dataclass
class ExternalDataset:
    """
    Describes one external evaluation dataset.

    For datasets with both classes present (SROIE, CORD, FUNSD),
    clean_dir supplies label-0 images and attack_dir supplies label-1
    images. For attack-only datasets (FigStep), set all_attack=True
    and put the image folder in attack_dir; clean_dir is left None.
    """

    name: str
    clean_dir: Optional[Path]
    attack_dir: Path
    all_attack: bool = False


# attack_dir = ATTACK_OUTPUT / <dataset> matches the convention already
# established by TypographicAttackGenerator (attack_generator.py),
# which writes generated attacks to ATTACK_OUTPUT / self.dataset.
DATASETS: tuple[ExternalDataset, ...] = (

    ExternalDataset(
        name="CORD",
        clean_dir=get_dataset_images("cord", split="test"),
        attack_dir=ATTACK_OUTPUT / "cord",
    ),

    ExternalDataset(
        name="FUNSD",
        clean_dir=get_dataset_images("funsd", split="test"),
        attack_dir=ATTACK_OUTPUT / "funsd",
    ),

    ExternalDataset(
        name="FigStep",
        clean_dir=None,
        attack_dir=(
            PROJECT_ROOT
            / "datasets"
            / "FigStep"
            / "data"
            / "images"
            / "SafeBench"
        ),
        all_attack=True,
    ),

)


class ExternalEvaluator:
    """
    Evaluates the existing trained typographic model, via
    TypographicPredictor, on external held-out datasets.
    """

    def __init__(self) -> None:

        logger.info("Loading TypographicPredictor...")

        self.predictor = TypographicPredictor()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ####################################################

    @staticmethod
    def _list_images(directory: Optional[Path]) -> list[Path]:
        """
        Lists every image file under directory, recursively.

        Recursion matters here specifically because some external
        datasets (e.g. FigStep's SafeBench) organize images into
        topic subfolders rather than one flat folder — a non-recursive
        scan would silently find zero images for those, the same
        failure mode as the ATTACK_OUTPUT path bug in
        build_receipt_dataset.py.

        Args:
            directory: Directory to scan. None or missing directories
                are treated as containing zero images (logged as a
                warning, not an error).

        Returns:
            Sorted list of image paths.
        """

        if directory is None:

            return []

        if not directory.is_dir():

            logger.warning(
                "Directory does not exist, skipping: %s", directory
            )

            return []

        images = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if not images:

            logger.warning(
                "No images found under %s (including subfolders).",
                directory,
            )

        return images

    ####################################################

    def collect_labeled_images(
        self,
        dataset: ExternalDataset
    ) -> list[tuple[Path, int]]:
        """
        Builds the (image_path, true_label) list for one dataset.

        Args:
            dataset: The ExternalDataset to collect images for.

        Returns:
            List of (image_path, label) pairs, label 0 = clean,
            label 1 = attack.
        """

        pairs: list[tuple[Path, int]] = []

        if dataset.all_attack:

            for image_path in self._list_images(dataset.attack_dir):

                pairs.append((image_path, 1))

            return pairs

        for image_path in self._list_images(dataset.clean_dir):

            pairs.append((image_path, 0))

        for image_path in self._list_images(dataset.attack_dir):

            pairs.append((image_path, 1))

        return pairs

    ####################################################

    def run_predictions(
        self,
        pairs: list[tuple[Path, int]]
    ) -> tuple[list[int], list[int], list[float]]:
        """
        Runs the existing TypographicPredictor on every image and
        collects ground-truth labels, predicted labels, and attack
        probabilities.

        Args:
            pairs: (image_path, true_label) list from
                collect_labeled_images.

        Returns:
            y_true, y_pred, y_score lists (same length, aligned).
        """

        y_true: list[int] = []

        y_pred: list[int] = []

        y_score: list[float] = []

        skipped = 0

        for i, (image_path, label) in enumerate(pairs, 1):

            logger.info(
                "[%d/%d] Predicting: %s", i, len(pairs), image_path.name
            )

            try:

                result = self.predictor.predict(image_path)

            except Exception as exc:

                logger.error(
                    "Skipping '%s' due to error: %s", image_path.name, exc
                )

                skipped += 1

                continue

            predicted_label = 1 if result["prediction"] == "ATTACK" else 0

            y_true.append(label)

            y_pred.append(predicted_label)

            y_score.append(result["probability"])

        if skipped:

            logger.warning(
                "Skipped %d image(s) due to prediction errors.", skipped
            )

        return y_true, y_pred, y_score

    ####################################################

    def compute_metrics(
        self,
        y_true: list[int],
        y_pred: list[int],
        y_score: list[float]
    ) -> dict:
        """
        Computes accuracy, precision, recall, F1, ROC-AUC (if both
        classes are present), and a 2x2 confusion matrix.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_score: Predicted attack probabilities (for ROC-AUC).

        Returns:
            Dict with accuracy, precision, recall, f1, roc_auc
            (float or None), and confusion_matrix (2x2 ndarray,
            labels ordered [0, 1]).
        """

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if len(set(y_true)) > 1:

            metrics["roc_auc"] = roc_auc_score(y_true, y_score)

        else:

            logger.warning(
                "Only one class present in ground truth; skipping ROC-AUC."
            )

            metrics["roc_auc"] = None

        metrics["confusion_matrix"] = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        )

        return metrics

    ####################################################

    def save_metrics_csv(self, dataset_name: str, metrics: dict) -> Path:
        """
        Writes one dataset's metrics to a single-row CSV.

        Args:
            dataset_name: Dataset name (used to build the filename).
            metrics: Dict returned by compute_metrics.

        Returns:
            Path the CSV was written to.
        """

        row = {
            "Dataset": dataset_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC_AUC": metrics["roc_auc"],
        }

        output_path = RESULTS_DIR / f"{dataset_name.lower()}_metrics.csv"

        pd.DataFrame([row], columns=_SUMMARY_COLUMNS).to_csv(
            output_path, index=False
        )

        logger.info("Metrics saved: %s", output_path)

        return output_path

    ####################################################

    def save_confusion_matrix_plot(self, dataset_name: str, cm) -> Path:
        """
        Renders and saves a confusion matrix plot for one dataset.

        Args:
            dataset_name: Dataset name (used to build the filename
                and plot title).
            cm: 2x2 confusion matrix (labels ordered [0, 1]).

        Returns:
            Path the plot was written to.
        """

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Safe", "Attack"],
        )

        disp.plot()

        plt.title(f"{dataset_name} Confusion Matrix")

        plt.tight_layout()

        output_path = (
            RESULTS_DIR / f"{dataset_name.lower()}_confusion_matrix.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        logger.info("Confusion matrix saved: %s", output_path)

        return output_path

    ####################################################

    def evaluate_dataset(self, dataset: ExternalDataset) -> Optional[dict]:
        """
        Runs the full evaluation for one dataset: collects images,
        runs predictions, computes metrics, and saves the CSV +
        confusion-matrix plot.

        Args:
            dataset: The ExternalDataset to evaluate.

        Returns:
            Dict with Dataset/Accuracy/Precision/Recall/F1/ROC_AUC
            (matching _SUMMARY_COLUMNS), or None if zero images were
            found or successfully predicted for this dataset.
        """

        logger.info("=" * 40)

        logger.info("Evaluating dataset: %s", dataset.name)

        logger.info("=" * 40)

        pairs = self.collect_labeled_images(dataset)

        if not pairs:

            logger.warning(
                "No images found for '%s'; skipping this dataset.",
                dataset.name,
            )

            return None

        logger.info("Found %d image(s) for '%s'.", len(pairs), dataset.name)

        y_true, y_pred, y_score = self.run_predictions(pairs)

        if not y_true:

            logger.warning(
                "All images for '%s' failed prediction; skipping.",
                dataset.name,
            )

            return None

        metrics = self.compute_metrics(y_true, y_pred, y_score)

        logger.info("Accuracy  : %.4f", metrics["accuracy"])

        logger.info("Precision : %.4f", metrics["precision"])

        logger.info("Recall    : %.4f", metrics["recall"])

        logger.info("F1 Score  : %.4f", metrics["f1"])

        if metrics["roc_auc"] is not None:

            logger.info("ROC-AUC   : %.4f", metrics["roc_auc"])

        self.save_metrics_csv(dataset.name, metrics)

        self.save_confusion_matrix_plot(
            dataset.name, metrics["confusion_matrix"]
        )

        return {
            "Dataset": dataset.name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC_AUC": metrics["roc_auc"],
        }

    ####################################################

    def run_all(self) -> pd.DataFrame:
        """
        Evaluates every dataset in DATASETS, saves the final summary
        CSV, and prints the summary table.

        Returns:
            Summary DataFrame (also written to
            RESULTS_DIR/final_results.csv).
        """

        summary_rows = []

        for dataset in DATASETS:

            row = self.evaluate_dataset(dataset)

            if row is not None:

                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)

        output_path = RESULTS_DIR / "final_results.csv"

        summary_df.to_csv(output_path, index=False)

        logger.info("=" * 40)

        logger.info("Final Summary")

        logger.info("=" * 40)

        logger.info("\n%s", summary_df.to_string(index=False))

        logger.info("Summary saved: %s", output_path)

        return summary_df


##########################################################

if __name__ == "__main__":

    evaluator = ExternalEvaluator()

    results = evaluator.run_all()

    print()

    print(results.to_string(index=False))