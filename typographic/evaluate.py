"""
Typographic Model Evaluator

Loads the trained Random Forest model (typographic_model.pkl) and the
receipt-level dataset, evaluates it on a held-out stratified test
split, and produces accuracy/precision/recall/F1, a confusion matrix,
feature importances, and an ROC curve — saving each plot under
FEATURE_OUTPUT.
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

from sklearn.model_selection import train_test_split

from config import FEATURE_OUTPUT


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["TypographicEvaluator"]

MODEL_PATH: Path = FEATURE_OUTPUT / "typographic_model.pkl"

_DROP_COLUMNS: tuple[str, ...] = ("image_name", "label")


class TypographicEvaluator:
    """
    Evaluates the trained typographic attack-detection model.
    """

    def __init__(self) -> None:

        if not MODEL_PATH.is_file():

            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}'. Run train.py first."
            )

        self.model = joblib.load(MODEL_PATH)

        self.feature_names: list[str] = []

    ########################################################

    def load_dataset(self) -> pd.DataFrame:
        """
        Loads the receipt-level feature dataset and records feature
        column names.

        Returns:
            The loaded DataFrame.

        Raises:
            FileNotFoundError: If receipt_dataset.csv is missing.
            ValueError: If the dataset is empty or missing required
                columns.
        """

        dataset_path = FEATURE_OUTPUT / "receipt_dataset.csv"

        if not dataset_path.is_file():

            raise FileNotFoundError(
                f"Dataset not found at '{dataset_path}'. Run "
                "build_receipt_dataset.py first."
            )

        df = pd.read_csv(dataset_path)

        if df.empty:

            raise ValueError(f"Dataset at '{dataset_path}' is empty.")

        missing_columns = [
            col for col in _DROP_COLUMNS if col not in df.columns
        ]

        if missing_columns:

            raise ValueError(
                f"Dataset is missing required column(s) {missing_columns}."
            )

        self.feature_names = list(
            df.drop(columns=list(_DROP_COLUMNS)).columns
        )

        return df

    ########################################################

    def evaluate(self) -> dict[str, float]:
        """
        Runs the full evaluation: metrics, confusion matrix, feature
        importance, and ROC curve, saving each plot to disk.

        Returns:
            Dict of accuracy, precision, recall, f1, and roc_auc.
        """

        df = self.load_dataset()

        X = df.drop(columns=list(_DROP_COLUMNS))

        y = df["label"]

        class_counts = y.value_counts()

        if (class_counts < 2).any():

            raise ValueError(
                "Each class needs at least 2 samples for a stratified "
                f"train/test split. Got class counts: "
                f"{class_counts.to_dict()}"
            )

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        predictions = self.model.predict(X_test)

        probabilities = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }

        logger.info("=" * 40)
        logger.info("Evaluation Metrics")
        logger.info("=" * 40)
        logger.info("Accuracy  : %.4f", metrics["accuracy"])
        logger.info("Precision : %.4f", metrics["precision"])
        logger.info("Recall    : %.4f", metrics["recall"])
        logger.info("F1 Score  : %.4f", metrics["f1"])

        ########################################################
        # Confusion Matrix
        ########################################################

        cm = confusion_matrix(y_test, predictions)

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Safe", "Attack"],
        )

        disp.plot()

        plt.title("Confusion Matrix")

        plt.tight_layout()

        cm_path = FEATURE_OUTPUT / "confusion_matrix.png"

        plt.savefig(cm_path, dpi=300)

        plt.close()

        ########################################################
        # Feature Importance
        ########################################################

        importance = self.model.feature_importances_

        feature_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "Importance": importance,
        }).sort_values(by="Importance", ascending=False)

        logger.info("Top 10 Important Features:\n%s",
                     feature_importance.head(10).to_string(index=False))

        top10 = feature_importance.head(10)

        plt.figure(figsize=(10, 6))

        plt.barh(top10["Feature"][::-1], top10["Importance"][::-1])

        plt.xlabel("Importance Score")

        plt.ylabel("Feature")

        plt.title("Top 10 Random Forest Feature Importances")

        plt.tight_layout()

        fi_path = FEATURE_OUTPUT / "feature_importance.png"

        plt.savefig(fi_path, dpi=300)

        plt.close()

        ########################################################
        # ROC Curve
        ########################################################

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        roc_auc = auc(fpr, tpr)

        metrics["roc_auc"] = roc_auc

        plt.figure(figsize=(6, 6))

        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend(loc="lower right")

        plt.tight_layout()

        roc_path = FEATURE_OUTPUT / "roc_curve.png"

        plt.savefig(roc_path, dpi=300)

        plt.close()

        logger.info("ROC AUC Score: %.4f", roc_auc)

        logger.info(
            "Saved files:\n%s\n%s\n%s",
            cm_path,
            fi_path,
            roc_path,
        )

        return metrics


############################################################


if __name__ == "__main__":

    evaluator = TypographicEvaluator()

    evaluator.evaluate()