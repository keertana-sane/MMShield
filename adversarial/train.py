"""
train.py

Model training module for the Patch Integrity Module.

Trains three candidate classifiers -- Random Forest, XGBoost, and Support
Vector Machine -- on the fused feature dataset produced by
dataset_builder.py, compares them on the validation split using accuracy,
precision, recall, F1, and ROC-AUC, and automatically persists the
best-performing model to disk.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config import PATHS, SPLIT, TRAINING, LOGGING, ensure_directories


logger = logging.getLogger("train")


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
class ModelResult:
    """
    Evaluation results for a single trained candidate model.

    Attributes:
        model_name: Name of the candidate model ("random_forest",
            "xgboost", or "svm").
        accuracy: Accuracy on the validation split.
        precision: Precision on the validation split.
        recall: Recall on the validation split.
        f1: F1 score on the validation split.
        roc_auc: ROC-AUC on the validation split.
    """

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self) -> Dict[str, float]:
        """Return the result as a plain dictionary."""
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
        }


class ModelTrainer:
    """
    Trains and compares Random Forest, XGBoost, and SVM classifiers on the
    Patch Integrity feature dataset, selecting and persisting the
    best-performing model according to a configurable selection metric.

    Attributes:
        non_feature_columns: Columns present in the dataset CSVs that are
            identifiers or labels rather than model input features, and
            are excluded from the feature matrix.
        scaler: A StandardScaler fit on the training features, used to
            normalize inputs for all candidate models (in particular SVM).
    """

    NON_FEATURE_COLUMNS = (
    "image_id",
    "image_id_x",
    "image_id_y",
    "label",
    "candidate_rank",
    "threat_score",
    )

    def __init__(self) -> None:
        """Initialize the ModelTrainer."""
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.trained_models: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_split(self, csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a dataset split CSV and separate it into a feature matrix and
        a label vector.

        Args:
            csv_path: Path to the split CSV (train/validation/test).

        Returns:
            Tuple[np.ndarray, np.ndarray]: (feature matrix, label vector).

        Raises:
            FileNotFoundError: If the split CSV does not exist.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if not self.feature_columns:
            self.feature_columns = [
                col for col in df.columns if col not in self.NON_FEATURE_COLUMNS
            ]

        missing = set(self.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Split {csv_path} is missing expected columns: {missing}")

        X = df[self.feature_columns].to_numpy(dtype=np.float32)
        y = df[SPLIT.label_column].to_numpy(dtype=np.int32)
        return X, y

    def load_datasets(
        self,
        train_csv: Optional[Path] = None,
        validation_csv: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the train and validation splits, fitting the feature scaler
        on the training data only.

        Args:
            train_csv: Path to the training split CSV. Defaults to
                PATHS.train_csv.
            validation_csv: Path to the validation split CSV. Defaults to
                PATHS.validation_csv.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train_scaled, y_train, X_val_scaled, y_val).
        """
        train_csv = train_csv or PATHS.train_csv
        validation_csv = validation_csv or PATHS.validation_csv

        X_train, y_train = self._load_split(train_csv)
        X_val, y_val = self._load_split(validation_csv)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        logger.info(
            "Loaded datasets -> train: %s, validation: %s, features: %d",
            X_train_scaled.shape,
            X_val_scaled.shape,
            len(self.feature_columns),
        )
        return X_train_scaled, y_train, X_val_scaled, y_val

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, model_name: str):
        """
        Instantiate a candidate model by name using its configured
        hyperparameters.

        Args:
            model_name: One of "random_forest", "xgboost", "svm".

        Returns:
            A scikit-learn-compatible estimator instance.

        Raises:
            ValueError: If `model_name` is not recognized.
        """
        if model_name == "random_forest":
          return RandomForestClassifier(
          random_state=42,
          class_weight="balanced",
          **TRAINING.random_forest_params
        )
        elif model_name == "xgboost":
            return XGBClassifier(random_state=42, **TRAINING.xgboost_params)
        elif model_name == "svm":
          return SVC(
          random_state=42,
          class_weight="balanced",
          **TRAINING.svm_params
        )
        else:
            raise ValueError(f"Unrecognized model name: {model_name}")

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train a single candidate model.

        Args:
            model_name: One of "random_forest", "xgboost", "svm".
            X_train: Scaled training feature matrix.
            y_train: Training label vector.

        Returns:
            The fitted model instance.
        """
        logger.info("Training model: %s", model_name)

        model = self._build_model(model_name)

        if model_name == "xgboost":
         positives = np.sum(y_train == 1)
         negatives = np.sum(y_train == 0)

         model.set_params(
         scale_pos_weight=negatives / positives
         )

        model.fit(X_train, y_train)

        return model

    def evaluate_model(
        self, model_name: str, model, X_val: np.ndarray, y_val: np.ndarray
    ) -> ModelResult:
        """
        Evaluate a trained model on the validation split.

        Args:
            model_name: Name of the model being evaluated.
            model: The fitted model instance.
            X_val: Scaled validation feature matrix.
            y_val: Validation label vector.

        Returns:
            ModelResult: The computed evaluation metrics.
        """
        y_pred = model.predict(X_val)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_val)[:, 1]
        else:
            y_score = model.decision_function(X_val)

        result = ModelResult(
            model_name=model_name,
            accuracy=float(accuracy_score(y_val, y_pred)),
            precision=float(precision_score(y_val, y_pred, zero_division=0)),
            recall=float(recall_score(y_val, y_pred, zero_division=0)),
            f1=float(f1_score(y_val, y_pred, zero_division=0)),
            roc_auc=float(roc_auc_score(y_val, y_score)),
        )

        logger.info(
            "%s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            model_name,
            result.accuracy,
            result.precision,
            result.recall,
            result.f1,
            result.roc_auc,
        )
        return result

    def compare_models(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> List[ModelResult]:
        """
        Train and evaluate every candidate model configured in
        TRAINING.candidate_models.

        Args:
            X_train: Scaled training feature matrix.
            y_train: Training label vector.
            X_val: Scaled validation feature matrix.
            y_val: Validation label vector.

        Returns:
            List[ModelResult]: Evaluation results for every candidate
            model, in the order they were trained.
        """
        results: List[ModelResult] = []

        for model_name in TRAINING.candidate_models:
            model = self.train_model(model_name, X_train, y_train)
            self.trained_models[model_name] = model
            result = self.evaluate_model(model_name, model, X_val, y_val)
            results.append(result)

        return results

    def select_best_model(self, results: List[ModelResult]) -> ModelResult:
        """
        Select the best-performing model according to
        TRAINING.selection_metric.

        Args:
            results: List of ModelResult instances to compare.

        Returns:
            ModelResult: The best-performing result.

        Raises:
            ValueError: If `results` is empty.
        """
        if not results:
            raise ValueError("No model results to select from.")

        metric = TRAINING.selection_metric
        best = max(results, key=lambda r: getattr(r, metric))
        logger.info(
            "Best model selected: %s (%s=%.4f)",
            best.model_name,
            metric,
            getattr(best, metric),
        )
        return best

    def save_model(self, model_name: str, output_path: Optional[Path] = None) -> Path:
        """
        Persist a trained model (along with its feature scaler and column
        ordering) to disk via pickle.

        Args:
            model_name: Name of the trained model to save (must already
                exist in `self.trained_models`).
            output_path: Destination path. Defaults to
                PATHS.best_model_path.

        Returns:
            Path: The path the model was saved to.

        Raises:
            KeyError: If `model_name` has not been trained yet.
        """
        if model_name not in self.trained_models:
            raise KeyError(f"Model '{model_name}' has not been trained.")

        output_path = output_path or PATHS.best_model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_name": model_name,
            "model": self.trained_models[model_name],
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
        }

        with output_path.open("wb") as f:
            pickle.dump(payload, f)

        logger.info("Saved best model (%s) to %s", model_name, output_path)
        return output_path

    def save_metadata(self, results: List[ModelResult], best: ModelResult) -> None:
        """
        Save a JSON summary of all model comparison results and the
        selected best model to PATHS.model_metadata_json.

        Args:
            results: All candidate model evaluation results.
            best: The selected best-performing result.
        """
        PATHS.model_metadata_json.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "selection_metric": TRAINING.selection_metric,
            "best_model": best.model_name,
            "results": [r.as_dict() for r in results],
            "feature_columns": self.feature_columns,
        }

        with PATHS.model_metadata_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model comparison metadata written to %s", PATHS.model_metadata_json)

    def run(self) -> ModelResult:
        """
        Run the full training pipeline: load data, train and compare all
        candidate models, select and save the best one.

        Returns:
            ModelResult: The evaluation result of the best-performing
            model.
        """
        ensure_directories()
        X_train, y_train, X_val, y_val = self.load_datasets()
        results = self.compare_models(X_train, y_train, X_val, y_val)
        best = self.select_best_model(results)
        self.save_model(best.model_name)
        self.save_metadata(results, best)
        return best


def main() -> None:
    """Entry point for running model training and comparison standalone."""
    _configure_logging()
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
