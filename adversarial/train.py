"""
train.py

Model training module for the Patch Integrity Module.

Trains three candidate classifiers -- Random Forest, XGBoost, and Support
Vector Machine -- on the official combined training feature dataset
produced by dataset_builder.py (patch_dataset_combined_train.csv),
compares them using stratified k-fold cross-validation (accuracy,
precision, recall, F1, ROC-AUC), selects the best according to
TRAINING.selection_metric, refits the winner on the FULL training set,
and persists it to disk.

This module performs NO internal train/validation/test split. The
project uses official per-dataset train/test boundaries
(config.DATASETS), so there is no separate validation CSV to evaluate
against here. Model selection uses TRAINING.cv_folds cross-validation
on the training data alone. All held-out generalization metrics are
computed exclusively by evaluate.py against the official combined test
CSV (patch_dataset_combined_test.csv).

Cross-validation leakage note:
    Feature scaling is fit INSIDE each cross-validation fold via an
    sklearn Pipeline(StandardScaler -> classifier), not once on the
    full training set beforehand. This ensures CV-estimated metrics
    used for model selection are not inflated by the scaler having
    already seen validation-fold statistics. The final persisted model
    (after the winning model type is selected) is fit with a scaler
    trained on the FULL training set, since at that point there is no
    more held-out data within this file to leak into.
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
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


_CV_SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


@dataclass
class ModelResult:
    """
    Cross-validated evaluation results for a single candidate model,
    computed on the training set only (no held-out test data is used
    for model selection).

    Attributes:
        model_name: Name of the candidate model ("random_forest",
            "xgboost", or "svm").
        accuracy: Mean cross-validated accuracy across TRAINING.cv_folds.
        precision: Mean cross-validated precision.
        recall: Mean cross-validated recall.
        f1: Mean cross-validated F1 score.
        roc_auc: Mean cross-validated ROC-AUC.
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
    Trains and compares Random Forest, XGBoost, and SVM classifiers on
    the Patch Integrity feature dataset via leakage-free cross-validation
    (scaler refit per fold) on the official training split, selecting
    and persisting the best-performing model according to a configurable
    selection metric.

    Attributes:
        NON_FEATURE_COLUMNS: Columns present in the dataset CSVs that
            are identifiers, labels, or provenance metadata rather than
            model input features, and are excluded from the feature
            matrix.
        scaler: A StandardScaler fit on the full training features
            during the FINAL fit only (not used during cross-validated
            model selection), used to normalize inputs for the
            persisted model (in particular SVM).
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
            csv_path: Path to the split CSV.

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

    def load_dataset(
        self,
        train_csv: Optional[Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the official combined training split as RAW (unscaled)
        features. Scaling is deliberately NOT applied here: during
        cross-validation, scaling happens inside each fold via a
        Pipeline (see cross_validate_model), and during the final fit,
        scaling happens in fit_final_model on the full training set.
        Fitting a single scaler on the full set here, before CV, would
        leak validation-fold statistics into every fold's training
        step.

        Args:
            train_csv: Path to the combined training CSV. Defaults to
                PATHS.get_combined_dataset_csv("train").

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X_train_raw, y_train).
        """
        train_csv = train_csv or PATHS.get_combined_dataset_csv("train")

        X_train, y_train = self._load_split(train_csv)

        logger.info(
            "Loaded training set -> %s, features: %d, label distribution: %s",
            X_train.shape,
            len(self.feature_columns),
            dict(zip(*np.unique(y_train, return_counts=True))),
        )
        return X_train, y_train

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, model_name: str, y_train: Optional[np.ndarray] = None):
        """
        Instantiate a candidate model by name using its configured
        hyperparameters.

        Args:
            model_name: One of "random_forest", "xgboost", "svm".
            y_train: Training label vector, used only to compute
                XGBoost's scale_pos_weight from the class imbalance.
                Required when model_name == "xgboost".

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
            model = XGBClassifier(random_state=42, **TRAINING.xgboost_params)
            if y_train is not None:
                positives = np.sum(y_train == 1)
                negatives = np.sum(y_train == 0)
                if positives > 0:
                    model.set_params(scale_pos_weight=negatives / positives)
            return model
        elif model_name == "svm":
          return SVC(
          random_state=42,
          class_weight="balanced",
          **TRAINING.svm_params
        )
        else:
            raise ValueError(f"Unrecognized model name: {model_name}")

    # ------------------------------------------------------------------
    # Cross-validated model selection
    # ------------------------------------------------------------------

    def cross_validate_model(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> ModelResult:
        """
        Cross-validate a single candidate model on the training set using
        TRAINING.cv_folds stratified folds. Scaling is performed INSIDE
        a Pipeline so StandardScaler is refit on each fold's training
        portion only, preventing scaler leakage from validation folds.

        Args:
            model_name: One of "random_forest", "xgboost", "svm".
            X_train: RAW (unscaled) training feature matrix.
            y_train: Training label vector.

        Returns:
            ModelResult: Mean cross-validated metrics for this model.
        """
        logger.info(
            "Cross-validating model: %s (cv_folds=%d)",
            model_name,
            TRAINING.cv_folds,
        )

        base_model = self._build_model(model_name, y_train=y_train)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

        cv = StratifiedKFold(
            n_splits=TRAINING.cv_folds, shuffle=True, random_state=42
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=_CV_SCORING,
            n_jobs=-1,
        )

        result = ModelResult(
            model_name=model_name,
            accuracy=float(np.mean(scores["test_accuracy"])),
            precision=float(np.mean(scores["test_precision"])),
            recall=float(np.mean(scores["test_recall"])),
            f1=float(np.mean(scores["test_f1"])),
            roc_auc=float(np.mean(scores["test_roc_auc"])),
        )

        logger.info(
            "%s (cv mean) -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f",
            model_name,
            result.accuracy,
            result.precision,
            result.recall,
            result.f1,
            result.roc_auc,
        )
        return result

    def compare_models(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> List[ModelResult]:
        """
        Cross-validate every candidate model configured in
        TRAINING.candidate_models on the training set.

        Args:
            X_train: RAW (unscaled) training feature matrix.
            y_train: Training label vector.

        Returns:
            List[ModelResult]: Cross-validated results for every
            candidate model, in the order they were evaluated.
        """
        results: List[ModelResult] = []

        for model_name in TRAINING.candidate_models:
            result = self.cross_validate_model(model_name, X_train, y_train)
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
            "Best model selected: %s (%s=%.4f, cross-validated)",
            best.model_name,
            metric,
            getattr(best, metric),
        )
        return best

    def fit_final_model(
        self, model_name: str, X_train: np.ndarray, y_train: np.ndarray
    ) -> object:
        """
        Refit the selected model on the FULL training set (all folds
        combined). The feature scaler (self.scaler) is fit here, on the
        full training set — this is the only scaler that gets
        persisted and later used by evaluate.py/predict.py against
        genuinely held-out data, so no leakage occurs at this step.

        Args:
            model_name: Name of the selected best model.
            X_train: RAW (unscaled) training feature matrix (full set).
            y_train: Training label vector (full set).

        Returns:
            The fitted model instance.
        """
        logger.info("Refitting %s on the full training set...", model_name)

        X_train_scaled = self.scaler.fit_transform(X_train)

        model = self._build_model(model_name, y_train=y_train)
        model.fit(X_train_scaled, y_train)

        self.trained_models[model_name] = model
        return model

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
        Save a JSON summary of all cross-validated model comparison
        results and the selected best model to
        PATHS.model_metadata_json.

        Args:
            results: All candidate model cross-validation results.
            best: The selected best-performing result.
        """
        PATHS.model_metadata_json.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "selection_metric": TRAINING.selection_metric,
            "cv_folds": TRAINING.cv_folds,
            "best_model": best.model_name,
            "results": [r.as_dict() for r in results],
            "feature_columns": self.feature_columns,
        }

        with PATHS.model_metadata_json.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model comparison metadata written to %s", PATHS.model_metadata_json)

    def run(self) -> ModelResult:
        """
        Run the full training pipeline: load the official training set
        (raw/unscaled), cross-validate and compare all candidate models
        (scaler refit per fold), select the best by cross-validated
        selection_metric, refit it on the full training set (scaler fit
        on the full set here), and save it.

        Returns:
            ModelResult: The cross-validated result of the
            best-performing model.
        """
        ensure_directories()
        X_train, y_train = self.load_dataset()
        results = self.compare_models(X_train, y_train)
        best = self.select_best_model(results)
        self.fit_final_model(best.model_name, X_train, y_train)
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