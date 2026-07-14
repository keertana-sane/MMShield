"""
test_other_classifiers.py
---------------------------
Tests whether a more powerful classifier (Random Forest, XGBoost) can
extract signal from our existing feature matrix that Logistic
Regression could not.

WHY THIS TEST MATTERS
------------------------
Our validate_saa.py results showed EXACTLY 50.00% accuracy with 0.00%
standard deviation across every fold, for classical features, SRM
features, and combinations thereof. This specific pattern (dead-even
accuracy, zero variation) is the signature of NO SEPARABLE SIGNAL in
the data -- not evidence that Logistic Regression specifically is too
weak a classifier.

This script tests that claim directly: if Random Forest or XGBoost
also land at ~50%, this confirms the problem is the FEATURES
themselves (they don't differ meaningfully between clean and stego),
not the choice of classifier. If either does meaningfully better, that
would be surprising and worth investigating further.

This reuses the feature matrix already saved by validate_saa.py, so it
runs fast -- no need to re-extract features (including the slow DiT
step) again.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def evaluate_classifier(classifier, X: np.ndarray, y: np.ndarray, description: str):
    """
    Train the given classifier with 5-fold cross-validation, using a
    StandardScaler first (harmless for tree-based models, still
    important in case we test other classifiers later).
    """
    pipeline = make_pipeline(StandardScaler(), classifier)
    scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"\n--- {description} ---")
    print(f"Per-fold accuracy: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")


def main():
    X = np.load("outputs/feature_matrix_X.npy")
    y = np.load("outputs/feature_matrix_y.npy")

    print(f"Loaded feature matrix: {X.shape}")

    evaluate_classifier(
        RandomForestClassifier(n_estimators=200, random_state=42),
        X, y,
        "Random Forest (200 trees) on ALL 50 features",
    )

    if XGBOOST_AVAILABLE:
        evaluate_classifier(
            XGBClassifier(eval_metric="logloss", random_state=42),
            X, y,
            "XGBoost on ALL 50 features",
        )
    else:
        print("\nxgboost not installed -- skipping XGBoost test.")
        print("Install with: python3 -m pip install xgboost --break-system-packages")


if __name__ == "__main__":
    main()
