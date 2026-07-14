"""
validate_saa.py
----------------
Validates the full SAA pipeline (classical features + DiT context +
SRM co-occurrence features) across all 120 images (60 clean, 60
stego), producing a real, defensible accuracy number for your report.

WHAT THIS DOES
---------------
1. Runs extractor.py's stego_analyzer() on every image in your dataset
   (not just 2 examples like before), building a full feature matrix.
2. Trains a simple classifier (Logistic Regression) with cross-
   validation on this feature matrix.
3. Reports accuracy across FOUR different feature subsets, so you can
   see exactly which feature group (if any) carries real signal:
        - classical features only (15)
        - classical + DiT doc-context (47)
        - everything, including SRM co-occurrence (50)
        - SRM co-occurrence features ONLY (3), in isolation

WHY LOGISTIC REGRESSION
--------------------------
A simple, low-capacity model. If it can find good separation, that's
a strong, credible signal your features carry real information -- not
an artifact of an overly complex model finding patterns that don't
generalize.

WHY WE NORMALIZE FEATURES FIRST
------------------------------------
total_frequency_energy is on the order of 10 billion, while most other
features are single or double digits. Without scaling, Logistic
Regression would let that one feature dominate purely due to its huge
numeric scale, not because it's actually more informative.
StandardScaler rescales every feature to mean 0, std 1, so all
features compete on equal footing.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from prepare_dataset import collect_labeled_image_paths
from extractor import extract_all_features_as_dict, FEATURE_ORDER


# The first 15 keys in FEATURE_ORDER are classical, the next 32 are
# DiT doc-context, and the final 3 are SRM co-occurrence features.
CLASSICAL_FEATURE_COUNT = 15
DOC_CONTEXT_FEATURE_COUNT = 32


def build_feature_matrix():
    """
    Run the full extractor on every image in the dataset and build:
        X: a (120, 50) matrix of feature values
        y: a (120,) array of labels (0 = clean, 1 = stego)
    """
    all_pairs = collect_labeled_image_paths()

    feature_rows = []
    labels = []

    print(f"Extracting features for {len(all_pairs)} images (this includes DiT, so it will take a while)...")

    for i, (path, label) in enumerate(all_pairs):
        feature_dict = extract_all_features_as_dict(path)
        row = [feature_dict[key] for key in FEATURE_ORDER]

        feature_rows.append(row)
        labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(all_pairs)} images")

    X = np.array(feature_rows, dtype=np.float64)
    y = np.array(labels)

    return X, y


def evaluate_features(X: np.ndarray, y: np.ndarray, description: str):
    """
    Train Logistic Regression with 5-fold cross-validation and print
    accuracy results, using a StandardScaler to normalize features
    first.
    """
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"\n--- {description} ---")
    print(f"Per-fold accuracy: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")


def main():
    X, y = build_feature_matrix()

    print(f"\nFull feature matrix shape: {X.shape}")
    print(f"Labels: {np.bincount(y)} (index 0 = clean count, index 1 = stego count)")

    # Classical features only (columns 0-14)
    X_classical = X[:, :CLASSICAL_FEATURE_COUNT]
    evaluate_features(X_classical, y, "Classical features ONLY (15 features)")

    # Classical + DiT doc-context (columns 0-46)
    X_classical_and_dit = X[:, :CLASSICAL_FEATURE_COUNT + DOC_CONTEXT_FEATURE_COUNT]
    evaluate_features(X_classical_and_dit, y, "Classical + DiT doc-context features (47 features)")

    # Everything, including SRM co-occurrence features (all 50 columns)
    evaluate_features(X, y, "Classical + DiT + SRM co-occurrence features (50 features)")

    # SRM co-occurrence features ONLY (last 3 columns), in isolation
    X_srm_only = X[:, CLASSICAL_FEATURE_COUNT + DOC_CONTEXT_FEATURE_COUNT:]
    evaluate_features(X_srm_only, y, "SRM co-occurrence features ONLY (3 features)")

    # Save the raw feature matrix and labels for later use (e.g. if you
    # want to try a different classifier later without re-running
    # extraction, which is the slow part).
    np.save("outputs/feature_matrix_X.npy", X)
    np.save("outputs/feature_matrix_y.npy", y)
    print("\nSaved feature matrix to outputs/feature_matrix_X.npy and outputs/feature_matrix_y.npy")


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    main()