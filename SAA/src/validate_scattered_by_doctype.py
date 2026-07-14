"""
validate_scattered_by_doctype.py
-----------------------------------
Breaks down the scattered-payload validation result by document type
(FUNSD vs CORD) to understand why the combined result had such high
variance across cross-validation folds (75.83% mean, but 21.47% std,
swinging between 50% and 100% per fold).

This reuses the feature matrix already saved by validate_scattered.py,
so it runs instantly -- no need to re-extract features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from validate_scattered import collect_scattered_pairs


def main():
    X = np.load("outputs/scattered_feature_matrix_X.npy")
    y = np.load("outputs/scattered_feature_matrix_y.npy")

    # Recover which rows correspond to FUNSD vs CORD, using the same
    # order collect_scattered_pairs() produced when the matrix was built.
    all_pairs = collect_scattered_pairs()
    is_funsd = np.array(["funsd" in path for path, _ in all_pairs])
    is_cord = np.array(["cord" in path for path, _ in all_pairs])

    print(f"FUNSD rows: {is_funsd.sum()}, CORD rows: {is_cord.sum()}")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    print("\n--- FUNSD only ---")
    X_funsd, y_funsd = X[is_funsd], y[is_funsd]
    scores_funsd = cross_val_score(pipeline, X_funsd, y_funsd, cv=5)
    print(f"Per-fold accuracy: {scores_funsd}")
    print(f"Mean accuracy: {scores_funsd.mean():.2%}")
    print(f"Standard deviation: {scores_funsd.std():.2%}")

    print("\n--- CORD only ---")
    X_cord, y_cord = X[is_cord], y[is_cord]
    scores_cord = cross_val_score(pipeline, X_cord, y_cord, cv=5)
    print(f"Per-fold accuracy: {scores_cord}")
    print(f"Mean accuracy: {scores_cord.mean():.2%}")
    print(f"Standard deviation: {scores_cord.std():.2%}")


if __name__ == "__main__":
    main()