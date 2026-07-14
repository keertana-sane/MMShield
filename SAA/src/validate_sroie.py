"""
validate_sroie.py
--------------------
Tests whether our classical + SRM + chi-square features can detect
scattered LSB steganography (15% density) specifically on SROIE
images, to check the hypothesis that SROIE (scanner-based) behaves
more like FUNSD (100% detection) than CORD (56.67%, near chance),
since SROIE is scanned rather than camera-photographed.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from validate_scattered import extract_features_for_scattered, FEATURE_ORDER


def collect_sroie_pairs() -> list:
    """Build (image_path, label) pairs from clean + stego_scattered SROIE images."""
    pairs = []

    clean_dir = "datasets/sroie"
    stego_dir = "datasets/stego_scattered/sroie"

    for filename in sorted(os.listdir(clean_dir)):
        if filename.endswith(".png"):
            pairs.append((os.path.join(clean_dir, filename), 0))

    for filename in sorted(os.listdir(stego_dir)):
        if filename.endswith(".png"):
            pairs.append((os.path.join(stego_dir, filename), 1))

    return pairs


def main():
    all_pairs = collect_sroie_pairs()
    print(f"Total SROIE images (clean + stego): {len(all_pairs)}")

    feature_rows = []
    labels = []

    print("Extracting features...")
    for i, (path, label) in enumerate(all_pairs):
        feature_dict = extract_features_for_scattered(path)
        row = [feature_dict[key] for key in FEATURE_ORDER]
        feature_rows.append(row)
        labels.append(label)

    X = np.array(feature_rows, dtype=np.float64)
    y = np.array(labels)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {np.bincount(y)}")

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, X, y, cv=5)

    print("\n--- SROIE only (scattered 15% payload) ---")
    print(f"Per-fold accuracy: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")


if __name__ == "__main__":
    main()