"""
validate_scattered.py
------------------------
Validates whether our EXISTING classical features (entropy, noise,
LSB, frequency, variance, SRM co-occurrence, chi-square) can detect
steganography now that the payload is embedded at a realistic density
(15% of pixels, scattered pseudo-randomly), rather than the extremely
sparse sequential payload used in our first six experiments.

WHY WE'RE RE-TESTING OLD FEATURES ON NEW DATA
--------------------------------------------------
All six earlier experiments failed because the payload was too sparse
(roughly 0.001 bits per pixel) for ANY statistical method to detect,
regardless of sophistication. Now that embed_lsb_scattered.py embeds
at 15% pixel density (comparable to the 0.1-0.4 bpp range used in
published steganalysis benchmarks), we expect our EXISTING classical
features to finally show real separation -- without needing a new
detection method at all. This directly tests that hypothesis.

Note: we deliberately skip DiT here. We already have two independent,
conclusive experiments showing DiT's resize destroys pixel-level
information regardless of payload density (the resize interpolation
problem is unrelated to payload sparsity) -- so there's no reason to
re-test it.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from entropy import stego_analyzer_entropy
from noise import stego_analyzer_noise
from lsb import stego_analyzer_lsb
from frequency import stego_analyzer_frequency
from variance import stego_analyzer_variance
from srm_features import stego_analyzer_srm
from chi_square import stego_analyzer_chisquare


FEATURE_ORDER = [
    "entropy_manual",
    "noise_mean",
    "noise_std",
    "noise_abs_mean",
    "lsb_ratio",
    "lsb_entropy",
    "total_frequency_energy",
    "high_freq_ratio",
    "local_variance_mean",
    "local_variance_std",
    "local_variance_max",
    "hist_skewness",
    "hist_kurtosis",
    "edge_density",
    "edge_mean_strength",
    "srm_diagonal_ratio",
    "srm_entropy",
    "srm_energy",
    "chisq_max_pvalue",
    "chisq_mean_pvalue",
    "chisq_fraction_high",
]


def extract_features_for_scattered(image_path: str) -> dict:
    """Run all classical + SRM + chi-square feature extractors on one image."""
    combined = {}
    combined.update(stego_analyzer_entropy(image_path))
    combined.update(stego_analyzer_noise(image_path))
    combined.update(stego_analyzer_lsb(image_path))
    combined.update(stego_analyzer_frequency(image_path))
    combined.update(stego_analyzer_variance(image_path))
    combined.update(stego_analyzer_srm(image_path))
    combined.update(stego_analyzer_chisquare(image_path))
    return combined


def collect_scattered_pairs() -> list:
    """Build (image_path, label) pairs from clean + stego_scattered images."""
    pairs = []

    clean_dirs = {
        "datasets/funsd": "datasets/stego_scattered/funsd",
        "datasets/cord": "datasets/stego_scattered/cord",
    }

    for clean_dir, stego_dir in clean_dirs.items():
        for filename in sorted(os.listdir(clean_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(clean_dir, filename), 0))

        for filename in sorted(os.listdir(stego_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(stego_dir, filename), 1))

    return pairs


def build_feature_matrix():
    all_pairs = collect_scattered_pairs()

    feature_rows = []
    labels = []

    print(f"Extracting features for {len(all_pairs)} images (no DiT this time, should be fast)...")

    for i, (path, label) in enumerate(all_pairs):
        feature_dict = extract_features_for_scattered(path)
        row = [feature_dict[key] for key in FEATURE_ORDER]

        feature_rows.append(row)
        labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(all_pairs)} images")

    X = np.array(feature_rows, dtype=np.float64)
    y = np.array(labels)

    return X, y


def evaluate_features(X, y, description):
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"\n--- {description} ---")
    print(f"Per-fold accuracy: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")


def main():
    X, y = build_feature_matrix()

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels: {np.bincount(y)} (index 0 = clean count, index 1 = stego count)")

    evaluate_features(X, y, "All classical + SRM + chi-square features (21 features) on SCATTERED 15% payload")

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/scattered_feature_matrix_X.npy", X)
    np.save("outputs/scattered_feature_matrix_y.npy", y)
    print("\nSaved to outputs/scattered_feature_matrix_X.npy and outputs/scattered_feature_matrix_y.npy")


if __name__ == "__main__":
    main()