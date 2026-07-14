"""
patch_features.py
------------------
Patch-based forensic feature extraction -- the core methodological
contribution of this module.

WHY THIS FILE EXISTS
---------------------
validate_saa.py showed that GLOBAL whole-image statistics (entropy,
LSB ratio, noise residual, etc. computed once across the entire image)
failed to separate clean from stego images (50% accuracy = random
chance). The likely reason: our test payload (~500 bits) modifies only
~0.065% of pixels, concentrated in a small region (since embed_lsb.py
writes bits sequentially starting from pixel 0). A global average over
~762,000 pixels dilutes a localized anomaly in a few hundred pixels
into invisibility -- like diluting a spoonful of dye in a swimming pool.

THE FIX: LOOK AT PATCHES, NOT THE WHOLE IMAGE
--------------------------------------------------
Instead of computing one entropy value for the whole image, we:
    1. Split the image into small patches (e.g. 64x64 pixels each).
    2. Compute each classical feature (entropy, LSB ratio, noise
       residual) SEPARATELY for every patch.
    3. Aggregate those per-patch values using several statistics:
        - mean:   the old "global average" behavior, kept for
                   comparison
        - max:     captures the single most anomalous patch --
                   exactly where a localized attack would show up
        - std:     how much patches disagree with each other -- a
                   natural image's patches should look fairly similar;
                   one wildly different patch raises std
        - p95:     95th percentile, a robust "near-maximum" that is
                   less sensitive to a single noisy outlier patch than
                   max alone

This turns each classical feature into 4 features instead of 1 --
more information preserved, at the cost of a larger feature vector.

WHY THIS DIRECTLY TESTS OUR HYPOTHESIS
------------------------------------------
If "global statistics dilute localized anomalies" is really the
problem, patch-based MAX and P95 statistics should recover the ability
to separate clean vs. stego, since a localized anomaly would show up
clearly as one unusually different patch, even while the whole-image
average stays misleadingly normal.
"""

import numpy as np
from PIL import Image

# Reuse the exact same underlying computations as your global features,
# just applied per-patch instead of on the whole image at once.
from entropy import compute_entropy_manual
from lsb import extract_lsb_plane, compute_lsb_ratio
from noise import compute_noise_residual


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def split_into_patches(image: np.ndarray, patch_size: int = 64) -> list:
    """
    Split a grayscale image into non-overlapping patch_size x patch_size
    patches.

    Any leftover partial patches at the image's right/bottom edge
    (when the image dimensions aren't an exact multiple of patch_size)
    are simply skipped -- this keeps every patch a consistent size,
    which matters for fair comparison between patches.

    Returns
    -------
    list of np.ndarray
        Each element is one patch_size x patch_size patch.
    """
    height, width = image.shape
    patches = []

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)

    return patches


def aggregate_patch_values(values: list) -> dict:
    """
    Given a list of per-patch values for ONE feature (e.g. entropy),
    compute summary statistics across all patches.

    Returns
    -------
    dict with keys: mean, max, std, p95
    """
    values_array = np.array(values)

    return {
        "mean": float(np.mean(values_array)),
        "max": float(np.max(values_array)),
        "std": float(np.std(values_array)),
        "p95": float(np.percentile(values_array, 95)),
    }


def compute_patch_entropy_values(patches: list) -> list:
    """Compute entropy for every patch, returning a list of values."""
    return [compute_entropy_manual(patch) for patch in patches]


def compute_patch_lsb_ratio_values(patches: list) -> list:
    """Compute LSB ratio for every patch, returning a list of values."""
    values = []
    for patch in patches:
        lsb_plane = extract_lsb_plane(patch)
        values.append(compute_lsb_ratio(lsb_plane))
    return values


def compute_patch_noise_std_values(patches: list, blur_radius: float = 2.0) -> list:
    """
    Compute noise residual standard deviation for every patch.

    Note: computing a Gaussian blur on a small 64x64 patch is much
    cheaper than on a full document image, so this stays fast even
    though we now do it once per patch instead of once per image.
    """
    values = []
    for patch in patches:
        residual = compute_noise_residual(patch, blur_radius=blur_radius)
        values.append(float(np.std(residual)))
    return values


def stego_analyzer_patches(image_path: str, patch_size: int = 64) -> dict:
    """
    Interface function for patch-based features. Splits the image into
    patches, computes entropy / LSB ratio / noise std for each patch,
    then aggregates each into mean/max/std/p95.

    Returns
    -------
    dict with 12 keys total (3 features x 4 aggregation stats):
        patch_entropy_mean, patch_entropy_max, patch_entropy_std, patch_entropy_p95,
        patch_lsb_ratio_mean, patch_lsb_ratio_max, patch_lsb_ratio_std, patch_lsb_ratio_p95,
        patch_noise_std_mean, patch_noise_std_max, patch_noise_std_std, patch_noise_std_p95
    """
    gray = load_image_as_grayscale(image_path)
    patches = split_into_patches(gray, patch_size=patch_size)

    entropy_values = compute_patch_entropy_values(patches)
    lsb_ratio_values = compute_patch_lsb_ratio_values(patches)
    noise_std_values = compute_patch_noise_std_values(patches)

    entropy_stats = aggregate_patch_values(entropy_values)
    lsb_ratio_stats = aggregate_patch_values(lsb_ratio_values)
    noise_std_stats = aggregate_patch_values(noise_std_values)

    result = {}
    for stat_name, value in entropy_stats.items():
        result[f"patch_entropy_{stat_name}"] = round(value, 4)
    for stat_name, value in lsb_ratio_stats.items():
        result[f"patch_lsb_ratio_{stat_name}"] = round(value, 4)
    for stat_name, value in noise_std_stats.items():
        result[f"patch_noise_std_{stat_name}"] = round(value, 4)

    return result


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/stego/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
        "datasets/stego/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_patches(path)
            print(f"\n{path}")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")