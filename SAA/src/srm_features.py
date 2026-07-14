"""
srm_features.py
----------------
SRM-style (Spatial Rich Model) handcrafted features: high-pass noise
residuals + co-occurrence statistics.

WHY WE'RE TRYING THIS NOW
----------------------------
Three previous approaches each failed for a specific, diagnosed reason:
    1. Global statistics (entropy, LSB ratio, noise std) -- diluted by
       averaging across the whole image (~762,000 pixels vs ~500
       modified bits).
    2. Patch-based aggregation (mean/max/std/p95 over 64x64 patches) --
       still diluted, since the embedding is a 1-pixel-tall strip, not
       a square region, so even small patches average mostly-untouched
       rows against the affected one.
    3. Chi-square attack -- broke on both ends of our document
       spectrum: degenerates on FUNSD's near-binary scans (too few
       distinct pixel values per window), and saturates falsely on
       CORD's JPEG-smoothed photographic noise (already looks
       "LSB-like" even when clean).

Each failure taught us something. This approach is different in kind,
not just degree: instead of looking at pixel VALUES or their
histogram, we look at the RELATIONSHIP between neighboring pixels in
a filtered residual domain -- capturing local structure that simple
marginal statistics (histograms, entropy) cannot see.

THE CORE IDEA (SIMPLIFIED FROM THE FULL RICH MODEL)
--------------------------------------------------------
The original Rich Models paper (Fridrich & Kodovsky, 2012) combines
dozens of different high-pass filters and co-occurrence directions
into one huge feature set. For a prototype-stage project, we implement
ONE representative version of this idea:

1. HIGH-PASS RESIDUAL: apply a small filter kernel that highlights
   local pixel-to-pixel differences (similar in spirit to noise.py,
   but using a sharper linear kernel rather than a Gaussian blur
   subtraction).

2. QUANTIZE AND TRUNCATE: round the residual to integers and clip to
   a small range (e.g. -2 to +2). This deliberately focuses on
   FINE-GRAINED correlations -- exactly the scale at which LSB
   flips operate -- rather than large-scale texture.

3. CO-OCCURRENCE MATRIX: for every pair of HORIZONTALLY adjacent
   residual values, count how often each combination (e.g. "-1 next
   to +2") occurs. This is a joint distribution, not just a marginal
   one -- it captures correlations between neighbors that a plain
   histogram of residual values would completely miss.

WHY THIS MIGHT SUCCEED WHERE THE OTHERS DIDN'T
----------------------------------------------------
LSB replacement modifies individual pixels somewhat independently of
their neighbors (a message bit doesn't "know" about the pixel next to
it). This breaks the natural LOCAL CORRELATION that clean images have
(due to smooth features, gradual lighting changes, camera sensor
characteristics). The co-occurrence matrix is specifically sensitive
to this kind of correlation breakdown, in a way that single-pixel
statistics (histograms, entropy, chi-square on value pairs) are not.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


HIGH_PASS_KERNEL = np.array([[-1, 2, -1]])


def compute_high_pass_residual(image: np.ndarray) -> np.ndarray:
    """
    Apply the high-pass kernel to extract a residual highlighting
    local horizontal pixel-to-pixel structure.
    """
    image_float = image.astype(np.float64)
    residual = convolve(image_float, HIGH_PASS_KERNEL, mode="reflect")
    return residual


def quantize_and_truncate(residual: np.ndarray, truncation: int = 2) -> np.ndarray:
    """
    Round the residual to integers and clip to [-truncation, +truncation].
    """
    rounded = np.round(residual)
    truncated = np.clip(rounded, -truncation, truncation)
    return truncated.astype(np.int8)


def compute_cooccurrence_matrix(quantized_residual: np.ndarray, truncation: int = 2) -> np.ndarray:
    """
    Build a co-occurrence matrix of horizontally adjacent residual values.
    """
    size = 2 * truncation + 1
    matrix = np.zeros((size, size), dtype=np.int64)

    shifted = quantized_residual + truncation

    left_values = shifted[:, :-1]
    right_values = shifted[:, 1:]

    for i in range(size):
        for j in range(size):
            matrix[i, j] = np.sum((left_values == i) & (right_values == j))

    return matrix


def summarize_cooccurrence(matrix: np.ndarray) -> dict:
    """
    Reduce the co-occurrence matrix down to summary statistics:
    diagonal ratio, entropy, and energy.
    """
    total = matrix.sum()
    if total == 0:
        return {"srm_diagonal_ratio": 0.0, "srm_entropy": 0.0, "srm_energy": 0.0}

    probabilities = matrix / total

    diagonal_sum = np.trace(matrix)
    diagonal_ratio = float(diagonal_sum / total)

    nonzero_probs = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

    energy = float(np.sum(probabilities ** 2))

    return {
        "srm_diagonal_ratio": round(diagonal_ratio, 4),
        "srm_entropy": round(entropy, 4),
        "srm_energy": round(energy, 4),
    }


def stego_analyzer_srm(image_path: str, truncation: int = 2) -> dict:
    """
    Interface function for this module -- computes the high-pass
    residual, quantizes it, builds a co-occurrence matrix, and
    returns summary statistics.
    """
    gray = load_image_as_grayscale(image_path)
    residual = compute_high_pass_residual(gray)
    quantized = quantize_and_truncate(residual, truncation=truncation)
    cooccurrence = compute_cooccurrence_matrix(quantized, truncation=truncation)

    return summarize_cooccurrence(cooccurrence)


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
            result = stego_analyzer_srm(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")
