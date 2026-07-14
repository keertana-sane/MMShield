"""
variance.py
-----------
Phase 2 of the Steganographic Anomaly Analyzer (SAA):
Local Variance, Histogram Features, and Edge Features.

This file covers three related classical features that all describe
"how much local structure/texture" exists in an image, from slightly
different angles.

------------------------------------------------------------------
1. LOCAL VARIANCE
------------------------------------------------------------------
Global variance (variance of the whole image at once) tells you how
spread out pixel values are overall. But it can hide important detail:
an image could have low global variance while still having small
patches of intense local variation (e.g. a bit of hidden noise in one
corner of an otherwise flat background).

Local variance fixes this by sliding a small window (e.g. 3x3 or 5x5)
across the image and computing the variance WITHIN each window. This
gives a "local variance map" -- bright areas = high local texture/detail,
dark areas = flat/smooth regions.

Why it matters for steganalysis: content-adaptive steganography (WOW,
S-UNIWARD, HILL) deliberately hides data in HIGH local-variance regions,
because changes there are less visually noticeable. So the local variance
map can help reveal *where* an attacker would prefer to hide data, and
whether the actual variance pattern seems disturbed there.

------------------------------------------------------------------
2. HISTOGRAM FEATURES
------------------------------------------------------------------
The pixel intensity histogram (built in entropy.py too) can be
summarized with statistics beyond just entropy:
    - histogram peak count (how many distinct "modes"/bumps exist)
    - skewness (is the distribution lopsided towards dark or light?)
    - kurtosis (how "peaked" vs "flat" is the distribution?)

Natural document images tend to have fairly simple, predictable
histograms (e.g. one big peak near white background, a smaller peak
near black text). Steganography can subtly smooth or perturb this
shape, which skewness/kurtosis can help capture.

------------------------------------------------------------------
3. EDGE FEATURES
------------------------------------------------------------------
Edges are places where pixel intensity changes sharply (text
boundaries, object outlines, table borders). We detect edges using a
Sobel filter (a classic edge-detection technique that highlights
horizontal + vertical intensity gradients).

We summarize edges with:
    - edge_density: what fraction of pixels are "edge pixels"
    - edge_mean_strength: how strong the edges are on average

Why it matters: documents (forms, invoices) are naturally edge-heavy
(text, lines, borders), so edge stats help characterize a document's
"structural fingerprint." Unusual or excessive edge activity in a
document that otherwise looks like plain text can be a red flag.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import generic_filter, sobel
from scipy.stats import skew, kurtosis


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


# ------------------------------------------------------------------
# 1. LOCAL VARIANCE
# ------------------------------------------------------------------

def compute_local_variance_map(image: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Slide a window_size x window_size window across the image and compute
    the variance of pixel values within each window.

    Note: generic_filter with a Python function is somewhat slow for large
    images, but for document-sized images (roughly 700-1000px per side)
    it's fine for prototyping. If this becomes a bottleneck later, this
    can be swapped for a faster vectorized approach.

    Returns
    -------
    np.ndarray
        Same shape as input. Each value = local variance around that pixel.
    """
    image_float = image.astype(np.float64)
    local_variance_map = generic_filter(image_float, np.var, size=window_size)
    return local_variance_map


def summarize_local_variance(local_variance_map: np.ndarray) -> dict:
    """Reduce the local variance map down to summary statistics."""
    return {
        "local_variance_mean": round(float(np.mean(local_variance_map)), 4),
        "local_variance_std": round(float(np.std(local_variance_map)), 4),
        "local_variance_max": round(float(np.max(local_variance_map)), 4),
    }


# ------------------------------------------------------------------
# 2. HISTOGRAM FEATURES
# ------------------------------------------------------------------

def compute_histogram_features(image: np.ndarray) -> dict:
    """
    Compute summary statistics of the pixel intensity histogram beyond
    entropy: skewness and kurtosis.

    Skewness: 0 = symmetric distribution, positive = tail toward bright
    values, negative = tail toward dark values.

    Kurtosis: describes "peakedness." Higher kurtosis = distribution is
    more concentrated with a sharp peak and heavier tails; lower kurtosis
    = flatter, more spread-out distribution. (scipy returns "excess
    kurtosis", where a normal distribution scores 0.)
    """
    flat_pixels = image.flatten().astype(np.float64)

    return {
        "hist_skewness": round(float(skew(flat_pixels)), 4),
        "hist_kurtosis": round(float(kurtosis(flat_pixels)), 4),
    }


# ------------------------------------------------------------------
# 3. EDGE FEATURES
# ------------------------------------------------------------------

def compute_edge_map(image: np.ndarray) -> np.ndarray:
    """
    Compute an edge strength map using the Sobel operator, which
    highlights areas of rapid intensity change (horizontal + vertical).

    Steps:
    1. sobel(image, axis=0) -> highlights horizontal edges (vertical gradient)
    2. sobel(image, axis=1) -> highlights vertical edges (horizontal gradient)
    3. Combine both directions using Euclidean distance (like a vector
       magnitude), giving overall "edge strength" at every pixel.
    """
    image_float = image.astype(np.float64)

    edge_horizontal = sobel(image_float, axis=0)
    edge_vertical = sobel(image_float, axis=1)

    edge_magnitude = np.hypot(edge_horizontal, edge_vertical)
    return edge_magnitude


def summarize_edge_features(edge_map: np.ndarray, edge_threshold: float = 50.0) -> dict:
    """
    Summarize the edge map with:
        edge_density       : fraction of pixels above edge_threshold
                              (i.e. fraction of pixels considered "edges")
        edge_mean_strength : average edge strength across the whole image
    """
    edge_density = float(np.mean(edge_map > edge_threshold))
    edge_mean_strength = float(np.mean(edge_map))

    return {
        "edge_density": round(edge_density, 4),
        "edge_mean_strength": round(edge_mean_strength, 4),
    }


# ------------------------------------------------------------------
# INTERFACE FUNCTION
# ------------------------------------------------------------------

def stego_analyzer_variance(image_path: str, window_size: int = 5, edge_threshold: float = 50.0) -> dict:
    """
    Interface function for this module -- loads an image and returns all
    Phase 2 features combined: local variance, histogram features, and
    edge features.
    """
    gray = load_image_as_grayscale(image_path)

    local_variance_map = compute_local_variance_map(gray, window_size=window_size)
    variance_features = summarize_local_variance(local_variance_map)

    histogram_features = compute_histogram_features(gray)

    edge_map = compute_edge_map(gray)
    edge_features = summarize_edge_features(edge_map, edge_threshold=edge_threshold)

    combined = {}
    combined.update(variance_features)
    combined.update(histogram_features)
    combined.update(edge_features)

    return combined


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_variance(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")