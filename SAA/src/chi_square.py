"""
chi_square.py
-------------
Implements the classic Chi-Square Attack (Westfeld & Pfitzmann, 1999)
for detecting sequential LSB steganography.

WHY WE NEEDED THIS
--------------------
Global whole-image statistics (entropy, noise, LSB ratio) failed to
separate clean from stego images (50% accuracy). Patch-based
aggregation (mean/max/std/p95) also failed, because our embedding
modifies a thin 1-pixel-tall strip at the start of the image, and even
64x64 patches average that strip against 63 unaffected rows, diluting
the signal below what entropy/noise can detect.

The Chi-Square Attack is different: it's not a general anomaly
detector, it's a statistical test built SPECIFICALLY for how LSB
replacement changes pixel value pair frequencies -- and it's classic
for finding WHERE sequential embedding starts/stops, which is exactly
what our embed_lsb.py does (writes bits starting from pixel 0).

THE CORE IDEA
--------------
Group all 256 possible pixel values into 128 "Pairs of Values" (POVs):
    (0, 1), (2, 3), (4, 5), ..., (254, 255)

These pairs are chosen because LSB replacement can only ever convert
a value within a pair into its partner (e.g. 100 <-> 101), never
across pairs (e.g. 100 can never become 103 via LSB replacement alone).

For a NATURAL, un-tampered image, the two values in a pair usually
have quite different frequencies (e.g. value 100 appears 850 times,
value 101 appears 620 times -- no particular reason they'd be equal).

After LSB EMBEDDING, roughly half the pixels in a pair get their LSB
overwritten by message bits (which look statistically like a coin
flip). This pushes both values in the pair toward their average
frequency: (850 + 620) / 2 = 735 each.

The chi-square statistic formally measures "how close are the observed
frequencies to what we'd expect if LSB embedding had happened here?"
A LOW chi-square statistic (equivalently, a HIGH p-value) means the
data matches the "smoothed toward average" pattern LSB embedding
produces -- i.e. strong evidence of embedding.

SLIDING WINDOW: FINDING *WHERE* THE EMBEDDING IS
----------------------------------------------------
Since embed_lsb.py writes the message sequentially starting at pixel
0, the embedding only affects the first ~500 pixels out of ~762,000.
If we compute the chi-square statistic across the WHOLE image at once,
this tiny embedded region gets diluted by all the untouched pixels
that follow it -- the same dilution problem as before.

Instead, we slide a small window (e.g. 2000 pixels) across the
flattened image, computing the chi-square p-value for each window
separately. A window that overlaps the embedded region should show a
much higher p-value (more "smoothed", more evidence of LSB tampering)
than a window sitting entirely in untouched territory.
"""

import numpy as np
from PIL import Image
from scipy.stats import chisquare


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def compute_chi_square_pvalue(pixel_values: np.ndarray) -> float:
    """
    Compute the chi-square attack p-value for one chunk of pixel values.

    Steps:
    1. Build a histogram of all 256 possible values in this chunk.
    2. Group into 128 Pairs of Values: (0,1), (2,3), ..., (254,255).
    3. For each pair, the "expected" count under the LSB-embedding
       hypothesis is the AVERAGE of the two observed counts (since
       embedding pushes both values toward their average).
    4. Run a chi-square goodness-of-fit test comparing observed counts
       to these expected counts.

    Returns
    -------
    float
        The p-value from the chi-square test. HIGH p-value (close to
        1.0) = strong evidence of LSB embedding. LOW p-value (close to
        0.0) = looks like a natural, untampered region.
    """
    histogram, _ = np.histogram(pixel_values, bins=256, range=(0, 256))

    observed = []
    expected = []

    for i in range(0, 256, 2):
        count_even = histogram[i]
        count_odd = histogram[i + 1]

        pair_average = (count_even + count_odd) / 2.0

        if pair_average == 0:
            continue

        observed.append(count_even)
        expected.append(pair_average)

    if len(observed) < 2:
        # Not enough data to run a meaningful test (e.g. an almost
        # entirely blank chunk) -- return a neutral p-value.
        return 0.0

    # Normalize expected so it sums to the same total as observed --
    # scipy's chisquare requires this to run correctly.
    observed = np.array(observed, dtype=np.float64)
    expected = np.array(expected, dtype=np.float64)

    expected_sum = expected.sum()
    if expected_sum == 0:
        # Guard against divide-by-zero -- if expected counts are all
        # zero, there's no meaningful test to run here.
        return 0.0

    expected = expected * (observed.sum() / expected_sum)

    # Guard against any pair where expected is still 0 but observed
    # isn't -- scipy's chisquare can't handle a zero expected value.
    valid = expected > 0
    if valid.sum() < 2:
        return 0.0

    observed = observed[valid]
    expected = expected[valid]

    _, p_value = chisquare(f_obs=observed, f_exp=expected)

    if np.isnan(p_value):
        return 0.0

    return float(p_value)


def sliding_window_chi_square(image: np.ndarray, window_size: int = 2000, step_size: int = 1000, max_windows: int = 2000) -> list:
    """
    Slide a window across the FLATTENED image (same pixel order used
    by embed_lsb.py: row-major, via .flatten()), computing the
    chi-square p-value for each window.

    Parameters
    ----------
    window_size : int
        How many consecutive pixels to include in each chi-square test.
    step_size : int
        How far to move the window between tests. Smaller step = finer
        resolution on WHERE embedding starts, but more computation.
    max_windows : int
        Safety cap on the total number of windows processed. Some
        images can have tens of millions of pixels, which would
        otherwise take an impractically long time. When an image would
        produce more than max_windows, we increase the step size so we
        still sample evenly across the WHOLE image, just at coarser
        resolution, keeping runtime bounded regardless of image size.

    Returns
    -------
    list of float
        One p-value per window position, in order along the image.
    """
    flat_pixels = image.flatten()
    num_possible_windows = max(1, (len(flat_pixels) - window_size) // step_size)

    if num_possible_windows > max_windows:
        step_size = max(step_size, (len(flat_pixels) - window_size) // max_windows)

    p_values = []

    for start in range(0, len(flat_pixels) - window_size + 1, step_size):
        window = flat_pixels[start:start + window_size]
        p_value = compute_chi_square_pvalue(window)
        p_values.append(p_value)

        if len(p_values) >= max_windows:
            break

    return p_values

def stego_analyzer_chisquare(image_path: str, window_size: int = 2000, step_size: int = 1000) -> dict:
    """
    Interface function for this module -- runs the sliding-window
    chi-square attack and summarizes the results.

    Returns
    -------
    dict with:
        chisq_max_pvalue   : the single highest p-value found across
                              all windows
        chisq_mean_pvalue  : average p-value across all windows
        chisq_fraction_high: fraction of windows with p-value > 0.9
    """
    gray = load_image_as_grayscale(image_path)
    p_values = sliding_window_chi_square(gray, window_size=window_size, step_size=step_size)

    p_values_array = np.array(p_values)

    return {
        "chisq_max_pvalue": round(float(np.max(p_values_array)), 4),
        "chisq_mean_pvalue": round(float(np.mean(p_values_array)), 4),
        "chisq_fraction_high": round(float(np.mean(p_values_array > 0.9)), 4),
    }


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
            result = stego_analyzer_chisquare(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")