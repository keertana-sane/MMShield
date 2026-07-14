"""
noise.py
--------

Compute the noise residual of an image.

WHY NOISE RESIDUAL MATTERS FOR STEGANALYSIS
--------------------------------------------
The "noise residual" of an image is what's left over after you remove the
smooth, predictable content (the actual picture/document) and keep only
the fine-grained, high-frequency variation -- essentially the image's
"texture noise."

How we get it:
    1. Denoise the image (e.g. with a median filter or Gaussian blur) to
       get a smoothed version -- this approximates "what the image would
       look like without noise."
    2. Subtract the smoothed version from the original image.
    3. What's left (the residual) captures fine details: sensor noise,
       compression artifacts, texture -- and potentially, hidden payload.

Why this helps detect steganography:
    Many steganography techniques (especially adaptive ones like WOW,
    S-UNIWARD, HILL) deliberately hide data in "noisy" regions of an
    image, because changes there are less noticeable. This means the
    STATISTICS of the noise residual (its variance, energy, distribution)
    can shift in the presence of a hidden payload, even though the
    original-looking image content barely changes.

We summarize the noise residual with a few numbers (mean, std deviation,
average absolute value) rather than keeping every pixel, so it fits neatly
into the fixed-length feature vector your fusion network expects.
"""

import numpy as np
from PIL import Image, ImageFilter


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def compute_noise_residual(image: np.ndarray, blur_radius: float = 2.0) -> np.ndarray:
    """
    Compute the noise residual of a grayscale image.

    Steps:
    1. Blur the image (this removes fine detail, keeping only smooth content).
    2. Subtract the blurred version from the original.
       Residual = Original - Blurred

    Parameters
    ----------
    image : np.ndarray
        A 2D grayscale image array.
    blur_radius : float
        How strong the blur is. Larger = more smoothing = residual picks
        up coarser texture as well as fine noise. 2.0 is a reasonable
        starting point for document-sized images.

    Returns
    -------
    np.ndarray
        The noise residual, same shape as input, but values can be
        negative (since it's a difference), so this is a signed array.
    """
    pil_image = Image.fromarray(image)
    blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_array = np.array(blurred).astype(np.float64)

    original_float = image.astype(np.float64)
    residual = original_float - blurred_array

    return residual


def summarize_noise_residual(residual: np.ndarray) -> dict:
    """
    Reduce the full noise residual (one value per pixel) down to a small
    set of summary statistics -- this is what actually goes into your
    feature vector.

    Returns
    -------
    dict with:
        noise_mean       : average residual value (should be near 0)
        noise_std        : how spread out the residual values are
        noise_abs_mean   : average absolute residual value (a measure of
                            "how much fine detail/noise" is present overall)
    """
    return {
        "noise_mean": round(float(np.mean(residual)), 4),
        "noise_std": round(float(np.std(residual)), 4),
        "noise_abs_mean": round(float(np.mean(np.abs(residual))), 4),
    }


def stego_analyzer_noise(image_path: str, blur_radius: float = 2.0) -> dict:
    """
    Interface function for this module -- loads an image, computes the
    noise residual, and returns summary statistics.
    """
    gray = load_image_as_grayscale(image_path)
    residual = compute_noise_residual(gray, blur_radius=blur_radius)
    return summarize_noise_residual(residual)


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_noise(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")
            