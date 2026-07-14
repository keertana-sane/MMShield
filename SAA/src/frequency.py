"""
frequency.py
------------
Compute frequency-domain features of an image.

WHY FREQUENCY ANALYSIS MATTERS FOR STEGANALYSIS
-------------------------------------------------
Every image can be thought of as a mix of "frequencies":
    - LOW frequencies  = smooth, slowly-changing areas (backgrounds,
      large flat regions)
    - HIGH frequencies = sharp, fast-changing areas (edges, fine text,
      fine texture, noise)

We can decompose an image into its frequency components using the
2D Fast Fourier Transform (FFT). This is the same underlying math idea
behind JPEG compression's DCT (Discrete Cosine Transform) -- both convert
"pixel space" into "frequency space."

Why this helps detect steganography:
    - JPEG steganography (e.g. F5, JSteg) embeds data directly into
      frequency-domain coefficients, so tampering can show up as unusual
      energy in specific frequency bands.
    - Even for spatial-domain (LSB-style) steganography, flipping many
      individual bits tends to inject small amounts of extra HIGH
      frequency energy into the image, since bit-flips are abrupt,
      small-scale changes -- similar in nature to noise or sharp edges.

We summarize the frequency content with a few numbers:
    - total_frequency_energy : overall magnitude of all frequency
      components combined
    - high_freq_ratio        : what fraction of that energy lives in the
      HIGH frequency band vs low frequency band

A natural document image is expected to be dominated by low-frequency
energy (large flat regions). An unusually high proportion of high-frequency
energy can be a signal worth flagging -- though, like every single feature
in this module, it's not proof on its own; it's one piece of evidence to
combine with entropy, noise, and LSB stats.
"""

import numpy as np
from PIL import Image


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def compute_fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute the 2D FFT of the image and return the magnitude spectrum.

    Steps:
    1. np.fft.fft2 converts the image from pixel-space to frequency-space.
       The result is complex numbers (real + imaginary parts).
    2. np.fft.fftshift moves the zero-frequency (low-frequency / "DC")
       component to the center of the array, which makes it much easier
       to reason about "distance from center = frequency."
    3. np.abs(...) takes the magnitude of each complex value, giving us
       a real-valued "how much energy is at this frequency" map.

    Returns
    -------
    np.ndarray
        Same shape as input image. Center = low frequencies,
        edges/corners = high frequencies.
    """
    fft_result = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude = np.abs(fft_shifted)
    return magnitude


def split_low_high_frequency_energy(magnitude: np.ndarray, low_freq_radius_fraction: float = 0.25) -> tuple:
    """
    Split the frequency magnitude map into "low frequency" (a disk around
    the center) and "high frequency" (everything else) regions, then sum
    the energy in each.

    Parameters
    ----------
    magnitude : np.ndarray
        Output of compute_fft_magnitude.
    low_freq_radius_fraction : float
        What fraction of the image's half-diagonal counts as "low
        frequency." 0.25 means the innermost quarter-radius circle
        around the center is treated as low frequency.

    Returns
    -------
    (low_energy, high_energy) : tuple of floats
    """
    height, width = magnitude.shape
    center_y, center_x = height // 2, width // 2

    # Build a grid of distances from the center for every pixel
    y_indices, x_indices = np.indices((height, width))
    distances = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

    max_radius = np.sqrt(center_y ** 2 + center_x ** 2)
    low_freq_cutoff = max_radius * low_freq_radius_fraction

    low_freq_mask = distances <= low_freq_cutoff
    high_freq_mask = ~low_freq_mask

    low_energy = float(np.sum(magnitude[low_freq_mask]))
    high_energy = float(np.sum(magnitude[high_freq_mask]))

    return low_energy, high_energy


def stego_analyzer_frequency(image_path: str) -> dict:
    """
    Interface function for this module -- loads an image, computes its
    frequency spectrum, and returns summary statistics.

    Returns
    -------
    dict with:
        total_frequency_energy : sum of all frequency magnitudes
        high_freq_ratio        : fraction of total energy that is
                                  "high frequency" (0 to 1)
    """
    gray = load_image_as_grayscale(image_path)
    magnitude = compute_fft_magnitude(gray)

    low_energy, high_energy = split_low_high_frequency_energy(magnitude)
    total_energy = low_energy + high_energy

    high_freq_ratio = high_energy / total_energy if total_energy > 0 else 0.0

    return {
        "total_frequency_energy": round(total_energy, 2),
        "high_freq_ratio": round(float(high_freq_ratio), 4),
    }


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_frequency(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")