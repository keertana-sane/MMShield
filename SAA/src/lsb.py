"""
lsb.py
------

Compute LSB (Least Significant Bit) statistics of an image.

WHY LSB STATISTICS MATTER FOR STEGANALYSIS
--------------------------------------------
Every pixel value in an 8-bit grayscale image is stored as a number
0-255, which in binary is 8 bits, e.g.:

    200 = 11001000

The LAST bit (the "least significant bit") barely affects the pixel's
appearance -- flipping it changes the value by at most 1 (e.g. 200 -> 201),
which is invisible to the human eye. This makes it a favorite hiding spot
for simple steganography: replace the LSB of many pixels with bits of a
secret message.

In a NATURAL image, the least significant bit of each pixel tends to be
somewhat random already (due to camera sensor noise, compression, etc.),
but it's usually NOT a perfect 50/50 split of 0s and 1s -- it often
correlates weakly with the actual image content.

In an image with LSB steganography embedded, the least significant bits
get overwritten with message bits, which (if the message is compressed or
encrypted) look statistically like a close to perfect 50/50 random
distribution. This is the classic giveaway: LSB steganography tends to
push the LSB plane closer to pure randomness than a natural image would be.

We capture this with:
    1. lsb_ratio: what fraction of pixels have LSB == 1 (should be close
       to 0.5 for stego images, but can vary more for natural images).
    2. lsb_entropy: the Shannon entropy of the LSB bit-plane itself
       (treated as its own tiny "image" of 0s and 1s). Closer to 1.0 bit
       means closer to a perfect coin-flip distribution.
"""

import numpy as np
from PIL import Image


def load_image_as_grayscale(path: str) -> np.ndarray:
    """Load an image from disk and convert it to grayscale as a NumPy array."""
    img = Image.open(path).convert("L")
    return np.array(img)


def extract_lsb_plane(image: np.ndarray) -> np.ndarray:
    """
    Extract the least significant bit of every pixel.

    Using a bitwise AND with 1 isolates just the last bit:
        e.g. 200 (11001000) & 1 = 0
             201 (11001001) & 1 = 1

    Returns
    -------
    np.ndarray
        Same shape as input, but every value is either 0 or 1.
    """
    return image & 1


def compute_lsb_ratio(lsb_plane: np.ndarray) -> float:
    """
    Fraction of pixels where the LSB is 1.
    A value close to 0.5 suggests the LSB plane looks close to random
    (a possible stego indicator); values further from 0.5 suggest more
    natural, non-random structure.
    """
    return float(np.mean(lsb_plane))


def compute_lsb_entropy(lsb_plane: np.ndarray) -> float:
    """
    Shannon entropy of the LSB plane, treating it as a two-value
    (0/1) distribution.

    Max possible value is 1.0 bit (perfect 50/50 split of 0s and 1s).
    A value very close to 1.0 can indicate the LSBs have been overwritten
    with something that looks like random data (e.g. an embedded payload).
    """
    ones_fraction = np.mean(lsb_plane)
    zeros_fraction = 1 - ones_fraction

    # Avoid log2(0) if the plane is all 0s or all 1s
    probabilities = [p for p in (ones_fraction, zeros_fraction) if p > 0]

    entropy = -sum(p * np.log2(p) for p in probabilities)
    return float(entropy)


def stego_analyzer_lsb(image_path: str) -> dict:
    """
    Interface function for this module -- loads an image, extracts the
    LSB plane, and returns summary statistics.
    """
    gray = load_image_as_grayscale(image_path)
    lsb_plane = extract_lsb_plane(gray)

    return {
        "lsb_ratio": round(compute_lsb_ratio(lsb_plane), 4),
        "lsb_entropy": round(compute_lsb_entropy(lsb_plane), 4),
    }


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_lsb(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")