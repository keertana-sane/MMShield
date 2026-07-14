"""
entropy.py
----------
Step 1 of the Steganographic Anomaly Analyzer (SAA):
Compute the Shannon entropy of an image.

WHY ENTROPY MATTERS FOR STEGANALYSIS
-------------------------------------
Entropy measures how "spread out" or "unpredictable" the pixel values in an
image are. A natural document image (mostly white background + some black
text) tends to have a fairly LOW-to-MODERATE entropy, because most pixels
are similar (white background dominates).

When someone hides data inside an image (e.g. via LSB steganography), they
are flipping many individual bits across the image. This tends to make the
pixel value distribution slightly more "random" than a natural image would
be -- which can show up as a small shift in entropy.

Entropy alone is a weak signal (it can be fooled), but it's your first
building block. Later this becomes one number in your feature vector,
combined with variance, noise residual, LSB histogram, and frequency energy.
"""

import numpy as np
from PIL import Image


def compute_entropy_manual(image: np.ndarray) -> float:
    """
    Manually compute the Shannon entropy of a grayscale image.

    Steps:
    1. Build a histogram of pixel intensity values (0-255).
    2. Normalize the histogram into a probability distribution
       (i.e. what fraction of pixels have each intensity value).
    3. Apply the Shannon entropy formula:

           H = -sum( p_i * log2(p_i) )

       for every intensity value i that actually appears in the image
       (we skip p_i == 0, since log2(0) is undefined and those bins
       contribute nothing anyway).

    Parameters
    ----------
    image : np.ndarray
        A 2D grayscale image array with pixel values in [0, 255].

    Returns
    -------
    float
        The Shannon entropy in bits. Ranges from 0 (completely uniform,
        e.g. a plain white image) to 8 (maximally random, every pixel
        value equally likely, for an 8-bit image).
    """
    # Step 1: histogram -- count how many pixels have each value 0-255
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))

    # Step 2: normalize into probabilities (this is just counts / total)
    total_pixels = image.size
    probabilities = histogram / total_pixels

    # Step 3: keep only non-zero probabilities (avoid log2(0))
    probabilities = probabilities[probabilities > 0]

    # Shannon entropy formula
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return float(entropy)


def compute_entropy_library(image: np.ndarray) -> float:
    """
    Same result as compute_entropy_manual, but using skimage's built-in
    function. Useful as a sanity check that your manual version is correct.
    """
    from skimage.measure import shannon_entropy
    return float(shannon_entropy(image))


def load_image_as_grayscale(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to grayscale as a NumPy array.
    Steganalysis features are usually computed on grayscale, since color
    channels can be handled separately later if needed.
    """
    img = Image.open(path).convert("L")  # "L" = grayscale mode in PIL
    return np.array(img)


def stego_analyzer_entropy(image_path: str) -> dict:
    """
    The interface function for this module, matching the shape described
    in your project plan:

        features = stego_analyzer("invoice.png")
        print(features)

    For now this only returns the entropy piece. Later, variance.py,
    lsb.py, frequency.py, and noise.py will each contribute their own
    keys to a combined dictionary in extractor.py.
    """
    gray = load_image_as_grayscale(image_path)
    manual = compute_entropy_manual(gray)
    library = compute_entropy_library(gray)

    return {
        "entropy_manual": round(manual, 4),
        "entropy_library": round(library, 4),
    }


if __name__ == "__main__":
    # Quick test: run entropy on one FUNSD image and one CORD image
    # so you can see real numbers come out.
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_entropy(path)
            print(f"{path} -> {result}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")