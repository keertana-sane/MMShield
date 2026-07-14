"""
extractor.py
------------
The combiner for the Steganographic Anomaly Analyzer (SAA).

This file brings together every feature module built so far:
    - entropy.py       -> entropy                       (classical)
    - noise.py         -> noise residual stats           (classical)
    - lsb.py           -> LSB ratio/entropy               (classical)
    - frequency.py     -> frequency energy stats           (classical)
    - variance.py      -> local variance + histogram + edges (classical)
    - deep_features.py -> DiT document-context embedding  (deep)

...and combines them into:
    1. A single dictionary (easy for YOU to read/debug -- named keys)
    2. A single fixed-length NumPy array (what the fusion network,
       AATFN, actually needs -- just numbers, in a consistent order)

WHY A FIXED ORDER MATTERS
--------------------------
A neural network doesn't care about key names like "entropy_manual" --
it just sees a list of numbers at fixed positions. If the order of
values changed between images (e.g. sometimes entropy first, sometimes
noise first), the network would learn garbage. So this file defines ONE
fixed order (FEATURE_ORDER below) and always builds the array in that
exact order, every time, for every image.

IMPORTANT SCOPE NOTE
----------------------
The first 15 keys in FEATURE_ORDER are CLASSICAL features, computed on
full-resolution pixels -- these are your actual steganographic evidence
source (entropy, noise, LSB, frequency, variance).

The remaining 32 "doc_context_*" keys come from DiT (deep_features.py)
and represent general DOCUMENT STRUCTURE/CONTEXT, not steganographic
evidence -- two independent experiments (a fine-tuned DiT classifier,
and a Logistic Regression trained on frozen DiT embeddings) both landed
at ~50% accuracy, confirming DiT's embedding carries no usable stego
signal for LSB-based attacks (most likely because DiT's 224x224 resize
destroys single-bit pixel perturbations before the model ever sees
them -- see deep_features.py for full details). They're still included
here because AATFN can benefit from knowing what kind of document it's
looking at, even though this block isn't doing stego detection.
"""

import numpy as np

from entropy import stego_analyzer_entropy
from noise import stego_analyzer_noise
from lsb import stego_analyzer_lsb
from frequency import stego_analyzer_frequency
from variance import stego_analyzer_variance
from deep_features import stego_analyzer_deep
from srm_features import stego_analyzer_srm

# This defines the exact order every feature will appear in the final
# NumPy array. If you add a new feature later, add its key here too.
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
] + [f"doc_context_{i}" for i in range(32)] + [
    "srm_diagonal_ratio",
    "srm_entropy",
    "srm_energy",
]


def extract_all_features_as_dict(image_path: str) -> dict:
    """
    Run every feature module (classical + deep) on one image and merge
    all resulting dictionaries into a single dictionary.

    Returns
    -------
    dict
        All feature name -> value pairs combined, e.g.:
        {
            "entropy_manual": 5.91,
            "noise_mean": 0.0007,
            ...
            "doc_context_0": -0.18,
            ...
        }
    """
    combined = {}

    combined.update(stego_analyzer_entropy(image_path))
    combined.update(stego_analyzer_noise(image_path))
    combined.update(stego_analyzer_lsb(image_path))
    combined.update(stego_analyzer_frequency(image_path))
    combined.update(stego_analyzer_variance(image_path))
    combined.update(stego_analyzer_deep(image_path))
    combined.update(stego_analyzer_srm(image_path))

    return combined


def dict_to_feature_vector(feature_dict: dict) -> np.ndarray:
    """
    Convert a feature dictionary into a fixed-length NumPy array, using
    FEATURE_ORDER to guarantee the same order every single time.

    Raises
    ------
    KeyError
        If a key listed in FEATURE_ORDER is missing from feature_dict --
        this is intentional, so a silently-missing feature never sneaks
        through as an all-zero or misaligned vector.
    """
    vector = [feature_dict[key] for key in FEATURE_ORDER]
    return np.array(vector, dtype=np.float64)


def stego_analyzer(image_path: str) -> np.ndarray:
    """
    The main interface function for the whole feature extractor
    (classical + deep), matching your project's planned shape:

        features = stego_analyzer("invoice.png")
        print(features)

    Returns
    -------
    np.ndarray
        A fixed-length vector (15 classical + 32 doc-context = 47
        total) in the exact order defined by FEATURE_ORDER. This is
        what gets concatenated with the other team members' feature
        vectors in AATFN.
    """
    feature_dict = extract_all_features_as_dict(image_path)
    return dict_to_feature_vector(feature_dict)


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            feature_dict = extract_all_features_as_dict(path)
            feature_vector = dict_to_feature_vector(feature_dict)

            print(f"\n{path}")
            print("Dictionary form (for reading/debugging):")
            print(feature_dict)
            print(f"\nVector form (length {len(feature_vector)}, for the model):")
            print(feature_vector)
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")