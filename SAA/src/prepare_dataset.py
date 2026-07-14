"""
prepare_dataset.py
-------------------
Organizes your clean and stego images into a labeled dataset, split into
train and test sets, ready to be used for fine-tuning DiT.

WHAT THIS DOES
---------------
Right now you have:
    datasets/funsd/           (30 clean forms)
    datasets/cord/             (30 clean receipts)
    datasets/stego/funsd/      (30 stego forms, same filenames)
    datasets/stego/cord/        (30 stego receipts, same filenames)

This script walks through all of these, assigns each image a label
(0 = clean, 1 = stego), and splits everything into a TRAIN set and a
TEST set. The test set is held out and never used for training -- it's
only used afterward to check how well the fine-tuned model actually
generalizes to images it hasn't seen.

WHY WE NEED A SPLIT AT ALL (WITH ONLY 120 IMAGES)
----------------------------------------------------
Even with a small dataset, it's important to never evaluate a model on
the same images it was trained on -- that would make it look artificially
good (it could just be memorizing, not learning general patterns). We
use a 80/20 split: 96 images to train on, 24 held out to test on.

This is intentionally simple (no fancy stratified k-fold or cross
validation) since your dataset is small and this is a proof-of-concept
stage, not a final production model.
"""

import os
import random


def collect_labeled_image_paths() -> list:
    """
    Walk through all clean and stego folders and build a list of
    (image_path, label) pairs.

    Returns
    -------
    list of (str, int) tuples
        label 0 = clean, label 1 = stego
    """
    pairs = []

    clean_dirs = ["datasets/funsd", "datasets/cord"]
    stego_dirs = ["datasets/stego/funsd", "datasets/stego/cord"]

    for clean_dir in clean_dirs:
        for filename in sorted(os.listdir(clean_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(clean_dir, filename), 0))

    for stego_dir in stego_dirs:
        for filename in sorted(os.listdir(stego_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(stego_dir, filename), 1))

    return pairs


def split_train_test(pairs: list, test_fraction: float = 0.2, seed: int = 42) -> tuple:
    """
    Shuffle and split the labeled pairs into train and test sets.

    Parameters
    ----------
    pairs : list of (path, label)
    test_fraction : float
        Fraction of data to hold out for testing (0.2 = 20%).
    seed : int
        Fixed random seed so the split is reproducible -- running this
        script twice gives you the exact same train/test split, which
        matters for fair comparisons later.

    Returns
    -------
    (train_pairs, test_pairs) : tuple of lists
    """
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    split_index = int(len(shuffled) * (1 - test_fraction))
    train_pairs = shuffled[:split_index]
    test_pairs = shuffled[split_index:]

    return train_pairs, test_pairs


if __name__ == "__main__":
    all_pairs = collect_labeled_image_paths()
    print(f"Total labeled images found: {len(all_pairs)}")

    clean_count = sum(1 for _, label in all_pairs if label == 0)
    stego_count = sum(1 for _, label in all_pairs if label == 1)
    print(f"  clean: {clean_count}, stego: {stego_count}")

    train_pairs, test_pairs = split_train_test(all_pairs)
    print(f"\nTrain set: {len(train_pairs)} images")
    print(f"Test set: {len(test_pairs)} images")

    print("\nFirst 5 train examples:")
    for path, label in train_pairs[:5]:
        print(f"  {path} -> label {label}")

    print("\nFirst 5 test examples:")
    for path, label in test_pairs[:5]:
        print(f"  {path} -> label {label}")