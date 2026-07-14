"""
test_dit_raw_embeddings.py
----------------------------
Experiment: does the FROZEN, non-fine-tuned DiT embedding already
contain any steganographic signal at all?

WHY THIS EXPERIMENT MATTERS
-----------------------------
Our fine-tuning experiment (finetune_dit.py) showed DiT-based
classification stuck at ~50% accuracy -- no better than random
guessing. But that result alone is ambiguous: it could mean either:

    (a) DiT's pretrained representation contains no steganographic
        information at all (the 224x224 resize destroyed it before
        the model ever saw it), OR
    (b) The representation DOES contain some signal, but our
        fine-tuning setup (5 epochs, tiny dataset, frozen backbone
        mostly untouched) simply failed to extract it.

This experiment isolates which explanation is correct. We:
    1. Run every image through the ORIGINAL, untouched pretrained DiT
       (no fine-tuning at all) and extract its 768-dimensional
       embedding (the pooled output of the transformer, before any
       classification head).
    2. Train a completely separate, simple classifier (Logistic
       Regression) directly on these frozen embeddings.

Logistic Regression is a good choice here because it's a very simple,
low-capacity model -- if it can't find ANY separating signal in the
embeddings, that's strong evidence the embeddings themselves don't
contain useful information for this task (rather than "our fine-tuning
process was too weak").

INTERPRETING THE RESULT
--------------------------
- If Logistic Regression also lands around 50% accuracy: this confirms
  explanation (a) -- the pretrained DiT representation itself contains
  little/no steganographic signal, most likely because image resizing
  destroys LSB-level information before the model ever processes it.
  This is a strong, defensible finding for the report.

- If Logistic Regression does notably BETTER than 50%: this would
  suggest the raw embeddings DO carry some signal, and our fine-tuning
  setup needs improvement (more epochs, different learning rate, etc.)
  rather than DiT being fundamentally unsuitable.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from prepare_dataset import collect_labeled_image_paths


MODEL_NAME = "microsoft/dit-base"


def extract_raw_embedding(image_path: str, processor, model) -> np.ndarray:
    """
    Run one image through the FROZEN, untouched pretrained DiT and
    return its 768-dimensional pooled embedding.

    We use AutoModel (not AutoModelForImageClassification) here on
    purpose -- this loads just the base transformer encoder, with NO
    classification head at all, so there's no risk of accidentally
    using any fine-tuned weights.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():  # no training happening, so no need to track gradients
        outputs = model(**inputs)

    # last_hidden_state shape: [1, num_patches + 1, hidden_size]
    # We take the mean across all patch tokens as a simple pooled
    # embedding (a common, simple way to get one vector per image).
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    return embedding.numpy()


def main():
    print("Loading FROZEN pretrained DiT (no fine-tuning, no classification head)...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()  # inference mode, not training mode

    print("Extracting embeddings for all 120 images (clean + stego)...")
    all_pairs = collect_labeled_image_paths()

    embeddings = []
    labels = []

    for i, (path, label) in enumerate(all_pairs):
        embedding = extract_raw_embedding(path, processor, model)
        embeddings.append(embedding)
        labels.append(label)

        if (i + 1) % 20 == 0:
            print(f"  processed {i + 1}/{len(all_pairs)} images")

    X = np.array(embeddings)
    y = np.array(labels)

    print(f"\nEmbedding matrix shape: {X.shape}")  # should be (120, 768)

    print("\nTraining Logistic Regression with 5-fold cross-validation...")
    classifier = LogisticRegression(max_iter=1000)
    scores = cross_val_score(classifier, X, y, cv=5)

    print(f"\nCross-validation accuracy per fold: {scores}")
    print(f"Mean accuracy: {scores.mean():.2%}")
    print(f"Standard deviation: {scores.std():.2%}")

    print("\n--- Interpretation ---")
    if scores.mean() < 0.60:
        print("Result is close to random guessing (50%).")
        print("This suggests the frozen DiT embedding itself carries little")
        print("steganographic signal -- most likely because resizing to")
        print("224x224 destroys LSB-level information before the model")
        print("ever processes the image.")
    else:
        print("Result is notably better than random guessing.")
        print("This suggests the raw embeddings DO carry some signal, and")
        print("the earlier fine-tuning setup may need adjustment (more")
        print("epochs, different learning rate, etc.) to actually use it.")


if __name__ == "__main__":
    main()
    