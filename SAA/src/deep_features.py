"""
deep_features.py
----------------
Extracts a document-context embedding from Microsoft's pretrained DiT
(Document Image Transformer).

IMPORTANT SCOPE NOTE -- READ THIS FIRST
------------------------------------------
Based on two independent experiments (finetune_dit.py and
test_dit_raw_embeddings.py), we confirmed that DiT's embedding does
NOT carry usable steganographic signal for our LSB-based attacks:
    - A fine-tuned DiT classifier stayed at ~50% accuracy (random chance)
    - A Logistic Regression trained directly on frozen DiT embeddings
      also scored exactly 50%, with 0% variation across 5 folds

The most likely explanation: DiT resizes every image to 224x224 before
processing it. This resizing (interpolation) destroys the single-bit
pixel perturbations that LSB steganography relies on, before the model
ever sees them.

Given this, DiT's embedding in this project is used ONLY for what it's
actually good at: representing general DOCUMENT STRUCTURE AND CONTEXT
(what type of document this is, its layout, its visual "shape") --
NOT for detecting steganography. Steganographic detection is handled
entirely by the classical, full-resolution features in entropy.py,
noise.py, lsb.py, frequency.py, and variance.py.

This keeps the pipeline honest: DiT contributes complementary context
to the AATFN fusion step, while the classical features remain the
actual evidence source for tampering.
"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "microsoft/dit-base"

# Loaded once and reused across calls, so we don't reload the model
# from disk every single time stego_analyzer_deep() is called.
_processor = None
_model = None


def _load_model():
    """
    Load the frozen, pretrained DiT encoder (no classification head,
    no fine-tuning). Cached globally so repeated calls are fast.
    """
    global _processor, _model

    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()  # inference mode -- we are not training this model

    return _processor, _model


def extract_document_context_embedding(image_path: str) -> np.ndarray:
    """
    Run an image through frozen DiT and return its 768-dimensional
    pooled embedding, representing general document structure/context
    (NOT steganographic evidence -- see module docstring above).

    Returns
    -------
    np.ndarray
        A 768-length vector.
    """
    processor, model = _load_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool across all patch tokens to get one vector per image.
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

    return embedding.numpy()


def reduce_embedding_dimensionality(embedding: np.ndarray, target_size: int = 32) -> np.ndarray:
    """
    DiT's raw embedding is 768-dimensional -- quite large compared to
    our 15 classical features. To avoid the deep features completely
    dominating the final fused vector (768 vs 15 is a huge imbalance),
    we reduce it down to a smaller fixed size.

    This uses simple average-pooling in chunks (not a learned
    projection) -- it's a deliberately simple, transparent method
    appropriate for a prototype-stage project. A learned projection
    (e.g. PCA fit on a larger dataset, or a small trained linear layer)
    would be a reasonable improvement for a later iteration.

    Parameters
    ----------
    embedding : np.ndarray
        The original 768-length embedding.
    target_size : int
        Desired output length (default 32). Must evenly divide 768
        for this simple chunk-averaging approach to split evenly.

    Returns
    -------
    np.ndarray
        A vector of length target_size.
    """
    chunk_size = len(embedding) // target_size
    reduced = np.array([
        embedding[i * chunk_size:(i + 1) * chunk_size].mean()
        for i in range(target_size)
    ])
    return reduced


def stego_analyzer_deep(image_path: str, target_size: int = 32) -> dict:
    """
    Interface function for this module, matching the style of your
    other stego_analyzer_* functions.

    Returns
    -------
    dict
        {"doc_context_0": ..., "doc_context_1": ..., ...} -- named
        this way (not "stego_*") to make clear these are document
        context features, not steganographic evidence.
    """
    raw_embedding = extract_document_context_embedding(image_path)
    reduced_embedding = reduce_embedding_dimensionality(raw_embedding, target_size=target_size)

    return {
        f"doc_context_{i}": round(float(value), 4)
        for i, value in enumerate(reduced_embedding)
    }


if __name__ == "__main__":
    import os

    test_images = [
        "datasets/funsd/funsd_0.png",
        "datasets/cord/cord_0.png",
    ]

    for path in test_images:
        if os.path.exists(path):
            result = stego_analyzer_deep(path)
            print(f"\n{path}")
            print(f"Number of document-context features: {len(result)}")
            print(f"First 5 values: {dict(list(result.items())[:5])}")
        else:
            print(f"[skip] {path} not found - run this script from your SAA/ folder")