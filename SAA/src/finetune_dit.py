"""
finetune_dit.py
----------------
Fine-tunes Microsoft's pretrained DiT (Document Image Transformer) to
distinguish clean vs. stego document images, using your 96 training
images from prepare_dataset.py.

WHAT'S HAPPENING HERE, CONCEPTUALLY
--------------------------------------
DiT was already pretrained by Microsoft on 42 million document images
(IIT-CDIP), using a self-supervised task (predicting hidden/masked
patches of document images). This means it already deeply understands
what documents generally look like -- text regions, tables, whitespace,
structure -- WITHOUT ever being told about steganography.

We are NOT training DiT from scratch. We are taking its already-smart
encoder and attaching a small new "classifier head" on top (just a
couple of small layers ending in 2 outputs: clean or stego). Then we
train ONLY this small addition (and lightly adjust the encoder) using
our 96 labeled images.

This is called "fine-tuning" or "transfer learning" -- we're reusing
almost all of DiT's existing knowledge, and only teaching it the small
extra task of noticing stego-specific patterns on top of what it
already knows about documents in general.

WHY THIS IS FEASIBLE ON A CPU WITH ONLY 96 IMAGES
------------------------------------------------------
Because the encoder already knows document structure, it doesn't need
to learn from scratch. We only need a few passes (epochs) over our
small dataset to nudge it -- this keeps training time reasonable even
without a GPU.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from prepare_dataset import collect_labeled_image_paths, split_train_test


MODEL_NAME = "microsoft/dit-base"
SAVE_PATH = "models/dit_finetuned"
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 1e-5


class DocumentStegoDataset(Dataset):
    """
    A small PyTorch Dataset wrapper around our (image_path, label) pairs.

    PyTorch's training loop expects data to come through a Dataset +
    DataLoader, which handles batching and shuffling for us during
    training.
    """

    def __init__(self, pairs: list, processor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, label = self.pairs[index]

        # DiT expects RGB images (even though our images are grayscale
        # documents) -- convert("RGB") just duplicates the gray value
        # across all 3 channels, which is fine for this model.
        image = Image.open(path).convert("RGB")

        # The processor resizes/normalizes the image exactly the way
        # DiT expects (this matches its original pretraining setup).
        inputs = self.processor(images=image, return_tensors="pt")

        # inputs["pixel_values"] comes back with an extra batch dimension
        # (shape [1, 3, H, W]) -- we squeeze it to [3, H, W] since the
        # DataLoader will add the batch dimension itself when combining
        # multiple examples together.
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


def build_model():
    """
    Load pretrained DiT and configure it for our 2-class problem
    (clean vs. stego).
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "clean", 1: "stego"},
        label2id={"clean": 0, "stego": 1},
        ignore_mismatched_sizes=True,  # original DiT head has different output size
    )

    return processor, model


def train():
    print("Loading pretrained DiT model (first run will download it, ~330MB)...")
    processor, model = build_model()

    print("Preparing dataset...")
    all_pairs = collect_labeled_image_paths()
    train_pairs, test_pairs = split_train_test(all_pairs)

    train_dataset = DocumentStegoDataset(train_pairs, processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cpu")  # no GPU assumed
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"\nStarting training: {NUM_EPOCHS} epochs, {len(train_pairs)} training images")
    print("(This will take a while on CPU -- that's expected, not a bug.)\n")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - loss: {avg_loss:.4f} - train accuracy: {accuracy:.2%}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    processor.save_pretrained(SAVE_PATH)
    print(f"\nFine-tuned model saved to {SAVE_PATH}/")


if __name__ == "__main__":
    train()