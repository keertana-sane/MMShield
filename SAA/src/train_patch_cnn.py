"""
train_patch_cnn.py
--------------------
Trains a small CNN from scratch to detect randomized LSB steganography,
using random FULL-RESOLUTION crops (never resized/downsampled) as
input.

WHY FULL-RESOLUTION CROPS, NOT RESIZED IMAGES
--------------------------------------------------
Our earlier DiT experiment failed because DiT resizes every image to
224x224, which destroys single-bit pixel perturbations through
interpolation before the model ever sees them.

Instead of resizing the WHOLE image down, we take a random crop of
FIXED PIXEL SIZE (e.g. 256x256) directly from the original,
full-resolution image -- no interpolation, no downsampling. Every
pixel the model sees is an untouched original pixel value.

Since our embedding (embed_lsb_random.py) now places the message at a
RANDOM position in each image, sometimes a given crop will contain
part of the embedded region, sometimes it won't -- this is expected
and realistic, similar to how patch-based real steganalysis works.

WHY A SMALL CNN, TRAINED FROM SCRATCH
------------------------------------------
We are NOT using a pretrained model this time. A small CNN trained
directly on our own crops, from scratch, has no resizing baggage --
though it also means it has much less "prior knowledge" to lean on,
so don't expect dramatic accuracy from only 120 source images.

WHY THIS TEST IS FAIR (NO MEMORIZATION SHORTCUT)
-------------------------------------------------------
Because embed_lsb_random.py used a different random position and a
different random bit-string payload for every image, there is no fixed
pattern for the model to memorize. If it achieves accuracy above
chance (50%), that is genuine evidence it learned something about the
STRUCTURE of LSB tampering, not a shortcut.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


CROP_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3


def collect_randomized_pairs() -> list:
    """
    Build a list of (image_path, label) pairs from the randomized
    dataset: clean images (label 0) and their stego_random counterparts
    (label 1).
    """
    pairs = []

    clean_dirs = {
        "datasets/funsd": "datasets/stego_random/funsd",
        "datasets/cord": "datasets/stego_random/cord",
    }

    for clean_dir, stego_dir in clean_dirs.items():
        for filename in sorted(os.listdir(clean_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(clean_dir, filename), 0))

        for filename in sorted(os.listdir(stego_dir)):
            if filename.endswith(".png"):
                pairs.append((os.path.join(stego_dir, filename), 1))

    return pairs


class RandomCropStegoDataset(Dataset):
    """
    For each (image_path, label) pair, returns a random full-resolution
    crop of size CROP_SIZE x CROP_SIZE, normalized to [0, 1].
    """

    def __init__(self, pairs: list, crop_size: int = CROP_SIZE):
        self.pairs = pairs
        self.crop_size = crop_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, label = self.pairs[index]

        image = Image.open(path).convert("L")
        image_array = np.array(image, dtype=np.float32) / 255.0

        height, width = image_array.shape
        crop_size = self.crop_size

        max_y = max(height - crop_size, 0)
        max_x = max(width - crop_size, 0)

        y = random.randint(0, max_y)
        x = random.randint(0, max_x)

        crop = image_array[y:y + crop_size, x:x + crop_size]

        if crop.shape != (crop_size, crop_size):
            padded = np.zeros((crop_size, crop_size), dtype=np.float32)
            padded[:crop.shape[0], :crop.shape[1]] = crop
            crop = padded

        crop_tensor = torch.tensor(crop).unsqueeze(0)

        return crop_tensor, label


class SmallStegoCNN(nn.Module):
    """
    A small CNN, trained from scratch. Deliberately simple -- a handful
    of convolutional layers, no pretrained weights, no aggressive
    pooling/downsampling early on, so fine pixel-level detail is
    preserved as long as possible before the final classification
    layers.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train():
    print("Collecting randomized clean/stego image pairs...")
    all_pairs = collect_randomized_pairs()
    print(f"Total images: {len(all_pairs)}")

    labels_only = [label for _, label in all_pairs]
    train_pairs, test_pairs = train_test_split(
        all_pairs, test_size=0.2, random_state=42, stratify=labels_only
    )
    print(f"Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    train_dataset = RandomCropStegoDataset(train_pairs)
    test_dataset = RandomCropStegoDataset(test_pairs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    model = SmallStegoCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {NUM_EPOCHS} epochs on CPU (this will take a while)...\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for crops, labels in train_loader:
            crops = crops.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()
            outputs = model(crops)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        avg_loss = total_loss / len(train_loader)

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for crops, labels in test_loader:
                crops = crops.to(device)
                labels = torch.tensor(labels).to(device)
                outputs = model(crops)
                predictions = outputs.argmax(dim=1)
                test_correct += (predictions == labels).sum().item()
                test_total += labels.size(0)

        test_accuracy = test_correct / test_total

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - loss: {avg_loss:.4f} - "
              f"train acc: {train_accuracy:.2%} - test acc: {test_accuracy:.2%}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/patch_cnn.pt")
    print("\nModel saved to models/patch_cnn.pt")


if __name__ == "__main__":
    train()
