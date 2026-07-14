"""
embed_lsb_scattered.py
------------------------
A realistic LSB embedder that scatters the payload pseudo-randomly
across a meaningful FRACTION of the image's pixels, matching how real
steganography tools (Steghide, OpenStego) and academic embedding
algorithms (WOW, HILL, S-UNIWARD) actually operate -- and matching the
payload DENSITY (bits-per-pixel) that published steganalysis papers
(SRNet, YeNet, XuNet) were actually evaluated against.

WHY WE'RE CHANGING THE EMBEDDING STYLE
------------------------------------------
Our six previous experiments all failed for variations of the same
root cause: the payload (300-900 bits) was spread across only a tiny
fraction of each image's ~750,000+ pixels (roughly 0.001 bits per
pixel). Every published steganalysis technique -- including the ones
in our own literature review -- was designed and evaluated at payload
rates of roughly 0.1 to 0.4 bits per pixel, a density 100-400x higher
than what we were testing against. No detection method (classical
statistics, chi-square, CNNs) can find a signal that sparse.

This is not an artificial shortcut. Real LSB tools do not write
messages sequentially starting at pixel 0 (as our earlier embed_lsb.py
did) -- they use a PSEUDO-RANDOM scatter pattern across a meaningful
fraction of the image, typically seeded by a shared key. This script
reflects that real-world behavior more accurately than our earlier,
simplified version.

HOW IT WORKS
-------------
1. Choose a payload rate (fraction of pixels to modify), e.g. 0.15
   (15% of all pixels).
2. Using a seeded random number generator, select that fraction of
   pixel positions, scattered pseudo-randomly across the whole image
   (not sequential, not clustered in one region).
3. Fill those positions with random bits (representing an embedded
   payload -- for detection purposes, the actual message content
   doesn't matter, only that a meaningful fraction of pixels have had
   their LSB overwritten).
"""

import os
import numpy as np
from PIL import Image


def embed_scattered_message(
    image_path: str,
    output_path: str,
    payload_rate: float = 0.15,
    seed: int = None,
) -> dict:
    """
    Embed a scattered LSB payload across payload_rate fraction of the
    image's pixels, at pseudo-random positions.
    """
    rng = np.random.RandomState(seed)

    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    flat_pixels = pixels.flatten().copy()

    total_pixels = len(flat_pixels)
    num_pixels_to_modify = int(total_pixels * payload_rate)

    positions = rng.choice(total_pixels, size=num_pixels_to_modify, replace=False)
    random_bits = rng.randint(0, 2, size=num_pixels_to_modify)

    flat_pixels[positions] = (flat_pixels[positions] & 0xFE) | random_bits

    stego_pixels = flat_pixels.reshape(pixels.shape)
    stego_image = Image.fromarray(stego_pixels.astype(np.uint8))
    stego_image.save(output_path)

    return {
        "total_pixels": total_pixels,
        "pixels_modified": num_pixels_to_modify,
        "payload_rate": payload_rate,
    }


def build_scattered_stego_dataset(clean_dir: str, stego_dir: str, payload_rate: float = 0.15, base_seed: int = 0) -> None:
    """
    For every .png file in clean_dir, embed a scattered LSB payload at
    payload_rate density, saving to stego_dir with the same filename.
    """
    os.makedirs(stego_dir, exist_ok=True)

    clean_files = sorted(f for f in os.listdir(clean_dir) if f.endswith(".png"))

    for i, filename in enumerate(clean_files):
        clean_path = os.path.join(clean_dir, filename)
        stego_path = os.path.join(stego_dir, filename)

        metadata = embed_scattered_message(clean_path, stego_path, payload_rate=payload_rate, seed=base_seed + i)
        print(f"  {filename}: modified {metadata['pixels_modified']} / {metadata['total_pixels']} pixels "
              f"({metadata['payload_rate']:.1%} rate)")

    print(f"Done: embedded scattered payloads into {len(clean_files)} images from {clean_dir} -> {stego_dir}")


if __name__ == "__main__":
    PAYLOAD_RATE = 0.15  # 15% of pixels modified -- within the 0.1-0.4 bpp range used in steganalysis literature

    build_scattered_stego_dataset("datasets/funsd", "datasets/stego_scattered/funsd", payload_rate=PAYLOAD_RATE, base_seed=1000)
    build_scattered_stego_dataset("datasets/cord", "datasets/stego_scattered/cord", payload_rate=PAYLOAD_RATE, base_seed=2000)