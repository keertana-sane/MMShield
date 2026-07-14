"""
embed_lsb_random.py
---------------------
A fixed version of the LSB embedder, designed specifically to prevent
a CNN from learning a "shortcut" instead of genuine steganalysis.

WHAT WAS WRONG WITH THE ORIGINAL embed_lsb.py FOR CNN TRAINING
--------------------------------------------------------------------
The original embedder always started writing at pixel 0, and cycled
through only 5 fixed messages. A CNN trained on that dataset could
"solve" the task by memorizing "does the start of this image decode to
one of these 5 known bit patterns?" -- which is memorization, not
steganalysis, and wouldn't generalize to a real, unseen attack.

THE FIX
--------
1. RANDOM POSITION: each image gets its message embedded starting at a
   randomly chosen pixel position (not always 0). This forces a model
   to learn to recognize LSB tampering STRUCTURALLY, wherever it
   occurs, rather than checking one fixed location.

2. RANDOM PAYLOAD: instead of cycling through 5 fixed sentences, we
   generate a random sequence of bits for each image. This removes any
   fixed, learnable "signature" content -- the model can only succeed
   by recognizing the STATISTICAL properties LSB replacement leaves
   behind (e.g. disrupted local pixel correlation), not by recognizing
   specific bit patterns.

3. VARIABLE LENGTH: payload length also varies per image (within a
   realistic range), so the model can't key off of a fixed payload
   size either.
"""

import os
import random
import numpy as np
from PIL import Image


def generate_random_bits(num_bits: int, seed: int = None) -> str:
    """Generate a random string of '0'/'1' characters of the given length."""
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(num_bits))


def embed_random_message(
    image_path: str,
    output_path: str,
    min_bits: int = 300,
    max_bits: int = 900,
    seed: int = None,
) -> dict:
    """
    Embed a random bit string at a random starting position within the
    image's LSBs.

    Returns
    -------
    dict
        Metadata about what was embedded (start position, length) --
        useful for debugging/verification, not used by the model
        itself.
    """
    rng = random.Random(seed)

    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    flat_pixels = pixels.flatten().copy()

    num_bits = rng.randint(min_bits, max_bits)
    message_bits = generate_random_bits(num_bits, seed=seed)

    max_start_position = len(flat_pixels) - num_bits
    start_position = rng.randint(0, max_start_position)

    for offset, bit_char in enumerate(message_bits):
        bit = int(bit_char)
        index = start_position + offset
        flat_pixels[index] = (flat_pixels[index] & 0xFE) | bit

    stego_pixels = flat_pixels.reshape(pixels.shape)
    stego_image = Image.fromarray(stego_pixels.astype(np.uint8))
    stego_image.save(output_path)

    return {
        "start_position": start_position,
        "num_bits": num_bits,
    }


def build_randomized_stego_dataset(clean_dir: str, stego_dir: str, base_seed: int = 0) -> None:
    """
    For every .png file in clean_dir, embed a RANDOM message at a
    RANDOM position, saving the result to stego_dir with the same
    filename.
    """
    os.makedirs(stego_dir, exist_ok=True)

    clean_files = sorted(f for f in os.listdir(clean_dir) if f.endswith(".png"))

    for i, filename in enumerate(clean_files):
        clean_path = os.path.join(clean_dir, filename)
        stego_path = os.path.join(stego_dir, filename)

        metadata = embed_random_message(clean_path, stego_path, seed=base_seed + i)
        print(f"  {filename}: embedded {metadata['num_bits']} bits starting at pixel {metadata['start_position']}")

    print(f"Done: embedded randomized messages into {len(clean_files)} images from {clean_dir} -> {stego_dir}")


if __name__ == "__main__":
    build_randomized_stego_dataset("datasets/funsd", "datasets/stego_random/funsd", base_seed=100)
    build_randomized_stego_dataset("datasets/cord", "datasets/stego_random/cord", base_seed=200)