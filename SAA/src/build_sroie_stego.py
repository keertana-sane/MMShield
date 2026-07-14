"""
build_sroie_stego.py
-----------------------
Embeds scattered LSB payloads (15% pixel density, same as FUNSD/CORD)
into the SROIE sample images, creating the stego counterpart dataset.
"""

from embed_lsb_scattered import build_scattered_stego_dataset


if __name__ == "__main__":
    PAYLOAD_RATE = 0.15

    build_scattered_stego_dataset(
        "datasets/sroie",
        "datasets/stego_scattered/sroie",
        payload_rate=PAYLOAD_RATE,
        base_seed=3000,
    )