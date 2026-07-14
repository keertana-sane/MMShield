"""
save_sroie_images.py
----------------------
Downloads the SROIE dataset (scanned receipts) from Hugging Face and
saves 30 sample images to datasets/sroie/, matching the same process
used earlier for FUNSD and CORD.
"""

import os
from datasets import load_dataset


def main():
    print("Loading SROIE dataset (first run will download it)...")
    sroie = load_dataset("sizhkhy/SROIE")
    print(sroie)

    print("\nChecking the structure of one example...")
    example = sroie["train"][0]
    print({k: (v if k != "image" else "<PIL Image>") for k, v in example.items()})

    os.makedirs("datasets/sroie", exist_ok=True)

    num_to_save = 30
    for i in range(num_to_save):
        sroie["train"][i]["images"].save(f"datasets/sroie/sroie_{i}.png")

    print(f"\nSaved {num_to_save} SROIE receipts to datasets/sroie/")


if __name__ == "__main__":
    main()