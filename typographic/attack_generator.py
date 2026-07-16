"""
Typographic Attack Generator

Generates synthetic typographic prompt-injection attacks on receipt
images by overlaying adversarial text onto empty regions (or, if none
are found, the bottom margin) of clean receipt images. Used to build
the "attack" class of the MMShield typographic detection dataset.

Multi-dataset + split support:
    Attacks can now be generated per dataset AND per split
    (train/test), so the official train/test separation established
    in config.DATASETS is preserved all the way through to attack
    images. Output is written to
    ATTACK_OUTPUT / <dataset> / <split>, and metadata to
    <dataset>_<split>_attack_metadata.csv, so train and test attack
    sets never share a folder or overwrite each other's metadata.
"""

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from ocr import OCRExtractor
from prompts import PROMPTS
from config import get_dataset_images
from config import ATTACK_OUTPUT
from config import METADATA_OUTPUT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

__all__ = ["TypographicAttackGenerator"]

_METADATA_FIELDNAMES: tuple[str, ...] = (
    "image",
    "prompt",
    "font_size",
    "color",
    "rotation",
    "x",
    "y",
)


class TypographicAttackGenerator:
    """
    Generates a dataset of typographic prompt-injection attack images
    by overlaying adversarial prompt text onto clean receipt images,
    for a specific dataset + split.
    """

    def __init__(self, dataset: str = "sroie", split: str = "train") -> None:
        self.dataset = dataset

        self.split = split

        self.input_folder = get_dataset_images(dataset, split)

        self.output_folder = ATTACK_OUTPUT / dataset / split
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.metadata_file = (
        METADATA_OUTPUT /
        f"{dataset}_{split}_attack_metadata.csv"
        )

        self.ocr = OCRExtractor()

        self.colors: list[tuple[int, int, int]] = [

            (0, 0, 0),

            (40, 40, 40),

            (80, 80, 80),

            (120, 120, 120),

            (170, 170, 170)

        ]

    ####################################################

    def choose_prompt(self) -> str:
        """Randomly selects one adversarial prompt string from PROMPTS."""

        return random.choice(PROMPTS)

    ####################################################

    def choose_font_size(self) -> int:
        """Randomly selects a font size in the range [8, 18] pixels."""

        return random.randint(8, 18)

    ####################################################

    def choose_color(self) -> tuple[int, int, int]:
        """Randomly selects an RGB text color from the predefined palette."""

        return random.choice(self.colors)

    ####################################################

    def choose_rotation(self) -> int:
        """Randomly selects a rotation angle in degrees."""

        return random.choice(

            [0, 10, 15, 20, 30]

        )

    ####################################################

    def load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        Attempts to load a scalable TrueType font at the requested
        size. Tries common cross-platform font locations first, then
        falls back to matplotlib's bundled DejaVu Sans font (always
        available since matplotlib is already a project dependency).
        Only if every TrueType option fails does it fall back to
        PIL's fixed-size bitmap default font — in which case `size`
        is NOT respected, and a warning is logged.

        Args:
            size: Desired font size in pixels.

        Returns:
            A loaded PIL ImageFont instance.
        """

        candidate_paths = [

            "Arial.ttf",

            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",

            "/Library/Fonts/Arial.ttf",

            "C:\\Windows\\Fonts\\arial.ttf",

        ]

        for path in candidate_paths:

            try:

                return ImageFont.truetype(path, size)

            except (OSError, IOError):

                continue

        try:

            from matplotlib import font_manager

            dejavu_path = font_manager.findfont("DejaVu Sans")

            return ImageFont.truetype(dejavu_path, size)

        except Exception:

            pass

        logger.warning(

            "No TrueType font could be loaded. Falling back to PIL's "
            "fixed-size bitmap default font. Requested font_size=%d "
            "will NOT be applied.",
            size

        )

        return ImageFont.load_default()

    ####################################################

    def find_empty_regions(self, image_path: Path) -> list[dict[str, int]]:
        """
        Uses OCR bounding boxes to locate vertical gaps between text
        regions where an adversarial prompt could plausibly be
        inserted without overlapping existing receipt content.

        Args:
            image_path: Path to the receipt image to analyze.

        Returns:
            A list of candidate regions, each a dict with x, y,
            width, and height keys.
        """

        regions = self.ocr.extract_text_regions(

            image_path

        )

        if len(regions) < 2:

            return []

        regions = sorted(

            regions,

            key=lambda r:

            min(

                p[1]

                for p in r["polygon"]

            )

        )

        candidates = []

        for i in range(

            len(regions) - 1

        ):

            current = regions[i]

            nxt = regions[i + 1]

            current_bottom = max(

                p[1]

                for p in current["polygon"]

            )

            next_top = min(

                p[1]

                for p in nxt["polygon"]

            )

            gap = next_top - current_bottom

            if gap > 30:

                candidates.append(

                    {

                        "x": 30,

                        "y": int(current_bottom) + 5,

                        "width": 400,

                        "height": int(gap)

                    }

                )

        return candidates

    ####################################################

    def choose_region(
        self,
        candidates: list[dict[str, int]],
        image: Image.Image
    ) -> dict[str, int]:
        """
        Selects a region to place the adversarial prompt in. If no
        empty-region candidates were found, falls back to a fixed
        band near the bottom of the image.

        Args:
            candidates: Candidate empty regions from find_empty_regions.
            image: The receipt image being attacked.

        Returns:
            A dict with x, y, width, and height keys.
        """

        width, height = image.size

        if len(candidates) == 0:

            return {

                "x": 30,

                "y": max(height - 80, 0),

                "width": max(width - 60, 1),

                "height": 40

            }

        return random.choice(candidates)

    ####################################################

    def draw_prompt(
        self,
        image: Image.Image,
        prompt: str,
        region: dict[str, int],
        font_size: int,
        color: tuple[int, int, int],
        rotation: int
    ) -> Image.Image:
        """
        Renders the adversarial prompt text onto a transparent layer
        sized to the text itself, optionally rotates that layer in
        place, then composites it onto the receipt image at the
        target region's anchor point.

        Args:
            image: The base receipt image (RGB).
            prompt: The adversarial prompt text to render.
            region: Target region dict with at least x and y keys.
            font_size: Font size in pixels.
            color: RGB color tuple for the text.
            rotation: Rotation angle in degrees.

        Returns:
            A new RGB image with the prompt composited onto it.

        Raises:
            ValueError: If prompt is empty (nothing to render).
        """

        if not prompt:

            raise ValueError("draw_prompt received an empty prompt string.")

        font = self.load_font(font_size)

        measuring_draw = ImageDraw.Draw(

            Image.new("RGBA", (1, 1))

        )

        text_bbox = measuring_draw.textbbox(

            (0, 0),

            prompt,

            font=font

        )

        text_width = max(text_bbox[2] - text_bbox[0], 1)

        text_height = max(text_bbox[3] - text_bbox[1], 1)

        padding = 4

        text_layer = Image.new(

            "RGBA",

            (text_width + padding * 2, text_height + padding * 2),

            (255, 255, 255, 0)

        )

        text_draw = ImageDraw.Draw(text_layer)

        text_draw.text(

            (padding - text_bbox[0], padding - text_bbox[1]),

            prompt,

            font=font,

            fill=color + (255,)

        )

        if rotation != 0:

            text_layer = text_layer.rotate(

                rotation,

                expand=True

            )

        overlay = Image.new(

            "RGBA",

            image.size,

            (255, 255, 255, 0)

        )

        overlay.paste(

            text_layer,

            (region["x"], region["y"]),

            text_layer

        )

        image = Image.alpha_composite(

            image.convert("RGBA"),

            overlay

        )

        return image.convert("RGB")

    ####################################################

    def generate_attack(
        self,
        image: Image.Image,
        image_name: str,
        candidates: list[dict[str, int]],
        attack_number: int,
        metadata: list[dict[str, Any]]
    ) -> None:
        """
        Generates a single attacked variant of a receipt image and
        appends its metadata to the shared metadata list.

        Args:
            image: The clean base receipt image (RGB).
            image_name: Stem of the source image filename.
            candidates: Candidate empty regions for this image.
            attack_number: Index of this attack variant for the image.
            metadata: Shared list that this attack's metadata row is
                appended to.
        """

        attacked = image.copy()

        region = self.choose_region(

            candidates,

            attacked

        )

        prompt = self.choose_prompt()

        font_size = self.choose_font_size()

        color = self.choose_color()

        rotation = self.choose_rotation()

        attacked = self.draw_prompt(

            attacked,

            prompt,

            region,

            font_size,

            color,

            rotation

        )

        output_name = (

            f"{image_name}_attack_{attack_number}.jpg"

        )

        output_path = (

            self.output_folder /

            output_name

        )

        attacked.save(output_path)

        metadata.append(

            {

                "image": output_name,

                "prompt": prompt,

                "font_size": font_size,

                "color": color,

                "rotation": rotation,

                "x": region["x"],

                "y": region["y"]

            }

        )

        logger.info("Saved: %s", output_name)

    ####################################################

    def generate_dataset(
        self,
        num_attacks: int = 5,
        max_images: Optional[int] = 20,
        random_seed: int = 42
    ) -> None:
        """
        Generates attacked variants for every clean receipt image
        under this generator's input_folder (dataset + split) and
        writes a metadata CSV summarizing every generated attack.

        Args:
            num_attacks: Number of attack variants to generate per
                source image.
            max_images: Maximum number of source images to process.
                If None, all available images are used.
            random_seed: Seed for Python's random module, controlling
                prompt/font/color/rotation/region selection, so the
                generated dataset is reproducible across runs.
        """

        random.seed(random_seed)

        images = []

        for extension in ("*.jpg", "*.jpeg", "*.png"):
          images.extend(self.input_folder.glob(extension))

        images = sorted(images)

        if not images:

            logger.warning(
                "No source images found under %s. Nothing to generate.",
                self.input_folder,
            )

        if max_images is not None:

            images = images[:max_images]

        logger.info(

            "Generating attacks for %d receipts (%s / %s)...",
            len(images),
            self.dataset,
            self.split,

        )

        logger.info("Random seed: %d", random_seed)

        metadata: list[dict[str, Any]] = []

        skipped: list[str] = []

        for image_path in images:

            logger.info("Processing %s", image_path.name)

            try:

                image = Image.open(

                    image_path

                ).convert("RGB")

                image_name = image_path.stem

                candidates = self.find_empty_regions(

                    image_path

                )

                for attack in range(

                    1,

                    num_attacks + 1

                ):

                    self.generate_attack(

                        image,

                        image_name,

                        candidates,

                        attack,

                        metadata

                    )

            except Exception as exc:

                logger.error(

                    "Skipping %s due to error: %s",
                    image_path.name,
                    exc

                )

                skipped.append(image_path.name)

                continue

        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)

        with open(

            self.metadata_file,

            "w",

            newline=""

        ) as f:

            writer = csv.DictWriter(

                f,

                fieldnames=_METADATA_FIELDNAMES

            )

            writer.writeheader()

            writer.writerows(metadata)

        logger.info("Attack generation completed.")

        logger.info("Generated %d attack images.", len(metadata))

        if skipped:

            logger.warning(

                "Skipped %d image(s) due to errors: %s",
                len(skipped),
                ", ".join(skipped)

            )


##########################################################


def main() -> None:
    """Parses command-line arguments and runs attack dataset generation."""

    parser = argparse.ArgumentParser(

        description="Generate typographic prompt-injection attack images."

    )

    parser.add_argument(

        "--num-attacks",
        type=int,
        default=1,
        help="Number of attack variants to generate per source image (default: 1)."

    )

    parser.add_argument(

        "--max-images",
        type=int,
        default=None,
        help="Maximum number of source images to process (default: all)."

    )

    parser.add_argument(

        "--seed",
        
        type=int,
        default=42,
        help="Random seed for reproducible attack generation (default: 42)."

    )

    parser.add_argument(
       "--dataset",
       type=str,
       default="sroie",
       choices=["sroie", "cord", "funsd"],
       help="Dataset to generate attacks for."
    )

    parser.add_argument(
       "--split",
       type=str,
       default="train",
       choices=["train", "test"],
       help="Which split's clean images to attack (default: train)."
    )

    args = parser.parse_args()

    generator = TypographicAttackGenerator(
    dataset=args.dataset,
    split=args.split,
)

    generator.generate_dataset(

        num_attacks=args.num_attacks,

        max_images=args.max_images,

        random_seed=args.seed

    )


if __name__ == "__main__":

    main()