from pathlib import Path
import pandas as pd

from config import TRAIN_IMAGES
from config import ATTACK_OUTPUT
from config import FEATURE_OUTPUT

from ocr import OCRExtractor
from typography import TypographyAnalyzer
from semantic import SemanticAnalyzer


class DatasetBuilder:

    def __init__(self):

        print("Loading modules...\n")

        self.ocr = OCRExtractor()
        self.typography = TypographyAnalyzer()
        self.semantic = SemanticAnalyzer()

        self.rows = []

    # ---------------------------------------------------
    # Process one image
    # ---------------------------------------------------

    def process_image(self, image_path, label):

        try:

            regions = self.ocr.extract_text_regions(image_path)

            for region in regions:

                features = self.typography.extract_features(region)

                semantic_features = self.semantic.extract_features(
                    region["text"]
                )

                features.update(semantic_features)

                features["label"] = label

                features["image_name"] = image_path.name

                self.rows.append(features)

        except Exception as e:

            print(f"Skipping {image_path.name}")

            print(e)

    # ---------------------------------------------------
    # Build Dataset
    # ---------------------------------------------------

    def build_dataset(

        self,

        max_clean=10,

        max_attack=None

    ):

        print("\nProcessing CLEAN receipts...\n")

        clean_images = sorted(

            TRAIN_IMAGES.glob("*.jpg")

        )

        if max_clean is not None:

            clean_images = clean_images[:max_clean]

        print(f"Clean images : {len(clean_images)}")

        for image in clean_images:

            self.process_image(

                image,

                label=0

            )

        print("\nProcessing ATTACK receipts...\n")

        attack_images = sorted(

            ATTACK_OUTPUT.glob("*.jpg")

        )

        if max_attack is not None:

            attack_images = attack_images[:max_attack]

        print(f"Attack images : {len(attack_images)}")

        for image in attack_images:

            self.process_image(

                image,

                label=1

            )

        df = pd.DataFrame(self.rows)

        output_file = (

            FEATURE_OUTPUT /

            "typographic_dataset.csv"

        )

        df.to_csv(

            output_file,

            index=False

        )

        print("\n==============================")

        print("Dataset Created Successfully!")

        print("==============================")

        print(f"\nSaved at:\n{output_file}")

        print(f"\nTotal Samples : {len(df)}")

        print("\nLabel Distribution")

        print(df["label"].value_counts())

        return df


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":

    builder = DatasetBuilder()

    builder.build_dataset(

        max_clean=10,

        max_attack=None

    )