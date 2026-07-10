import pandas as pd

from config import TRAIN_IMAGES
from config import ATTACK_OUTPUT
from config import FEATURE_OUTPUT

from ocr import OCRExtractor
from typography import TypographyAnalyzer
from semantic import SemanticAnalyzer
from receipt_features import ReceiptFeatureBuilder


class ReceiptDatasetBuilder:

    """
    Creates one feature vector per receipt.
    """

    def __init__(self):

        print("Initializing modules...\n")

        self.ocr = OCRExtractor()

        self.typography = TypographyAnalyzer()

        self.semantic = SemanticAnalyzer()

        self.receipt_builder = ReceiptFeatureBuilder()

        self.dataset = []

    ######################################################

    def process_receipt(self, image_path, label):

        print(f"Processing : {image_path.name}")

        regions = self.ocr.extract_text_regions(image_path)

        region_features = []

        for region in regions:

            typography_features = self.typography.extract_features(
                region
            )

            semantic_features = self.semantic.extract_features(
                region["text"]
            )

            typography_features.update(
                semantic_features
            )

            region_features.append(
                typography_features
            )

        receipt_features = self.receipt_builder.aggregate(
            region_features
        )

        if receipt_features is None:

            return

        receipt_features["image_name"] = image_path.name

        receipt_features["label"] = label

        self.dataset.append(
            receipt_features
        )
    
        ######################################################

    def build_dataset(

        self,

        max_clean=10,

        max_attack=None

    ):

        clean_images = sorted(

            TRAIN_IMAGES.glob("*.jpg")

        )

        if max_clean is not None:

            clean_images = clean_images[:max_clean]

        attack_images = sorted(

            ATTACK_OUTPUT.glob("*.jpg")

        )

        if max_attack is not None:

            attack_images = attack_images[:max_attack]

        print("\n==============================")
        print("Processing CLEAN receipts")
        print("==============================\n")

        for i, image in enumerate(clean_images, 1):

            print(f"[Clean {i}/{len(clean_images)}]")

            self.process_receipt(

                image,

                label=0

            )

        print("\n==============================")
        print("Processing ATTACK receipts")
        print("==============================\n")

        for i, image in enumerate(attack_images, 1):

            print(f"[Attack {i}/{len(attack_images)}]")

            self.process_receipt(

                image,

                label=1

            )

        df = pd.DataFrame(

            self.dataset

        )

        output_path = (

            FEATURE_OUTPUT /

            "receipt_dataset.csv"

        )

        df.to_csv(

            output_path,

            index=False

        )

        print("\n====================================")

        print("Receipt Dataset Created Successfully!")

        print("====================================")

        print(f"\nSaved to:\n{output_path}")

        print(f"\nTotal Receipts : {len(df)}")

        print("\nLabel Distribution")

        print(

            df["label"].value_counts()

        )

        return df


##########################################################

if __name__ == "__main__":

    builder = ReceiptDatasetBuilder()

    builder.build_dataset(

        max_clean=56,

        max_attack=56

    )