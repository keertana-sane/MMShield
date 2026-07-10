import joblib
import pandas as pd

from config import FEATURE_OUTPUT

from ocr import OCRExtractor
from typography import TypographyAnalyzer
from semantic import SemanticAnalyzer
from receipt_features import ReceiptFeatureBuilder


MODEL_PATH = FEATURE_OUTPUT / "typographic_model.pkl"


class TypographicPredictor:

    def __init__(self):

        print("Loading model...\n")

        self.model = joblib.load(MODEL_PATH)

        self.ocr = OCRExtractor()

        self.typography = TypographyAnalyzer()

        self.semantic = SemanticAnalyzer()

        self.receipt_builder = ReceiptFeatureBuilder()

    ###################################################

    def extract_receipt_features(self, image_path):

        regions = self.ocr.extract_text_regions(image_path)

        region_features = []

        for region in regions:

            features = self.typography.extract_features(region)

            semantic = self.semantic.extract_features(
                region["text"]
            )

            features.update(semantic)

            region_features.append(features)

        receipt = self.receipt_builder.aggregate(
            region_features
        )

        return receipt

    ###################################################

    def predict(self, image_path):

        receipt = self.extract_receipt_features(
            image_path
        )

        X = pd.DataFrame([receipt])

        prediction = self.model.predict(X)[0]

        probability = self.model.predict_proba(X)[0][1]

        print("\n==============================")

        if prediction == 1:

            print("Prediction : ATTACK")

        else:

            print("Prediction : SAFE")

        print(f"Attack Probability : {probability:.3f}")

        print("==============================\n")


##########################################################

if __name__ == "__main__":

    TEST_IMAGE = input(

        "\nEnter receipt image path : "

    )

    predictor = TypographicPredictor()

    predictor.predict(TEST_IMAGE)