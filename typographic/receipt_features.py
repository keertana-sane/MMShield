import numpy as np


class ReceiptFeatureBuilder:

    """
    Converts multiple OCR region features
    into ONE feature vector per receipt.
    """

    def __init__(self):
        pass

    def aggregate(self, region_features):

        if len(region_features) == 0:
            return None

        receipt = {}

        receipt["num_regions"] = len(region_features)

        numeric_features = [

            "confidence",

            "width",

            "height",

            "area",

            "aspect_ratio",

            "character_count",

            "word_count",

            "avg_word_length",

            "alphabet_ratio",

            "numeric_ratio",

            "uppercase_ratio",

            "whitespace_ratio",

            "special_character_ratio",

            "character_density",

            "estimated_font_size",

            "financial_keyword_score",

            "attack_keyword_score"

        ]

        for feature in numeric_features:

            values = [

                row[feature]

                for row in region_features

            ]

            receipt[f"{feature}_mean"] = float(

                np.mean(values)

            )

            receipt[f"{feature}_max"] = float(

                np.max(values)

            )

            receipt[f"{feature}_min"] = float(

                np.min(values)

            )

            receipt[f"{feature}_std"] = float(

                np.std(values)

            )

        return receipt