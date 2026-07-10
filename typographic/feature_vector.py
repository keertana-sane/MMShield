import pandas as pd
from pathlib import Path

from config import FEATURE_OUTPUT


class FeatureVectorBuilder:

    def __init__(self):
        self.rows = []

    def add_region(self, features):
        self.rows.append(features)

    def save_csv(self, filename="typography_features.csv"):

        df = pd.DataFrame(self.rows)

        output_path = FEATURE_OUTPUT / filename

        df.to_csv(output_path, index=False)

        print(f"\nFeature vector saved to:\n{output_path}")

        return df
