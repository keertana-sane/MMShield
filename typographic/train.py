import joblib
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import FEATURE_OUTPUT


MODEL_OUTPUT = FEATURE_OUTPUT / "typographic_model.pkl"


class TypographicTrainer:

    def __init__(self):

        self.model = RandomForestClassifier(

            n_estimators=200,

            random_state=42,

            n_jobs=-1

        )

    #####################################################

    def load_dataset(self):

        dataset_path = (

            FEATURE_OUTPUT /

            "receipt_dataset.csv"

        )

        print(f"\nLoading dataset:\n{dataset_path}\n")

        df = pd.read_csv(dataset_path)

        return df

    #####################################################

    def prepare_data(self, df):

        X = df.drop(

            columns=[

                "image_name",

                "label"

            ]

        )

        y = df["label"]

        return train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

    #####################################################

    def train(self):

        df = self.load_dataset()

        X_train, X_test, y_train, y_test = (

            self.prepare_data(df)

        )

        print("Training Random Forest...\n")

        self.model.fit(

            X_train,

            y_train

        )

        predictions = self.model.predict(

            X_test

        )

        accuracy = accuracy_score(

            y_test,

            predictions

        )

        print("=" * 40)

        print(f"Accuracy : {accuracy:.4f}")

        print("=" * 40)

        joblib.dump(

            self.model,

            MODEL_OUTPUT

        )

        print(f"\nModel Saved:\n{MODEL_OUTPUT}")


##########################################################

if __name__ == "__main__":

    trainer = TypographicTrainer()

    trainer.train()