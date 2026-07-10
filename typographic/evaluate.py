import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.model_selection import train_test_split

from config import FEATURE_OUTPUT


MODEL_PATH = FEATURE_OUTPUT / "typographic_model.pkl"


class TypographicEvaluator:

    def __init__(self):

        self.model = joblib.load(MODEL_PATH)

        self.feature_names = None

    ########################################################

    def load_dataset(self):

        dataset = FEATURE_OUTPUT / "receipt_dataset.csv"

        df = pd.read_csv(dataset)

        self.feature_names = list(

            df.drop(

                columns=[

                    "image_name",

                    "label"

                ]

            ).columns

        )

        return df

    ########################################################

    def evaluate(self):

        df = self.load_dataset()

        X = df.drop(

            columns=[

                "image_name",

                "label"

            ]

        )

        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42,

            stratify=y

        )

        predictions = self.model.predict(

            X_test

        )

        probabilities = self.model.predict_proba(

            X_test

        )[:,1]

        print("\n====================================")
        print("Evaluation Metrics")
        print("====================================\n")

        print(

            f"Accuracy  : {accuracy_score(y_test,predictions):.4f}"

        )

        print(

            f"Precision : {precision_score(y_test,predictions):.4f}"

        )

        print(

            f"Recall    : {recall_score(y_test,predictions):.4f}"

        )

        print(

            f"F1 Score  : {f1_score(y_test,predictions):.4f}"

        )

        ########################################################
        # Confusion Matrix
        ########################################################

        cm = confusion_matrix(

            y_test,

            predictions

        )

        disp = ConfusionMatrixDisplay(

            confusion_matrix=cm,

            display_labels=[

                "Safe",

                "Attack"

            ]

        )

        disp.plot()

        plt.title("Confusion Matrix")

        plt.tight_layout()

        plt.savefig(

            FEATURE_OUTPUT /

            "confusion_matrix.png",

            dpi=300

        )

        plt.show()

        ########################################################
        # Feature Importance
        ########################################################

        importance = self.model.feature_importances_

        feature_importance = pd.DataFrame({

            "Feature": self.feature_names,

            "Importance": importance

        })

        feature_importance = feature_importance.sort_values(

            by="Importance",

            ascending=False

        )

        print("\n====================================")
        print("Top 10 Important Features")
        print("====================================\n")

        print(

            feature_importance.head(10)

        )

        top10 = feature_importance.head(10)

        plt.figure(figsize=(10,6))

        plt.barh(

            top10["Feature"][::-1],

            top10["Importance"][::-1]

        )

        plt.xlabel("Importance Score")

        plt.ylabel("Feature")

        plt.title("Top 10 Random Forest Feature Importances")

        plt.tight_layout()

        plt.savefig(

            FEATURE_OUTPUT /

            "feature_importance.png",

            dpi=300

        )

        plt.show()

        ########################################################
        # ROC Curve
        ########################################################

        fpr, tpr, thresholds = roc_curve(

            y_test,

            probabilities

        )

        roc_auc = auc(

            fpr,

            tpr

        )

        plt.figure(figsize=(6,6))

        plt.plot(

            fpr,

            tpr,

            linewidth=2,

            label=f"AUC = {roc_auc:.3f}"

        )

        plt.plot(

            [0,1],

            [0,1],

            linestyle="--"

        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve")

        plt.legend(

            loc="lower right"

        )

        plt.tight_layout()

        plt.savefig(

            FEATURE_OUTPUT /

            "roc_curve.png",

            dpi=300

        )

        plt.show()

        print("\n====================================")
        print("ROC AUC Score")
        print("====================================")

        print(

            f"AUC : {roc_auc:.4f}"

        )

        print("\nSaved Files")

        print(

            FEATURE_OUTPUT /

            "confusion_matrix.png"

        )

        print(

            FEATURE_OUTPUT /

            "feature_importance.png"

        )

        print(

            FEATURE_OUTPUT /

            "roc_curve.png"

        )


############################################################


if __name__ == "__main__":

    evaluator = TypographicEvaluator()

    evaluator.evaluate()