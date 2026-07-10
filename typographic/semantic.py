import re


class SemanticAnalyzer:
    """
    Performs rule-based semantic analysis
    on extracted OCR text.
    """

    def __init__(self):

        self.financial_keywords = {

            "invoice",
            "receipt",
            "total",
            "amount",
            "gst",
            "tax",
            "vendor",
            "customer",
            "cash",
            "bill",
            "qty",
            "price",
            "discount",
            "payment",
            "balance",
            "date",
            "change",
            "rm"

        }

        self.attack_keywords = {

            "ignore",
            "instruction",
            "instructions",
            "system",
            "prompt",
            "assistant",
            "secret",
            "password",
            "confidential",
            "execute",
            "reveal",
            "delete",
            "bypass",
            "override"

        }

    def extract_features(self, text):

        text_lower = text.lower()

        words = re.findall(r"\b\w+\b", text_lower)

        financial_hits = sum(
            word in self.financial_keywords
            for word in words
        )

        attack_hits = sum(
            word in self.attack_keywords
            for word in words
        )

        return {

            "financial_keyword_score": financial_hits,

            "attack_keyword_score": attack_hits

        }
