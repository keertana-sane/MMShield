"""
Semantic Feature Extraction Module

Performs lightweight, rule-based keyword scoring on OCR region text to
capture two signals used by the classifier:

    - financial_keyword_score: how much a region "looks like" normal
      receipt/invoice content (line items, totals, tax, etc.)
    - attack_keyword_score: how much a region contains vocabulary
      associated with prompt-injection / instruction-override attempts

This is intentionally a simple bag-of-keywords model, not NLP/embedding
based — documented here for research transparency, since a reviewer may
ask why this isn't a more sophisticated semantic model. It is fast,
fully deterministic, and easy to audit, which matters for a security
component whose behavior needs to be explainable.

Known heuristic limitations:
    - Matching is exact-word only (via regex word boundaries) after
      lowercasing; it does not stem/lemmatize, so e.g. "ignored" will
      NOT match the keyword "ignore".
    - Keyword sets are hand-curated and English-only. Multilingual
      receipts or obfuscated attack text (e.g. leetspeak, unicode
      homoglyphs) will not be detected by this module.
    - Scores are raw counts, not normalized by text length here;
      length-normalization (if desired) happens downstream in
      receipt_features.py via the *_mean/*_max/*_min/*_std aggregation.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

__all__ = ["SemanticAnalyzer"]

FeatureDict = dict[str, int]

_WORD_PATTERN = re.compile(r"\b\w+\b")

# Keyword sets are module-level constants (shared, immutable across all
# instances) rather than re-built in every __init__ call.
_FINANCIAL_KEYWORDS: frozenset[str] = frozenset({
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
    "rm",
})

_ATTACK_KEYWORDS: frozenset[str] = frozenset({
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
    "override",
})


class SemanticAnalyzer:
    """
    Performs rule-based keyword scoring on OCR region text.
    """

    def __init__(
        self,
        financial_keywords: Optional[frozenset[str]] = None,
        attack_keywords: Optional[frozenset[str]] = None,
    ) -> None:
        """
        Args:
            financial_keywords: Optional override for the financial
                keyword set. Defaults to the built-in curated set.
            attack_keywords: Optional override for the attack keyword
                set. Defaults to the built-in curated set.
        """

        self.financial_keywords = (
            financial_keywords
            if financial_keywords is not None
            else _FINANCIAL_KEYWORDS
        )

        self.attack_keywords = (
            attack_keywords
            if attack_keywords is not None
            else _ATTACK_KEYWORDS
        )

    ####################################################

    def extract_features(self, text: Optional[str]) -> FeatureDict:
        """
        Scores a single region's text against the financial and
        attack keyword sets.

        Args:
            text: The recognized text for a region. None is treated
                as an empty string.

        Returns:
            Dict with financial_keyword_score and
            attack_keyword_score (raw integer counts).
        """

        text = text or ""

        words = _WORD_PATTERN.findall(text.lower())

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
            "attack_keyword_score": attack_hits,
        }