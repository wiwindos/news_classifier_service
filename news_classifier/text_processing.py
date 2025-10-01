"""Text preprocessing utilities for Russian news classification."""
from __future__ import annotations

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from pymorphy2 import MorphAnalyzer


def _ensure_stopwords_downloaded() -> None:
    """Download NLTK stopwords if they are missing."""
    try:
        stopwords.words("russian")
    except LookupError:
        nltk.download("stopwords")


class TextPreprocess:
    """Class for text preprocessing.

    To run full text pipeline use :meth:`process_text` method.
    """

    def __init__(self) -> None:
        _ensure_stopwords_downloaded()
        self.tokenizer: WordPunctTokenizer = WordPunctTokenizer()
        self.morph_analyzer: MorphAnalyzer = MorphAnalyzer()
        self.reg_exp: re.Pattern[str] = re.compile("[^А-Яа-яЁё]")
        self.stop_words: List[str] = stopwords.words("russian")

    @staticmethod
    def is_empty(text: str) -> bool:
        """Return ``True`` when the input string is empty."""

        return text == ""

    def get_pruned_text(self, text: str) -> str:
        """Remove any non Cyrillic symbols from the text."""

        return self.reg_exp.sub(" ", text)

    @staticmethod
    def get_replaced_form(text: str, target: str, symb: str) -> str:
        """Replace ``target`` symbols with ``symb`` in ``text``."""

        return text.replace(target, symb)

    @staticmethod
    def get_strip_form(text: str) -> str:
        """Strip surrounding whitespace from the text."""

        return text.strip()

    @staticmethod
    def get_lower_form(text: str) -> str:
        """Convert text to lowercase."""

        return text.lower()

    def get_tokenized_form(self, text: str) -> List[str]:
        """Split text into tokens."""

        return self.tokenizer.tokenize(text)

    def get_normal_form(self, text: List[str]) -> List[str]:
        """Lemmatise every token using :mod:`pymorphy2`."""

        return [self.morph_analyzer.normal_forms(word)[0] for word in text]

    def filter_words(self, text: List[str]) -> List[str]:
        """Filter stop words from token list."""

        return [word for word in text if word not in self.stop_words]

    def process_text(self, text: str) -> str:
        """Execute the full preprocessing pipeline for ``text``."""

        pruned = self.get_pruned_text(text)
        stripped = self.get_strip_form(pruned)
        replaced = self.get_replaced_form(stripped, "  ", " ")
        lowered = self.get_lower_form(replaced)
        tokenized = self.get_tokenized_form(lowered)
        normalized = self.get_normal_form(tokenized)
        filtered = self.filter_words(normalized)

        return " ".join(filtered)
