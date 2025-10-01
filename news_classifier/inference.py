"""Runtime helpers for loading and using the classifier."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
from catboost import CatBoostClassifier

from .text_processing import TextPreprocess


class NewsClassifier:
    """Wrapper around the trained CatBoost model and vectorizer."""

    def __init__(self, artifacts_dir: Path) -> None:
        artifacts_dir = Path(artifacts_dir)
        self._model = CatBoostClassifier()
        self._model.load_model(artifacts_dir / "model.cbm")
        self._vectorizer = joblib.load(artifacts_dir / "vectorizer.pkl")
        self._label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")
        metadata_path = artifacts_dir / "classification_report.json"
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as fp:
                self._metadata: Dict[str, object] = json.load(fp)
        else:
            self._metadata = {}
        self._processor = TextPreprocess()

    @property
    def labels(self) -> Sequence[str]:
        """Return ordered labels known to the classifier."""

        return list(self._label_encoder.classes_)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Predict class probabilities for ``texts``."""

        processed = [self._processor.process_text(text) for text in texts]
        vectors = self._vectorizer.transform(processed)
        return self._model.predict_proba(vectors)

    def predict(self, texts: Sequence[str]) -> List[str]:
        """Predict the most likely label for each text."""

        probabilities = self.predict_proba(texts)
        indices = probabilities.argmax(axis=1)
        return [self.labels[index] for index in indices]

    def describe(self) -> Dict[str, object]:
        """Return metadata captured during training if available."""

        return self._metadata
