"""Training utilities for the news classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .text_processing import TextPreprocess


DEFAULT_POSITIVE_LABELS: Tuple[str, ...] = (
    "экономика",
    "economics",
    "economy",
)


@dataclass
class TrainingResult:
    """Information about a completed training run."""

    model_path: Path
    vectorizer_path: Path
    label_encoder_path: Path
    report_path: Path
    macro_f1: float


def _validate_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def _prepare_texts(frame: pd.DataFrame, column: str) -> pd.Series:
    processor = TextPreprocess()
    return frame[column].astype(str).apply(processor.process_text)


def _encode_targets(frame: pd.DataFrame, column: str) -> tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(frame[column])
    return encoded, encoder


def _binarize_targets(
    frame: pd.DataFrame,
    column: str,
    positive_labels: Iterable[str],
    positive_class: str,
    negative_class: str,
) -> tuple[pd.Series, Set[str]]:
    """Collapse raw labels into binary targets."""

    positive_synonyms = {
        label.strip().casefold()
        for label in positive_labels
        if isinstance(label, str) and label.strip()
    }
    positive_synonyms.add(positive_class.strip().casefold())
    if not positive_synonyms:
        raise ValueError("At least one positive label synonym must be provided.")

    original = frame[column].fillna("").astype(str)
    normalized = original.str.strip().str.casefold()
    matches_positive = normalized.isin(positive_synonyms)

    if matches_positive.all():
        raise ValueError(
            "All samples were mapped to the positive class. Add non-economic examples."
        )
    if (~matches_positive).all():
        raise ValueError(
            "None of the targets match the configured positive synonyms."
        )

    mapped = np.where(matches_positive, positive_class, negative_class)
    return pd.Series(mapped, index=frame.index), positive_synonyms


def train_classifier(
    data_path: Path,
    text_column: str,
    target_column: str,
    output_dir: Path,
    max_features: int = 15_000,
    test_size: float = 0.2,
    random_state: int = 42,
    iterations: int = 500,
    learning_rate: float = 0.1,
    positive_labels: Iterable[str] | None = None,
    positive_class: str = "Экономика",
    negative_class: str = "Не экономика",
) -> TrainingResult:
    """Train the classifier and persist all required artifacts."""

    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(data_path)
    _validate_columns(frame, [text_column, target_column])

    label_candidates = positive_labels or DEFAULT_POSITIVE_LABELS
    binarized_targets, positive_synonyms = _binarize_targets(
        frame,
        target_column,
        positive_labels=label_candidates,
        positive_class=positive_class,
        negative_class=negative_class,
    )
    frame = frame.copy()
    frame[target_column] = binarized_targets

    texts = _prepare_texts(frame, text_column)
    targets, encoder = _encode_targets(frame, target_column)

    X_train, X_valid, y_train, y_valid = train_test_split(
        texts,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_valid_vec = vectorizer.transform(X_valid)

    train_pool = Pool(X_train_vec, y_train)
    valid_pool = Pool(X_valid_vec, y_valid)

    classifier = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        loss_function="Logloss",
        eval_metric="F1",
        auto_class_weights="Balanced",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    classifier.fit(train_pool, eval_set=valid_pool, verbose=False)

    predictions = classifier.predict(valid_pool)
    macro_f1 = f1_score(y_valid, predictions, average="macro")

    model_path = output_dir / "model.cbm"
    vectorizer_path = output_dir / "vectorizer.pkl"
    label_encoder_path = output_dir / "label_encoder.pkl"
    report_path = output_dir / "classification_report.json"

    classifier.save_model(model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(encoder, label_encoder_path)

    report: Dict[str, Dict[str, float]] = classification_report(
        y_valid,
        predictions,
        labels=list(range(len(encoder.classes_))),
        target_names=list(encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    class_indices, counts = np.unique(targets, return_counts=True)
    class_distribution = {
        encoder.inverse_transform([index])[0]: int(count)
        for index, count in zip(class_indices, counts)
    }
    metadata = {
        "text_column": text_column,
        "target_column": target_column,
        "data_path": str(data_path.resolve()),
        "macro_f1": macro_f1,
        "task_type": "binary",
        "positive_class": positive_class,
        "negative_class": negative_class,
        "positive_label_synonyms": sorted(positive_synonyms),
        "class_distribution": class_distribution,
        "label_mapping": {int(i): label for i, label in enumerate(encoder.classes_)},
        "report": report,
    }

    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    return TrainingResult(
        model_path=model_path,
        vectorizer_path=vectorizer_path,
        label_encoder_path=label_encoder_path,
        report_path=report_path,
        macro_f1=macro_f1,
    )
