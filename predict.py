"""Command line utility for running predictions with a trained model."""
from __future__ import annotations

import argparse
from pathlib import Path

from news_classifier import NewsClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict whether the news belongs to the Economics category"
    )
    parser.add_argument(
        "artifacts_dir",
        type=Path,
        help="Directory that contains model.cbm, vectorizer.pkl and label_encoder.pkl",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="News text to classify",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Show top-k most probable classes",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    classifier = NewsClassifier(args.artifacts_dir)
    prediction = classifier.predict([args.text])[0]
    probabilities = classifier.predict_proba([args.text])[0]
    labels = classifier.labels
    probability_map = dict(zip(labels, probabilities))

    print(f"Ответ: {prediction}")
    if prediction in probability_map:
        print(f"Вероятность класса \"{prediction}\": {probability_map[prediction]:.4f}")

    paired = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)

    print("Распределение вероятностей:")
    for label, prob in paired[: args.top_k]:
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
