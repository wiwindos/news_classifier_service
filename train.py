"""Command line entry-point for training the news classifier."""
from __future__ import annotations

import argparse
from pathlib import Path

from news_classifier import train_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Russian news classifier")
    parser.add_argument("data_path", type=Path, help="Path to CSV file with training data")
    parser.add_argument(
        "--text-column",
        default="title",
        help="Name of the column that contains news texts",
    )
    parser.add_argument(
        "--target-column",
        default="topic",
        help="Name of the column that contains category labels",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where training artifacts will be saved",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=15000,
        help="Maximum number of features for the TF-IDF vectorizer",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Share of the dataset reserved for validation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for dataset splitting",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Number of boosting iterations for CatBoost",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for CatBoost",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    result = train_classifier(
        data_path=args.data_path,
        text_column=args.text_column,
        target_column=args.target_column,
        output_dir=args.output_dir,
        max_features=args.max_features,
        test_size=args.test_size,
        random_state=args.random_state,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )

    print("Training finished successfully!")
    print(f"Macro F1-score: {result.macro_f1:.4f}")
    print(f"Model saved to: {result.model_path}")
    print(f"Vectorizer saved to: {result.vectorizer_path}")
    print(f"Label encoder saved to: {result.label_encoder_path}")
    print(f"Report saved to: {result.report_path}")


if __name__ == "__main__":
    main()
