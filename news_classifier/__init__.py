"""Utilities for training and serving a Russian news classifier."""

from .training import train_classifier
from .inference import NewsClassifier

__all__ = ["train_classifier", "NewsClassifier"]
