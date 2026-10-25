"""Majority-class baseline. Ignores the image and predicts the training mode."""

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline


def build_majority():
    model = Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])
    return model, {}
