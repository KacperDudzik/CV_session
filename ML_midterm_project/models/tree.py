"""Single decision tree classifier."""

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def build_tree(seed):
    model = Pipeline(
        [
            (
                "clf",
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=seed,
                ),
            )
        ]
    )
    return model, {
        "clf__max_depth": [8, 16, None],
        "clf__min_samples_leaf": [1, 5],
    }
