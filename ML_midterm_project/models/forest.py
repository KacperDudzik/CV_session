"""Random forest. Trees do not use feature scaling."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def build_forest(seed):
    model = Pipeline(
        [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        ]
    )
    return model, {
        "clf__max_depth": [16, None],
        "clf__min_samples_leaf": [1, 5],
    }
