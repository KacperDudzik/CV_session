"""Distance-weighted kNN. Features are standardized inside the pipeline."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn():
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(weights="distance")),
        ]
    )
    return model, {"clf__n_neighbors": [5, 15, 31]}
