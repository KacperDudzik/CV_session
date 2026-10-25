"""Classical models. Existing four stay; SVM and decision tree are added."""

from .forest import build_forest
from .knn import build_knn
from .logistic import build_logistic
from .majority import build_majority
from .svm import build_svm
from .tree import build_tree


def candidates(seed):
    """(name, unfitted pipeline, parameter grid) in simple-to-complex order."""
    return [
        ("majority", *build_majority()),
        ("logistic", *build_logistic(seed)),
        ("tree", *build_tree(seed)),
        ("knn", *build_knn()),
        ("svm", *build_svm(seed)),
        ("forest", *build_forest(seed)),
    ]
