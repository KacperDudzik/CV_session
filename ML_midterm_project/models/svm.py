"""Support Vector Machine with linear and RBF kernels."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(seed):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=SVC(class_weight="balanced", random_state=seed),
                    method="sigmoid",
                    cv=2,
                ),
            ),
        ]
    )
    return model, {
        "clf__estimator__kernel": ["linear", "rbf"],
        "clf__estimator__C": [0.1, 1.0],
    }
