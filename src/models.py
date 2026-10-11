"""Classifier definitions and (deliberately small) hyperparameter grids.
Pipeline step name for the classifier is 'clf'."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from .config import SEED

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_model(name: str):
    """Returns (estimator, param_grid)."""
    if name == "LR":
        return (LogisticRegression(max_iter=2000, random_state=SEED),
                {"clf__C": [0.1, 1.0, 10.0]})
    if name == "RF":
        return (RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
                {"clf__max_depth": [5, None]})
    if name == "SVM":
        return (SVC(probability=True, random_state=SEED),
                {"clf__C": [1.0, 10.0], "clf__kernel": ["rbf", "linear"]})
    if name == "MLP":
        return (MLPClassifier(max_iter=500, random_state=SEED),
                {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-3, 1e-2]})
    if name == "LGBM":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed (pip install lightgbm)")
        return (LGBMClassifier(n_estimators=150, num_leaves=15, random_state=SEED,
                               n_jobs=1, verbose=-1),
                {"clf__learning_rate": [0.05, 0.1]})
    raise ValueError(f"Unknown model: {name}")
