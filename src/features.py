"""Resampling (SMOTEENN) and feature selection / dimensionality reduction."""
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectFpr, f_classif
from sklearn.linear_model import LogisticRegression
from .config import SEED


def get_resampler():
    # k_neighbors=3: outcomes are rare, so CV folds may hold very few positives
    return SMOTEENN(smote=SMOTE(k_neighbors=3, random_state=SEED), random_state=SEED)


def get_selector(name: str):
    """Returns (transformer, param_grid_for_that_step). Step is named 'select' in the pipeline."""
    if name == "none":
        return "passthrough", {}
    if name == "lasso":
        est = LogisticRegression(penalty="l1", solver="liblinear", C=1.0,
                                 class_weight="balanced", random_state=SEED)
        return SelectFromModel(est), {"select__estimator__C": [0.1, 0.5, 1.0]}
    if name == "pca":
        return PCA(random_state=SEED), {"select__n_components": [0.90, 0.95]}
    if name == "gus":   # generic univariate selection: top-k by ANOVA F-score
        return SelectKBest(f_classif), {"select__k": [10, 20]}
    if name == "fpr":   # false positive rate test
        return SelectFpr(f_classif), {"select__alpha": [0.01, 0.05]}
    raise ValueError(f"Unknown selector: {name}")
