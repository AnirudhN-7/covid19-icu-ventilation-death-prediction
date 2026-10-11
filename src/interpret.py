"""Model-agnostic feature importance via permutation importance (drop in AUROC)."""
import pandas as pd
from sklearn.inspection import permutation_importance
from .config import SEED


def top_features(model, X, y, k=10, n_repeats=5):
    """Works with any fitted pipeline (including PCA/LASSO inside), since features are
    permuted on the raw input. Returns top-k features by mean AUROC drop."""
    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean,
                        "std": r.importances_std})
    return imp.sort_values("importance", ascending=False).head(k).reset_index(drop=True)
