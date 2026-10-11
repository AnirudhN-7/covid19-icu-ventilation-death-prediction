"""Pipeline construction, hyperparameter tuning (5-fold CV), and model selection."""
import warnings
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from .config import CV_FOLDS, SEED
from .features import get_resampler, get_selector
from .models import get_model

warnings.filterwarnings("ignore")


def build_pipeline(selector_name: str, model_name: str, resample: bool = True):
    """impute -> scale -> SMOTEENN -> selector -> classifier.
    Because everything lives in an imblearn Pipeline, resampling and selection are
    fitted ONLY on training folds (no leakage), and resampling is skipped at predict time."""
    selector, sel_grid = get_selector(selector_name)
    clf, clf_grid = get_model(model_name)
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if resample:
        steps.append(("resample", get_resampler()))
    steps += [("select", selector), ("clf", clf)]
    return Pipeline(steps), {**sel_grid, **clf_grid}


def tune(X, y, selector_name, model_name, n_jobs=-1):
    pipe, grid = build_pipeline(selector_name, model_name)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs,
                      error_score=np.nan, refit=True)
    gs.fit(X, y)
    return gs


def select_best_model(X_train, y_train, X_test, y_test, selectors, models, n_jobs=-1, verbose=True):
    """Tune every (selector, model) pair with 5-fold CV on the 70% training split, score on
    the held-out 30% of Cohort 1, and keep the best by AUROC (as in the paper).
    Returns (best_estimator, leaderboard_df)."""
    rows, best, best_auc = [], None, -1
    for sel in selectors:
        for mod in models:
            try:
                gs = tune(X_train, y_train, sel, mod, n_jobs)
                auc = roc_auc_score(y_test, gs.predict_proba(X_test)[:, 1])
            except Exception as e:   # e.g. LASSO selects 0 features, too few positives
                if verbose:
                    print(f"    skip {sel}+{mod}: {type(e).__name__}")
                continue
            rows.append({"selector": sel, "model": mod, "cv_auroc": gs.best_score_,
                         "test_auroc": auc, "best_params": gs.best_params_})
            if verbose:
                print(f"    {sel:>5} + {mod:<4} cv={gs.best_score_:.3f} test={auc:.3f}")
            if auc > best_auc:
                best, best_auc = gs.best_estimator_, auc
    if best is None:
        raise RuntimeError("All model configurations failed.")
    return best, pd.DataFrame(rows).sort_values("test_auroc", ascending=False)
