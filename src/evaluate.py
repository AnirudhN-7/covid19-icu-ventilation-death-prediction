"""Metrics, ROC/PR curves, bootstrapping, paired t-tests, box plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             roc_curve, precision_recall_curve)
from .config import SEED


def compute_metrics(y_true, y_prob, threshold=0.5):
    return {"AUROC": roc_auc_score(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
            "ACC": accuracy_score(y_true, (np.asarray(y_prob) >= threshold).astype(int))}


def plot_roc_pr(curves: dict, task: str, out_path):
    """curves: {combo_name: (y_true, y_prob)} -> one figure with ROC (left) and PR (right)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, p):.3f}")
        ax[1].plot(rc, pr, label=f"{name} AUC={average_precision_score(y, p):.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="gray")
    ax[0].set(title=f"{task.upper()} ROC", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax[1].set(title=f"{task.upper()} PRC", xlabel="Recall", ylabel="Precision")
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def bootstrap_indices(n, n_boot, y, seed=SEED):
    """Same resampled index sets are reused for every model -> enables PAIRED tests.
    Resamples that contain a single class are redrawn."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); out = []
    while len(out) < n_boot:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            out.append(idx)
    return out


def bootstrap_metrics(y_true, y_prob, idx_sets):
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    rows = [compute_metrics(y_true[i], y_prob[i]) for i in idx_sets]
    return pd.DataFrame(rows)


def ci95(values):
    return np.percentile(values, [2.5, 97.5])


def paired_ttests(boot: dict, pairs, metrics=("AUROC", "AUPRC")):
    """boot: {combo: DataFrame}. pairs: [(better_candidate, baseline)].
    One-sided paired t-test: H1 = first model > second."""
    rows = []
    for a, b in pairs:
        if a not in boot or b not in boot:
            continue
        for m in metrics:
            t, p = stats.ttest_rel(boot[a][m], boot[b][m], alternative="greater")
            rows.append({"model_A": a, "model_B": b, "metric": m,
                         "mean_A": boot[a][m].mean(), "mean_B": boot[b][m].mean(), "p_value": p})
    return pd.DataFrame(rows)


def boxplot_bootstrap(boot: dict, task: str, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, m in zip(ax, ["AUROC", "AUPRC"]):
        a.boxplot([boot[k][m] for k in boot], notch=False)
        a.set_xticks(range(1, len(boot) + 1)); a.set_xticklabels(list(boot))
        a.set(title=f"{task.upper()} {m} ({len(next(iter(boot.values())))} bootstraps)", ylabel=m)
        a.tick_params(axis="x", rotation=25); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
