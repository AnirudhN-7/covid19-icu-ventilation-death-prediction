"""Runs the full pipeline: for each task x data-combination, tune + select the best
model, validate on Cohort 2, bootstrap, run t-tests, plot, and save models.

Usage:
    python main.py --synthetic --fast          # quick smoke test on fake data
    python main.py --data-dir data             # real data, reduced search
    python main.py --data-dir data --full      # full search (slow)
"""
import argparse
import json
import joblib
import pandas as pd

from src import config as C
from src.data_loader import load_dataset, get_feature_columns, summarize
from src.preprocessing import clean_features, make_splits
from src.train import select_best_model
from src.evaluate import (compute_metrics, plot_roc_pr, bootstrap_indices, bootstrap_metrics,
                          paired_ttests, boxplot_bootstrap, ci95)
from src.interpret import top_features

PAIRS = [("RadioClin", "Radiom"), ("RadioClinLab", "ClinLab"),
         ("RadioClinLab", "RadioClin"), ("RadioClinLab", "Radiom")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DATA_DIR))
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic data first")
    ap.add_argument("--full", action="store_true", help="all 5 selectors x 5 models (slow)")
    ap.add_argument("--fast", action="store_true", help="tiny search, for testing")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAPS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--tasks", nargs="+", default=C.TASKS, choices=C.TASKS, help="subset of tasks to run")
    args = ap.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir)
    if args.synthetic:
        from src.synthetic_data import write_synthetic
        write_synthetic(data_dir)

    selectors = C.SELECTORS_FULL if args.full else (C.SELECTORS_FAST if args.fast else ["none", "lasso", "gus"])
    models = C.MODELS_FULL if args.full else (C.MODELS_FAST if args.fast else ["LR", "RF", "LGBM"])

    for d in (C.RESULTS_DIR, C.MODELS_DIR):
        d.mkdir(exist_ok=True)

    df = clean_features(load_dataset(data_dir))
    summarize(df)

    summary_rows, importance_all, ttests_all = [], [], []

    for task in args.tasks:
        print(f"\n=== TASK: {task.upper()} ===")
        dev_train, dev_test, val = make_splits(df, task)
        curves, boot, preds, fitted = {}, {}, {}, {}
        idx_sets = bootstrap_indices(len(val), args.n_boot, val[task])

        for combo, groups in C.DATA_COMBOS.items():
            missing = [g for g in groups if not get_feature_columns(df, [g])]
            if missing:
                print(f"[{combo}] skipped: no '{missing[0]}' features in this dataset"); continue
            cols = get_feature_columns(df, groups)
            print(f"[{combo}] {len(cols)} features")
            model, board = select_best_model(
                dev_train[cols], dev_train[task], dev_test[cols], dev_test[task],
                selectors, models, n_jobs=args.n_jobs)
            board.to_csv(C.RESULTS_DIR / f"leaderboard_{task}_{combo}.csv", index=False)
            best = board.iloc[0]

            p_dev = model.predict_proba(dev_test[cols])[:, 1]
            p_val = model.predict_proba(val[cols])[:, 1]
            m_dev, m_val = compute_metrics(dev_test[task], p_dev), compute_metrics(val[task], p_val)
            bm = bootstrap_metrics(val[task], p_val, idx_sets)
            lo, hi = ci95(bm["AUROC"])
            print(f"  -> best: {best['selector']}+{best['model']} | Cohort1-test AUROC={m_dev['AUROC']:.3f} "
                  f"| Cohort2 AUROC={m_val['AUROC']:.3f} [{lo:.3f}-{hi:.3f}]")

            curves[combo], boot[combo] = (val[task].values, p_val), bm
            summary_rows.append({
                "task": task, "data": combo, "selector": best["selector"], "model": best["model"],
                "C1test_AUROC": m_dev["AUROC"], "C1test_AUPRC": m_dev["AUPRC"], "C1test_ACC": m_dev["ACC"],
                "C2_AUROC": m_val["AUROC"], "C2_AUROC_CI_low": lo, "C2_AUROC_CI_high": hi,
                "C2_AUPRC": m_val["AUPRC"], "C2_ACC": m_val["ACC"]})
            joblib.dump(model, C.MODELS_DIR / f"{task}_{combo}.joblib")
            fitted[combo] = (model, cols)

        # feature importance on the richest data combination that was available
        for combo in ["RadioClinLab", "ClinLab", "RadioClin", "Clin", "Radiom"]:
            if combo in fitted:
                model, cols = fitted[combo]
                imp = top_features(model, val[cols], val[task])
                imp.insert(0, "task", task); imp.insert(1, "data", combo); importance_all.append(imp)
                break

        plot_roc_pr(curves, task, C.RESULTS_DIR / f"roc_pr_{task}.png")
        boxplot_bootstrap(boot, task, C.RESULTS_DIR / f"bootstrap_box_{task}.png")
        tt = paired_ttests(boot, PAIRS); tt.insert(0, "task", task); ttests_all.append(tt)

    pd.DataFrame(summary_rows).to_csv(C.RESULTS_DIR / ("summary_results.csv" if len(args.tasks) == len(C.TASKS) else "summary_results_" + "_".join(args.tasks) + ".csv"), index=False)
    if ttests_all:
        pd.concat(ttests_all).to_csv(C.RESULTS_DIR / "paired_ttests.csv", index=False)
    if importance_all:
        pd.concat(importance_all).to_csv(C.RESULTS_DIR / "feature_importance.csv", index=False)
    print("\nDone. See results/ and models/.")
    print(pd.DataFrame(summary_rows)[["task", "data", "selector", "model", "C2_AUROC", "C2_AUPRC"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
