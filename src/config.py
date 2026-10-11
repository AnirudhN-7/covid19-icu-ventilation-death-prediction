"""Central configuration: paths, column conventions, experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SEED = 42

# ---- Column conventions (rename your real dataset's columns to match) ----
ID_COL = "patient_id"
DATE_COL = "admission_date"          # optional; enables the paper's temporal cohort split
TASKS = ["icu", "mv", "death"]       # binary target columns (1 = event happened)

PREFIX = {"rad": "rad_", "clin": "clin_", "lab": "lab_"}

# Data combinations compared in the paper
DATA_COMBOS = {
    "Radiom": ["rad"],
    "RadioClin": ["rad", "clin"],
    "RadioClinLab": ["rad", "clin", "lab"],
    "ClinLab": ["clin", "lab"],
    "Clin": ["clin"],            # clinical only (admission-safe baseline; no labs/vitals)
}

# ---- Experiment settings ----
CV_FOLDS = 5
DEV_TEST_FRACTION = 0.30      # Cohort 1 -> 70% train / 30% test
VALIDATION_FRACTION = 0.30    # latest 30% of patients = Cohort 2 (unseen validation)
N_BOOTSTRAPS = 30

SELECTORS_FULL = ["none", "lasso", "pca", "gus", "fpr"]
MODELS_FULL = ["LR", "RF", "SVM", "MLP", "LGBM"]
SELECTORS_FAST = ["none", "lasso"]
MODELS_FAST = ["LR", "RF"]
