"""Cleaning, encoding, and train/test/validation splitting."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATE_COL, DEV_TEST_FRACTION, VALIDATION_FRACTION, SEED, PREFIX

_BINARY_MAP = {"yes": 1, "no": 0, "y": 1, "n": 0, "m": 1, "f": 0, "male": 1, "female": 0,
               "true": 1, "false": 0}


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make all feature columns numeric (map yes/no, M/F; coerce the rest). NaNs are
    left in place and imputed inside the model pipeline (avoids leakage)."""
    df = df.copy()
    feat_cols = [c for c in df.columns if c.startswith(tuple(PREFIX.values()))]
    for c in feat_cols:
        if df[c].dtype == object or df[c].dtype == bool:
            df[c] = df[c].astype(str).str.strip().str.lower().map(_BINARY_MAP).where(
                lambda s: s.notna(), pd.to_numeric(df[c], errors="coerce"))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # drop columns that are entirely empty or constant
    drop = [c for c in feat_cols if df[c].isna().all() or df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=drop)


def make_splits(df: pd.DataFrame, task: str, seed: int = SEED):
    """Returns (dev_train, dev_test, validation).
    - With an admission date: latest 30% of patients = Cohort 2 (validation), as in the paper.
    - Otherwise: stratified random split.
    Cohort 1 is then split 70/30 (stratified) into dev_train / dev_test."""
    if DATE_COL in df.columns:
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        cut = int(len(df) * (1 - VALIDATION_FRACTION))
        cohort1, validation = df.iloc[:cut], df.iloc[cut:]
    else:
        cohort1, validation = train_test_split(
            df, test_size=VALIDATION_FRACTION, stratify=df[task], random_state=seed)
    dev_train, dev_test = train_test_split(
        cohort1, test_size=DEV_TEST_FRACTION, stratify=cohort1[task], random_state=seed)
    return dev_train, dev_test, validation
