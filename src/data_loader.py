"""Load and merge radiomics / clinical / lab tables with outcomes."""
import pandas as pd
from .config import DATA_DIR, ID_COL, DATE_COL, PREFIX, TASKS

TABLES = ["outcomes", "clinical", "labs", "radiomics"]


def load_dataset(data_dir=DATA_DIR) -> pd.DataFrame:
    """Either one `merged.csv`, or separate outcomes/clinical/labs/radiomics CSVs
    joined on patient_id (inner join: patients must have every data type)."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    merged = data_dir / "merged.csv"
    if merged.exists():
        df = pd.read_csv(merged)
    else:
        missing = [t for t in TABLES if not (data_dir / f"{t}.csv").exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing {missing} in {data_dir}. Provide merged.csv or the 4 CSVs, "
                "or run with --synthetic to test the pipeline.")
        df = pd.read_csv(data_dir / "outcomes.csv")
        for t in TABLES[1:]:
            df = df.merge(pd.read_csv(data_dir / f"{t}.csv"), on=ID_COL, how="inner")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    for t in TASKS:
        if t not in df.columns:
            raise ValueError(f"Target column '{t}' not found.")
    return df


def get_feature_columns(df: pd.DataFrame, groups) -> list:
    prefixes = tuple(PREFIX[g] for g in groups)
    return [c for c in df.columns if c.startswith(prefixes)]


def summarize(df: pd.DataFrame):
    print(f"Patients: {len(df)}")
    for g in PREFIX:
        print(f"  {g}: {len(get_feature_columns(df, [g]))} features")
    for t in TASKS:
        print(f"  {t}: {int(df[t].sum())} positives ({df[t].mean():.1%})")
