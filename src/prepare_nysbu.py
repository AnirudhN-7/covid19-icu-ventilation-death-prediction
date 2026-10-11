"""Convert the COVID-19-NY-SBU tables (Kaggle: toxite/covid-19-cxr-ny-sbu) into the
project's merged.csv format (patient_id, icu, mv, death, clin_*, lab_*, [rad_*]).

Usage:
    python -m src.prepare_nysbu --details path/to/patient_details.csv --out data/merged.csv
    python -m src.prepare_nysbu --details ... --radiomics data/radiomics.csv   # once you have rad_ features
"""
import argparse
import re
import numpy as np
import pandas as pd
from .config import DATA_DIR, ID_COL

# ---- Columns that would LEAK the outcome (happen during/after the stay) ----
LEAKY = [
    "last.status", "is_icu", "was_ventilated", "invasive_vent_days", "length_of_stay",
    "Acute.Hepatic.Injury..during.hospitalization.", "Acute.Kidney.Injury..during.hospitalization.",
    "kidney_replacement_therapy",                                   # treatment given in hospital
    "therapeutic.exnox.Boolean", "therapeutic.heparin.Boolean", "Other.anticoagulation.therapy",
    "visit_start_datetime", "visit_concept_name", "covid19_statuses",  # dates are anonymised; visit type reflects disposition
]
# Pre-binned copies of raw lab/vital values -> redundant, dropped (we keep the raw numbers)
BINNED = re.compile(r"\.(over|under|above|below|between)|^A1C\.|\.over\d|\.under\d", re.I)

YESNO = {"yes": 1, "no": 0, "true": 1, "false": 0}


def _slug(loinc_col: str) -> str:
    name = loinc_col.split("_", 1)[1]
    name = re.split(r"\s\[|\sin\s|,", name)[0]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:28]


def build_table(details_path, max_missing=0.6) -> pd.DataFrame:
    d = pd.read_csv(details_path)
    out = pd.DataFrame({ID_COL: d["to_patient_id"]})

    # ---- Outcomes (three binary targets, same as the paper) ----
    out["icu"] = d["is_icu"].astype(str).str.lower().map(YESNO).astype(int)
    out["mv"] = d["was_ventilated"].astype(str).str.lower().map(YESNO).astype(int)
    out["death"] = (d["last.status"].str.lower() == "deceased").astype(int)

    # ---- Clinical features ----
    clin = pd.DataFrame(index=d.index)
    clin["clin_age_group"] = d["age.splits"].map({"[18,59]": 0, "(59,74]": 1, "(74,90]": 2})
    clin["clin_male"] = d["gender_concept_name"].map({"MALE": 1, "FEMALE": 0})
    for c in ["htn_v", "dm_v", "cad_v", "ckd_v", "malignancies_v", "copd_v", "other_lung_disease_v",
              "acei_v", "arb_v", "antibiotics_use_v", "nsaid_use_v", "cough_v", "dyspnea_admission_v",
              "nausea_v", "vomiting_v", "diarrhea_v", "abdominal_pain_v", "fever_v"]:
        clin["clin_" + c.replace("_v", "")] = d[c].astype(str).str.lower().map(YESNO)
    clin["clin_heart_failure"] = d["hf_ef_v"].map(lambda v: np.nan if pd.isna(v) else int(v != "No"))
    clin["clin_smoking"] = d["smoking_status_v"].map({"Never": 0, "Former": 1, "Current": 2})
    clin["clin_days_prior_sx"] = d["days_prior_sx"]
    clin["clin_bmi"] = d["39156-5_Body mass index (BMI) [Ratio]"]
    clin["clin_kidney_transplant"] = d["kidney_transplant"].notna().astype(int)  # only 'Yes' stored; blank = no

    # ---- Lab + vitals features (raw numeric values; LOINC-coded columns) ----
    raw = d["2951-2_Sodium [Moles/volume] in Serum or Plasma"]
    dup = d["2951-2_Sodium [Moles/volume] in Serum or Plasma.1"]
    assert raw.equals(dup) or raw.fillna(-1).equals(dup.fillna(-1)), "duplicate sodium columns differ"
    lab = pd.DataFrame(index=d.index)
    for c in d.columns:
        if re.match(r"^\d+-\d+_", c) and not c.endswith(".1") and "Body mass index" not in c:
            name = "lab_" + _slug(c)
            if "Cholesterol in" in c:                       # HDL / LDL / VLDL share one slug
                name += "_" + c.split("Cholesterol in ")[1].split(" ")[0].lower()
            lab[name] = d[c]
    lab["lab_urine_protein_abnormal"] = d["Urine.protein"].map({"Abnormal": 1, "Normal": 0})

    feats = pd.concat([clin, lab], axis=1)
    keep = feats.columns[feats.isna().mean() <= max_missing]
    dropped = sorted(set(feats.columns) - set(keep))
    print(f"Dropped >{max_missing:.0%}-missing features ({len(dropped)}): {dropped}")
    return pd.concat([out, feats[keep]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True, help="path to patient_details.csv")
    ap.add_argument("--radiomics", help="optional CSV with patient_id + rad_* columns")
    ap.add_argument("--out", default=str(DATA_DIR / "merged.csv"))
    ap.add_argument("--max-missing", type=float, default=0.6)
    a = ap.parse_args()

    df = build_table(a.details, a.max_missing)
    if a.radiomics:
        rad = pd.read_csv(a.radiomics)
        df = df.merge(rad, on=ID_COL, how="inner")
        print(f"Merged radiomics -> {len(df)} patients with imaging features")
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Saved {a.out}: {df.shape[0]} patients, "
          f"{sum(c.startswith('clin_') for c in df)} clin, {sum(c.startswith('lab_') for c in df)} lab, "
          f"{sum(c.startswith('rad_') for c in df)} rad features")
    print(df[["icu", "mv", "death"]].agg(["sum", "mean"]).round(3))


if __name__ == "__main__":
    main()
