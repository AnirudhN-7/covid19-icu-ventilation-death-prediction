"""Synthetic data generator, ONLY for testing the pipeline end-to-end.
Do NOT report results from synthetic data as real findings."""
import numpy as np
import pandas as pd
from .config import DATA_DIR, SEED


def generate(n=2400, n_rad=40, seed=SEED):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 16, n), 18, 95)
    clin = pd.DataFrame({
        "clin_age": age,
        "clin_sex": rng.integers(0, 2, n),
        "clin_dyspnea": rng.binomial(1, 0.15, n),
        "clin_fever": rng.binomial(1, 0.6, n),
        "clin_hypertension": rng.binomial(1, np.clip((age - 30) / 150, 0.02, 0.5)),
        "clin_diabetes": rng.binomial(1, np.clip((age - 30) / 250, 0.02, 0.3)),
    })
    sev = 0.03 * (age - 52) + 1.2 * clin["clin_dyspnea"] + rng.normal(0, 1, n)
    lab = pd.DataFrame({
        "lab_ldh": 220 + 60 * sev + rng.normal(0, 40, n),
        "lab_wbc": 6 + 1.0 * sev + rng.normal(0, 1.5, n),
        "lab_crp": np.clip(15 + 12 * sev + rng.normal(0, 15, n), 0.1, None),
        "lab_neutrophil": 4 + 0.9 * sev + rng.normal(0, 1.2, n),
        "lab_lymphocyte": np.clip(1.5 - 0.25 * sev + rng.normal(0, 0.4, n), 0.1, None),
        "lab_ddimer": np.clip(0.5 + 0.4 * sev + rng.normal(0, 0.4, n), 0.05, None),
        "lab_potassium": 4.0 + rng.normal(0, 0.4, n),
    })
    rad = pd.DataFrame(rng.normal(0, 1, (n, n_rad)), columns=[f"rad_f{i}" for i in range(n_rad)])
    rad["rad_f0"] += 0.8 * sev   # a few informative radiomic features
    rad["rad_f1"] += 0.5 * sev
    rad["rad_f2"] -= 0.4 * sev

    risk = sev + 0.5 * rad["rad_f0"]
    def outcome(intercept, scale):
        p = 1 / (1 + np.exp(-(intercept + scale * (risk - risk.mean()) / risk.std())))
        return rng.binomial(1, p)
    out = pd.DataFrame({"icu": outcome(-2.8, 1.6), "mv": outcome(-3.4, 1.7), "death": outcome(-3.9, 1.5)})

    ids = [f"P{i:05d}" for i in range(n)]
    dates = pd.date_range("2019-12-27", "2020-03-31", periods=n)
    lab = lab.mask(rng.random(lab.shape) < 0.08)  # simulate missing lab values
    return ids, dates, clin, lab, rad, out


def write_synthetic(data_dir=DATA_DIR):
    ids, dates, clin, lab, rad, out = generate()
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.Series(ids, name="patient_id"), pd.Series(dates, name="admission_date"), out], axis=1)\
        .to_csv(data_dir / "outcomes.csv", index=False)
    for name, frame in [("clinical", clin), ("labs", lab), ("radiomics", rad)]:
        frame.insert(0, "patient_id", ids)
        frame.to_csv(data_dir / f"{name}.csv", index=False)
    print(f"Synthetic data written to {data_dir}")


if __name__ == "__main__":
    write_synthetic()
