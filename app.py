"""Streamlit demo:  streamlit run app.py
Loads the models saved by main.py (models/<task>_<combo>.joblib) and predicts risk.
Features left blank (e.g. radiomics) are imputed with training medians by the pipeline."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from src.config import MODELS_DIR, TASKS, DATA_COMBOS

st.set_page_config(page_title="COVID-19 Patient Triage", layout="wide")
st.title("COVID-19 Patient Triage: ICU / Ventilation / Death risk")
st.caption("Educational project. Not a medical device.")

combo = st.sidebar.selectbox("Data available", list(DATA_COMBOS), index=3)
models = {t: MODELS_DIR / f"{t}_{combo}.joblib" for t in TASKS}
if not all(p.exists() for p in models.values()):
    st.error("Models not found. Run `python main.py` first."); st.stop()
models = {t: joblib.load(p) for t, p in models.items()}
features = list(models[TASKS[0]].feature_names_in_)

mode = st.radio("Input mode", ["Manual entry", "Upload CSV"], horizontal=True)
THRESH = {"low": 0.2, "high": 0.5}

def risk_label(p):
    return "HIGH" if p >= THRESH["high"] else "MODERATE" if p >= THRESH["low"] else "LOW"

def predict(frame):
    return pd.DataFrame({t.upper(): m.predict_proba(frame[features])[:, 1] for t, m in models.items()})

if mode == "Manual entry":
    shown = [f for f in features if not f.startswith("rad_")]
    vals = {}
    cols = st.columns(3)
    for i, f in enumerate(shown):
        vals[f] = cols[i % 3].number_input(f, value=None, placeholder="leave blank = median")
    if any(f.startswith("rad_") for f in features):
        st.info("Radiomics features are not entered manually; training medians are used.")
    if st.button("Predict risk"):
        row = pd.DataFrame([{f: vals.get(f, np.nan) for f in features}], dtype=float)
        probs = predict(row).iloc[0]
        c = st.columns(len(probs))
        for col, (name, p) in zip(c, probs.items()):
            col.metric(name, f"{p:.1%}", risk_label(p))
else:
    up = st.file_uploader("CSV with the feature columns", type="csv")
    if up:
        data = pd.read_csv(up)
        for f in features:
            if f not in data: data[f] = np.nan
        out = pd.concat([data.reset_index(drop=True), predict(data)], axis=1)
        st.dataframe(out)
        st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
