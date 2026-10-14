# COVID-19 Patient Triage: Predicting ICU Admission, Mechanical Ventilation and Death

UE24CS352A – Machine Learning, Mini-Project

**Team:** `Anirudh N - PES1UG24AM041`, `Aditya Shetty - PES1UG24AM018`
**Section:** `AIML-A`
**Assigned problem:** CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients (reference: Zhan & Li, Stanford CS229, 2020)

> Educational project. Not a medical device.

---

## 1. Problem Statement

Given data available at hospital admission, predict three outcomes for a COVID-19 patient:

1. **ICU admission**
2. **Mechanical ventilation**
3. **In-hospital death**

Early, accurate risk estimates could help hospitals prioritise scarce resources. Each outcome is treated as a separate binary classification task.

## 2. Dataset

We use the public **COVID-19-NY-SBU** dataset, which provides clinical, laboratory and chest imaging information for COVID-19 patients.

| Item | Value |
|---|---|
| Patients used | 1,365 |
| Image-derived (radiomics) features | 29 |
| Clinical features | 25 |
| Laboratory features | 27 |
| ICU positives | 260 (19.0%) |
| Ventilation positives | 213 (15.6%) |
| Death positives | 182 (13.3%) |

**Note on deviation from the reference paper.** The reference paper used CT radiomics from a private multi-hospital cohort in China, which is not publicly available. We reproduce its methodology on a public dataset whose imaging modality is chest X-ray, so absolute performance differs from the paper. The relative comparison between feature sets is the main focus.

## 3. Approach

The pipeline follows the reference paper:

1. **Feature sets compared:** image features only (`Radiom`), image + clinical (`RadioClin`), image + clinical + lab (`RadioClinLab`), clinical + lab (`ClinLab`), and clinical only (`Clin`).
2. **Class imbalance:** handled with SMOTEENN resampling (inside the training pipeline).
3. **Feature selection:** none, LASSO, and `gus` (univariate selection).
4. **Classifiers:** Logistic Regression (LR), Random Forest (RF), LightGBM (LGBM).
5. **Model selection:** each combination is cross-validated; the best is chosen on the Cohort 1 test split.
6. **Evaluation:** the selected model is evaluated on the held-out Cohort 2 with AUROC and AUPRC (with confidence intervals), since the classes are imbalanced.

## 4. Repository Structure

```
.
├── main.py            # Runs the full experiment (all tasks, feature sets, models)
├── app.py             # Streamlit demo app
├── requirements.txt   # Python dependencies
├── src/               # Data loading, feature selection, training, evaluation code
├── data/              # Dataset files (see Section 2)
├── results/           # Output tables and plots from main.py
└── models/            # Saved trained models used by the demo
```

## 5. Setup

Requires **Python 3.9+** (tested on Python 3.11, Windows).

```bash
# 1. Clone
git clone https://github.com/AnirudhN-7/covid19-icu-ventilation-death-prediction.git
cd covid19-icu-ventilation-death-prediction

# 2. (Recommended) create and activate a virtual environment
python -m venv venv
# Windows PowerShell:
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

Then place the dataset as described in Section 2.

## 6. Running the Project

**Train and evaluate all models** (can take several minutes; the first import of scikit-learn on Windows can also be slow):

```bash
python main.py
```

This prints cross-validation and test AUROC for every combination and writes results to `results/` and trained models to `models/`.

**Launch the demo app:**

```bash
streamlit run app.py
```

Open `http://localhost:8501` if it doesn't open automatically. Choose which data is available in the sidebar, enter values manually (blank fields default to the median) or upload a CSV, and the app returns risk scores for ICU admission, ventilation and death.

## 7. Results

Best model per feature set, evaluated on held-out Cohort 2 (AUROC):

| Task | Radiom | RadioClin | RadioClinLab | ClinLab | Clin |
|---|---|---|---|---|---|
| ICU | 0.619 | 0.713 | 0.842 | 0.842 | 0.645 |
| Ventilation | 0.647 | 0.725 | 0.784 | 0.794 | 0.688 |
| Death | 0.653 | 0.826 | 0.839 | 0.854 | 0.822 |

Full AUROC and AUPRC values are in `results/`.

## 8. Conclusions

- Adding laboratory data gives the largest improvement for all three outcomes.
- Image features alone are weakly predictive (AUROC about 0.62–0.65), but they improve when combined with clinical data.
- With clinical and lab data present, image features add little. `ClinLab` performs about the same as `RadioClinLab`. This differs from the reference paper, where adding CT radiomics helped. Likely reasons are the coarser chest X-ray features and the smaller number of image features in our dataset.
- Differences between `ClinLab` and `RadioClinLab` are within the confidence intervals, so they should be considered comparable.

**Limitations:** different imaging modality and dataset from the paper; no radiologist-score baseline; single dataset without external validation.

## 9. Reference

Zhan, Li. *CT-based Patient Triage of COVID-19: Radiomics Prediction of ICU Admission, Mechanical Ventilation, and Death of Patients.* Stanford CS229 Project Report, Spring 2020. https://cs229.stanford.edu/proj2020spr/report/Zhan_Li.pdf
