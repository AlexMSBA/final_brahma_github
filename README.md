# Flight Delay Prediction — Decision Tree + Random Forest

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlexMSBA/final_brahma_github/blob/main/decision_tree_and_random_forest_flight_delay.ipynb)

**BSAN 6070 — Machine Learning Final Project, Spring 2026 · Loyola Marymount University**

This repository contains the tree-based modeling component of a team project predicting U.S. domestic flight arrival delays at six major hub airports. It implements a Decision Tree baseline and a Random Forest ensemble using the same features, the same hub filter, and the same train/test split — so any change in performance can be attributed directly to ensembling.

## Predictive Question

> Can we predict whether a U.S. domestic flight at one of six major hub airports (ORD, CLT, LAX, DEN, DFW, ATL) will arrive 15 minutes or more late, using only airline, route, date, and schedule information available before departure?

- **Target:** `ArrDel15` — binary, 1 if arrival ≥ 15 minutes late
- **Years:** 2018, 2019, 2021, 2022 (2020 excluded as a pandemic-era outlier)
- **Hub Airports:** ORD · CLT · LAX · DEN · DFW · ATL

## Repository Layout

| File | Purpose |
|------|---------|
| `decision_tree_and_random_forest_flight_delay.ipynb` | Main notebook — preprocessing, hub filter, both models, tuning, evaluation |
| `cleaned_flights.parquet` | Cleaned BTS dataset (not committed — see below) |
| `README.md` | This file |

## Data

The notebook expects a cleaned BTS On-Time Performance dataset at:

```
/content/cleaned_flights.parquet
```

The cleaned file contains roughly 9.67M rows and 17 columns, with the target `ArrDel15` already encoded. The original raw data is published by the U.S. Bureau of Transportation Statistics and is not included in this repository due to size.

## Methodology Overview

1. **Leakage Removal** — Drop 11 post-arrival columns (`ArrDelay`, `ArrTime`, `AirTime`, `WheelsOn`, `TaxiIn`, etc.) before any modeling.
2. **Candidate Features (15)** — Four groups defined at scheduling time:
   - *Temporal* — `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek`
   - *Schedule* — `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `DepTimeBlk`
   - *Route* — `Origin`, `Dest`, `Distance`, `DistanceGroup`
   - *Airline* — `Marketing_Airline_Network`, `Operating_Airline`
3. **Hub Filter** — Keep flights touching one of the six hubs → 9.67M rows down to 4.22M (56% reduction).
4. **Preprocessing** — Label-encode categoricals; no scaling needed for tree-based models; median-impute residual nulls.
5. **Train/Test Split** — Stratified 80/20 → 3.30M train / 825K test rows. Class imbalance (4.25 : 1) handled via `class_weight='balanced'`.
6. **Sampling for Iteration** — A 100,000-row stratified subsample used for `RandomizedSearchCV`; final scoring always on the full hold-out test set.
7. **Mutual Information** — Confirms `CRSDepTime`, `CRSArrTime`, `DepTimeBlk` as the top-ranked features.
8. **Decision Tree** — Baseline + `RandomizedSearchCV` tuning.
9. **Random Forest** — Same per-tree settings as DT plus `n_estimators`, `max_features='sqrt'`, `bootstrap`, `oob_score`. Tuned via `RandomizedSearchCV`.

## Results

| Model | Accuracy | F1 | ROC-AUC |
|-------|---------:|---:|--------:|
| Decision Tree (tuned) | 0.572 | 0.350 | **0.6210** |
| Random Forest (tuned) | 0.615 | 0.358 | **0.6361** |

**Random Forest lift over Decision Tree at every hub** (test-set ROC-AUC):

| Hub | DT | RF | Lift |
|-----|---:|---:|----:|
| ATL | 0.607 | 0.640 | +0.033 |
| DFW | 0.597 | 0.617 | +0.020 |
| LAX | 0.580 | 0.595 | +0.015 |
| CLT | 0.629 | 0.641 | +0.012 |
| DEN | 0.626 | 0.638 | +0.012 |
| ORD | 0.601 | 0.607 | +0.006 |

The tuned forest contains **187 trees** averaging depth 8, with an out-of-bag score of **0.6143** — closely tracking the test AUC and confirming the model is not overfitting. `CRSDepTime` is the single most important predictor by both Gini and permutation importance.

## How to Run

### Google Colab (recommended)

1. Click the "Open in Colab" badge above.
2. Mount Google Drive (or upload `cleaned_flights.parquet` to `/content/`).
3. Run all cells. A 100K-row sample is used by default for tuning; flip `USE_FULL_TRAIN_FOR_BASELINE` / `USE_FULL_TRAIN_FOR_RF_BASELINE` to `True` for full-dataset training.

### Local

```bash
git clone https://github.com/AlexMSBA/final_brahma_github.git
cd final_brahma_github
pip install pandas numpy matplotlib seaborn scikit-learn joblib graphviz pyarrow
jupyter notebook decision_tree_and_random_forest_flight_delay.ipynb
```

Update the data path inside the notebook from `/content/cleaned_flights.parquet` to wherever the cleaned dataset is stored on your machine.

## Saved Artifacts

After running, the notebook saves four pickles to `/content/`:

- `dt_flight_delay_model.pkl` — tuned Decision Tree
- `rf_flight_delay_model.pkl` — tuned Random Forest
- `dt_label_encoders.pkl`, `rf_label_encoders.pkl` — fitted `LabelEncoder`s for the categorical columns
- `dt_feature_names.pkl`, `rf_feature_names.pkl` — the 15-feature column order used at training time
- `dt_hub_airports.pkl`, `rf_hub_airports.pkl` — the six-hub list

These can be loaded with `joblib.load(...)` for downstream inference or Streamlit deployment.

## Authors

| Name | Component |
|------|-----------|
| Prince Dasho | Feature engineering, validation, XGBoost / Gradient Boosting |
| Anthony Hanna | Logistic Regression |
| Alex Frieder | Decision Tree + Random Forest (this notebook) |

## License

For academic use — Loyola Marymount University, BSAN 6070, Spring 2026.
