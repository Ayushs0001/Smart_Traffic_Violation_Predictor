<div align="center">

# 🚦 Smart Traffic Violation Predictor

### End-to-End Machine Learning System for Traffic Signal Violation Analytics & Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Models-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

*A complete data science pipeline — from messy raw data to a live, interactive prediction dashboard.*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [What It Predicts](#-what-it-predicts)
- [Live Dashboard](#-live-dashboard)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Tech Stack](#-tech-stack)
- [Repository Structure](#-repository-structure)
- [Results & Model Performance](#-results--model-performance)
- [Key Insights](#-key-insights)
- [Key Design Decision — Avoiding Data Leakage](#-key-design-decision--avoiding-data-leakage)
- [Installation & Usage](#-installation--usage)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🎯 Overview

**Smart Traffic Violation Predictor** is an end-to-end **Machine Learning** project (classical ML only — no deep learning) that analyzes traffic signal and vehicle behavior data to predict rule violations, classify violation types, estimate signal waiting time, and flag high-risk junctions.

It covers the **complete data science lifecycle**:

```
Raw messy data → Cleaning → EDA → Feature Engineering → Model Training →
Evaluation → Model Persistence → Interactive Dashboard
```

Built on a realistic synthetic dataset simulating **15 traffic junctions over 6 months** — 26,000+ vehicle observations covering speed, signal state, weather, lane compliance, helmet/seatbelt usage, and more.

---

## ❓ Problem Statement

Traffic authorities generate huge volumes of signal and violation data but rarely turn it into **predictive, actionable intelligence**. This project answers:

- Will a given vehicle likely commit a rule violation, based on context (time, weather, speed, traffic density)?
- If a violation happens, which type is most probable — speeding, red-light jumping, no helmet, etc.?
- How long will a vehicle wait at a given signal, and what happens if the signal is retimed?
- Which junctions are the riskiest, and where should enforcement/infrastructure investment be prioritized?

---

## 🧠 What It Predicts

| # | Task | ML Type | Algorithm | Metric |
|---|------|---------|-----------|--------|
| 1 | **Violation Risk** — will this vehicle violate a rule? | Binary Classification | Random Forest | ROC-AUC ≈ **0.72** |
| 2 | **Violation Type** — which rule will be broken? | Multiclass Classification | Random Forest | Macro-F1 ≈ **0.37** |
| 3 | **Waiting Time** — how long at the signal? | Regression | Random Forest | R² ≈ **0.95**, MAE ≈ **4.4s** |
| 4 | **Junction Risk Segment** — Low / Medium / High risk | Unsupervised Clustering | K-Means | Silhouette ≈ **0.30** (k=3) |

Each model was compared against simpler baselines (Logistic/Linear Regression) before selecting the best performer — see [Results](#-results--model-performance).

---

## 💻 Live Dashboard

`app.py` launches a **6-page interactive Streamlit dashboard**:

| Page | What it does |
|---|---|
| 📊 **Executive Overview** | KPIs, junction-wise violations, vehicle-type breakdown, monthly trend |
| 🔍 **Exploratory Analysis** | Violation patterns by hour, speed distribution, correlation heatmap |
| ⚠️ **Violation Risk Predictor** | Live prediction — input conditions, get violation probability + gauge chart |
| 🏷️ **Violation Type Predictor** | Predicts the most likely violation type with a probability ranking |
| ⏱️ **Waiting Time Predictor** | Signal-retiming "what-if" simulator |
| 🗺️ **Junction Risk Explorer** | Risk matrix, K-Means clusters, sortable risk table |

> Run it locally with `streamlit run app.py` — see [Installation & Usage](#-installation--usage).

---

## 📊 Dataset

A synthetic-but-realistic dataset simulating 6 months of traffic camera + signal data:

| Table | Rows | Description |
|---|---|---|
| `Traffic_Facts.csv` | 26,104 | Main fact table — one row per vehicle observation (intentionally messy) |
| `Junction_Master.csv` | 15 | Junction metadata (road type, speed limit, lanes, coordinates) |
| `Vehicle_Master.csv` | 6,000 | Vehicle pool (type, fuel, registration) |
| `Signal_Master.csv` | 15 | Signal timing (red/yellow/green durations, cycle time) |
| `Violation_Master.csv` | 9 | Violation types, severity, and assumed penalty |

The raw data intentionally contains **real-world data-quality issues** — mixed date formats, inconsistent categorical casing (`bike` / `BIKE` / `Two Wheeler`), duplicate rows, and nulls — all handled in the cleaning stage (`src/01_data_cleaning.py`).

---

## 🏗️ Pipeline Architecture

```
data/*.csv (raw, messy)
       │
       ▼
 01_data_cleaning.py     → standardizes text, parses mixed dates/times,
       │                    dedupes, merges dimension tables
       ▼
data/traffic_cleaned.csv  (26,000 rows × 45 features)
       │
       ├── 02_eda.py                             → exploratory plots
       ├── 03_train_violation_classifier.py       → models/violation_classifier.pkl
       ├── 04_train_violation_type_classifier.py  → models/violation_type_classifier.pkl
       ├── 05_junction_risk_clustering.py         → models/junction_risk_clustering.pkl
       └── 06_train_waiting_time_regressor.py     → models/waiting_time_regressor.pkl
                  │
                  ▼
        predict_demo.py   /   app.py (Streamlit)
        → loads all saved models and serves live predictions
```

---

## 🧱 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data handling | pandas, numpy |
| Machine Learning | scikit-learn (Random Forest, Gradient Boosting, Logistic/Linear Regression, K-Means, PCA) |
| Visualization | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |
| Notebook | Jupyter |
| Model persistence | joblib |

---

## 📁 Repository Structure

```
Smart_Traffic_Violation_Predictor/
├── data/
│   ├── Traffic_Facts.csv          # raw fact table (messy, as-collected)
│   ├── Junction_Master.csv
│   ├── Vehicle_Master.csv
│   ├── Signal_Master.csv
│   ├── Violation_Master.csv
│   └── traffic_cleaned.csv        # generated by 01_data_cleaning.py
├── src/
│   ├── 01_data_cleaning.py
│   ├── 02_eda.py
│   ├── 03_train_violation_classifier.py
│   ├── 04_train_violation_type_classifier.py
│   ├── 05_junction_risk_clustering.py
│   ├── 06_train_waiting_time_regressor.py
│   └── predict_demo.py
├── models/                        # trained .pkl pipelines (joblib)
├── reports/
│   ├── figures/                   # EDA + evaluation plots (PNG)
│   ├── violation_classifier_comparison.csv
│   ├── violation_type_classifier_report.txt
│   ├── waiting_time_regressor_comparison.csv
│   └── junction_risk_clusters.csv
├── Traffic_ML_Project.ipynb       # entire pipeline in one runnable notebook
├── app.py                         # Streamlit dashboard (live predictions)
├── requirements.txt
└── README.md
```

---

## 📈 Results & Model Performance

### 1. Violation Risk Classifier (binary)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.658 | 0.539 | 0.586 | 0.561 | 0.702 |
| **Random Forest ✅** | **0.776** | **0.983** | 0.407 | 0.575 | **0.718** |
| Gradient Boosting | 0.776 | 0.990 | 0.404 | 0.574 | 0.718 |

### 2. Violation Type Classifier (multiclass, 8 classes)

Accuracy **0.63**, Macro-F1 **0.37**. `Over Speeding` is predicted almost perfectly (it's directly derivable from speed vs. limit); classes with no contextual signal (e.g. random phone-usage flags) are, correctly, harder to predict — an honest reflection of the underlying data rather than a modeling flaw.

### 3. Waiting Time Regressor

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 6.57s | 8.32s | 0.905 |
| **Random Forest ✅** | **4.42s** | **6.20s** | **0.947** |

### 4. Junction Risk Clustering

K-Means with **k=3** (chosen via silhouette score) segments junctions into **Low / Medium / High Risk** based on violation rate, accident rate, density, waiting time, and signal efficiency — cross-validated against a manual rule-based risk score, with strong agreement on the top risk junctions.

> Full evaluation plots (confusion matrices, ROC curves, feature importances, PCA cluster maps) are in `reports/figures/`.

---

## 💡 Key Insights

- Overall violation rate: **37.4%**
- **Over-speeding** is the dominant violation type — **41%** of all violations
- Violations peak during **evening rush (6–7 PM)** and **morning rush (8–10 AM)**
- **Bikes + Cars** account for **~63%** of all violations
- Total fine collected across 6 months: **₹1.32 crore**
- **307 accidents** recorded, correlated with violations, poor road condition, and adverse weather

---

## 🔬 Key Design Decision — Avoiding Data Leakage

Features like `Helmet`, `Seat_Belt`, and `Phone_Usage` exist in the raw data but were **deliberately excluded** from the violation classifier. These flags directly *define* the violation label — including them would turn the task into a trivial lookup (>99% accuracy) instead of genuine prediction.

Instead, the model predicts risk purely from **contextual signals** — time of day, weather, vehicle type, speed relative to the limit, and traffic density — reflecting the real-world use case of estimating risk **before** observing driver behavior. This is why ROC-AUC sits at ~0.72 rather than ~1.0, and it's the honest, leakage-free number to report.

---

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/Ayushs0001/Smart_Traffic_Violation_Predictor.git
cd Smart_Traffic_Violation_Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline step-by-step
python src/01_data_cleaning.py
python src/02_eda.py
python src/03_train_violation_classifier.py
python src/04_train_violation_type_classifier.py
python src/05_junction_risk_clustering.py
python src/06_train_waiting_time_regressor.py

# 4. Try a live prediction from the command line
python src/predict_demo.py

# 5. Launch the interactive dashboard
streamlit run app.py
```

**Or** simply open `Traffic_ML_Project.ipynb` and Run All — it executes the entire pipeline end to end in one notebook.

---

## 🔮 Future Improvements

- Deploy `app.py` on Streamlit Community Cloud for a public live demo link
- Add hyperparameter tuning (GridSearchCV / Optuna) for further model improvement
- Incorporate real traffic-camera data if/when available, replacing the synthetic set
- Add a REST API (FastAPI) wrapper around the saved models for external integration
- Time-series forecasting of violation trends per junction

---

## ⚠️ Disclaimer

This project uses a **synthetic dataset** generated for portfolio and learning purposes. It does **not** represent any real city, junction, or officially enforced fine schedule — all penalty amounts and location data are illustrative.



