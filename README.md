# Medical Insurance Cost Prediction

> A full end-to-end machine learning system that predicts annual medical insurance charges from patient demographics — built with a reproducible training pipeline, multi-model benchmarking, and a deployed interactive web application.

---

## Key Results

| Model | R² Score | RMSE |
|---|---|---|
| XGBoost *(best)* | — | lowest |
| Random Forest | — | — |
| Decision Tree | — | — |
| Linear Regression | — | — |
| KNN | — | — |

> ⚠️ *Replace placeholders above with actual metrics from `models/model_comparison_summary.csv` after training.*

The best-performing model is selected automatically at runtime and loaded directly into the Streamlit app via `models/best_model.txt`.

---

## Demo

> 📸 *Insert Streamlit app screenshot here*

```
[ screenshot placeholder ]
```

---

## Tech Stack

`Python` `scikit-learn` `XGBoost` `pandas` `NumPy` `Streamlit` `Jupyter` `joblib` `Matplotlib` `Seaborn`

---

## Pipeline Overview

```
Dataset (Kaggle)
    │
    ▼
Exploratory Data Analysis
    │  distributions, correlations, outlier analysis
    ▼
Preprocessing
    │  encoding, feature scaling, train/test split
    ▼
Model Training  (5 algorithms, individual notebooks)
    │  Linear Regression · Decision Tree · KNN
    │  Random Forest · XGBoost
    ▼
Evaluation & Model Selection
    │  compare_models.ipynb → best_model.txt
    ▼
Streamlit Deployment
       real-time predictions via interactive UI
```

---

## Streamlit App

The web app loads the best-trained model automatically and provides a clean interface for real-time cost prediction.

**Inputs:** age, BMI, number of dependents, sex, smoker status, region  
**Features:**
- Preset patient profile options for quick exploration
- Real-time insurance charge prediction on input change
- Displays live model performance metrics (R², RMSE)
- Custom UI theme via Streamlit config

---

## Project Structure

```
Medical-Insurance-Prediction---ML/
├── algorithms/          # Training notebooks (one per model) + compare_models.ipynb
├── EDA/                 # EDA notebook + generated figures (distributions, correlations)
├── models/              # Serialized .joblib models, per-model metrics CSVs, best_model.txt
├── app/                 # Streamlit application (app.py)
├── streamlit/           # UI theme config
├── insurance.csv        # Source dataset
└── requirements.txt
```

---

## My Contributions *(Reem Awad)*

- Built and maintained the **end-to-end ML training pipeline** across all five algorithms
- Designed the **model evaluation and selection logic** in `compare_models.ipynb`, including automated best-model persistence
- Developed and deployed the **Streamlit web application**, including real-time prediction logic and UI layout
- Conducted **exploratory data analysis** (distributions, feature correlations, charge skew investigation)
- Owned the **repository structure and code organization** across training, evaluation, and deployment stages
- Collaborated with teammates on dataset preprocessing and feature engineering decisions

---

## Setup & Usage

**1. Clone and set up the environment**

```bash
git clone https://github.com/reemoawad/Medical-Insurance-Prediction---ML.git
cd Medical-Insurance-Prediction---ML
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Train the models**

Run each notebook in `algorithms/` in any order:

```
train_linear_regression.ipynb
train_decision_tree.ipynb
train_knn.ipynb
train_random_forest.ipynb
train_xgb.ipynb
```

Each notebook preprocesses the data, trains the model, saves a `.joblib` file and a `*_metrics.csv` to `models/`.

**3. Select the best model**

```bash
# Run algorithms/compare_models.ipynb
# Outputs: models/best_model.txt (used automatically by the app)
```

**4. Launch the app**

```bash
cd app
streamlit run app.py
```

---

## Dataset

[Medical Insurance Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) — Kaggle  
1,338 records · 7 features: age, sex, BMI, children, smoker, region, charges

---

*Built by Abdulmohsen Almunayes, Reem Awad, and Tri Bui*
