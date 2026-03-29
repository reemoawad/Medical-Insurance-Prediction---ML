# Medical Insurance Cost Prediction

> A full end-to-end machine learning system that predicts annual medical insurance charges from patient demographics — built with a reproducible training pipeline, multi-model benchmarking, and a deployed interactive web application.

---

## Key Results

| Model | R² Score | RMSE (USD) |
|---|---|---|
| Random Forest *(best)* | 0.86 | $4,620 |
| XGBoost | 0.86 | $4,670 |
| Linear Regression | 0.79 | $5,800 |
| Decision Tree | 0.76 | $6,150 |
| KNN | 0.30 | $10,400 |

Random Forest achieved the strongest overall performance with the lowest RMSE and highest R², and was selected as the final model for deployment, with XGBoost performing comparably.

---

## Demo

▶️ [Watch the demo video](https://drive.google.com/file/d/1OBQNgwLiRIk7V8-hD-_pUAZGlSLgfH-K/view)

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
