# Medical Insurance Cost Prediction

> End-to-end machine learning system for predicting medical insurance costs from patient demographics, with model benchmarking and a deployed interactive web app.

---

## Key Results

| Model | R² Score | RMSE (USD) |
|---|---|---|
| Random Forest **(Best Model)** | 0.86 | $4,620 |
| XGBoost | 0.86 | $4,670 |
| Linear Regression | 0.79 | $5,800 |
| Decision Tree | 0.76 | $6,150 |
| KNN | 0.30 | $10,400 |

Random Forest achieved the strongest overall performance with the lowest RMSE and highest R², and was selected as the final model for deployment, with XGBoost performing comparably.

---

## Highlights

- Built an end-to-end ML pipeline from preprocessing to deployment
- Benchmarked 5 regression models to identify optimal performance
- Achieved R² of 0.86 using ensemble methods (Random Forest, XGBoost)
- Deployed an interactive Streamlit app for real-time predictions

---

## Demo

▶️ [Watch the demo video](https://drive.google.com/file/d/1OBQNgwLiRIk7V8-hD-_pUAZGlSLgfH-K/view)

---

## Tech Stack

**Machine Learning:** scikit-learn, XGBoost  
**Data Processing:** pandas, NumPy  
**Visualization:** Matplotlib, Seaborn  
**Deployment:** Streamlit  
**Tools:** Python, Jupyter, joblib

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

The web app automatically loads the best-performing model and provides a clean, interactive interface for real-time insurance cost prediction.

**Inputs:** age, BMI, number of dependents, sex, smoker status, region  
**Features:**
- Preset patient profiles for quick exploration
- Real-time insurance cost prediction on input change
- Displays model performance metrics (R², RMSE) for transparency
- Custom UI styling via Streamlit configuration

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
1,338 records with 7 features: age, sex, BMI, children, smoker status, region, and charges

---

*Built by Abdulmohsen Almunayes, Reem Awad, and Tri Bui*
