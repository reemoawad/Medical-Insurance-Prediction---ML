COMPE 510 — Medical Insurance Cost Prediction

Final Project — Fall 2025

Authors: Abdulmohsen Almunayes, Reem Awad, Tri Bui

Project Description

This project uses machine learning to predict annual medical insurance charges using the Medical Insurance Personal Dataset from Kaggle.
We explore exploratory data analysis (EDA), train five regression algorithms, compare their performance, and deploy the best model using a Streamlit web app.

Project Structure:

insurance-ml-project/
│
├── algorithms/              # Training notebooks for each ML model
│   ├── train_knn.ipynb
│   ├── train_linear_regression.ipynb
│   ├── train_decision_tree.ipynb
│   ├── train_random_forest.ipynb
│   ├── train_xgb.ipynb
│   └── compare_models.ipynb
│
├── EDA/                     # Exploratory Data Analysis notebook + generated figures
│   ├── insurance_eda.ipynb
│   ├── numerical_distributions.png
│   ├── categorical_vs_charges.png
│   ├── correlation_matrix.png
│   ├── charges_distribution.png
│   └── (other EDA charts)
│
├── models/                  # Saved trained models + metrics
│   ├── knn_regressor.joblib
│   ├── linear_regression.joblib
│   ├── decision_tree_regressor.joblib
│   ├── random_forest_regressor.joblib
│   ├── xgboost_regressor.joblib
│   ├── *_metrics.csv
│   ├── model_comparison_summary.csv
│   └── best_model.txt       # Name of the best model (selected automatically)
│
├── app/                     # Streamlit web application
│   └── app.py
│
├── streamlit/               # Optional UI theme
│   └── config.toml
│
├── insurance.csv            # Dataset
├── requirements.txt         # Python dependencies
└── README.md                # (This file)


Setup Instructions

1. Create & Activate the Virtual Environment

    python3 -m venv .venv
    source .venv/bin/activate

2. Install All Dependencies

    pip install -r requirements.txt


Running the Machine Learning Models:

Run them one by one:

	1.	train_linear_regression.ipynb
	2.	train_decision_tree.ipynb
	3.	train_knn.ipynb
	4.	train_random_forest.ipynb
	5.	train_xgb.ipynb



Each notebook will automatically:

	•	Preprocess the dataset
	•	Train the model
	•	Generate diagnostic figures
	•	Save:
	•	a .joblib model
	•	a *_metrics.csv file


After training all models, run:

    algorithms/compare_models.ipynb


This notebook finds the best model (lowest RMSE) and writes its filename into:

    models/best_model.txt
    The Streamlit app reads this file automatically.


Running the Streamlit Application:

    cd app
    streamlit run app.py

    Features:
	•	Automatically loads the best model
	•	Clean modern UI
	•	Inputs for age, BMI, children, sex, smoker, region
	•	Preset profile options
	•	Real-time charge predictions
	•	Shows model performance metrics



