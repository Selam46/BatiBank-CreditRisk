# Bati Bank Credit Scoring Project

## Overview
This project builds a **Credit Scoring Model** for Bati Bank to classify customers as **Good Risk** or **Bad Risk** using transaction data. The trained model is served via a REST API for real-time predictions.

## Features
- **Exploratory Data Analysis (EDA)**: Understand patterns and handle missing values.
- **Feature Engineering**: RFMS (Recency, Frequency, Monetary, Severity) and WoE binning.
- **Model Training**: Logistic Regression and Random Forest with hyperparameter tuning.
- **Model Serving**: Flask API for real-time credit risk predictions.

## Folder Structure
BatiBank-CreditRisk/
├── data/                   # Contains raw, processed, and interim datasets
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned and feature-engineered datasets
│   └── interim/            # Intermediate data outputs
├── notebooks/              # Jupyter notebooks for each task
│   ├── task1_credit_risk_analysis.ipynb
│   ├── task2_eda.ipynb
│   ├── task3_feature_engineering.ipynb
│   ├── task4_default_estimator.ipynb
│   ├── task5_model_training.ipynb
│   └── task6_api_deployment.ipynb
├── scripts/                # Modular Python scripts
│   ├── preprocessing.py    # Preprocessing utilities
│   ├── feature_engineering.py  # RFMS and WoE functions
│   └── model_training.py   # Model training and evaluation
├── models/                 # Saved trained models
│   ├── best_random_forest.pkl
│   └── logistic_regression.pkl
├── api/                    # Flask API for model serving
│   ├── app.py              # Main API script
│   ├── model_utils.py      # Utility functions for loading models
│   └── requirements.txt    # API-specific dependencies
├── outputs/                # Generated outputs
│   ├── figures/            # Plots and visualizations
│   ├── reports/            # Final reports
│   └── predictions/        # Model predictions
├── README.md               # Project documentation
├── requirements.txt        # Dependencies for the project
└── setup.py                # Makes scripts installable as a package


## Setup Instructions
1. **Clone the Repository**:
   
   git clone https://github.com/your-repo/bati-bank-credit-scoring.git
   cd bati-bank-credit-scoring

## Setup Environment 
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

## Run the Flask API
cd api
python app.py
