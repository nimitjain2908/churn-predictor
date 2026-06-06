# Customer Churn Prediction Dashboard

An end-to-end machine learning pipeline that predicts customer churn using LightGBM, explains predictions using SHAP, tracks experiments with MLflow, and presents results in an interactive Streamlit dashboard.

---

## Demo

![Churn Dashboard](outputs/shap_summary.png)

---

## Features

- **Churn Prediction** — LightGBM classifier trained on Telco Customer Churn dataset (AUC: 0.84)
- **Explainability** — SHAP beeswarm plot for global feature importance and waterfall chart for individual prediction explanation
- **Experiment Tracking** — MLflow logs parameters and metrics across training runs
- **Interactive Dashboard** — 3-tab Streamlit app with overview, explainability, and customer simulator

---

## Project Structure
churn-predictor/
├── data/
│   └── telco_churn.csv       # Raw dataset
├── src/
│   ├── preprocess.py         # Data cleaning, encoding, train/test split
│   ├── train.py              # Model training and MLflow logging
│   ├── explain.py            # SHAP value computation and plots
│   └── predict.py            # Single and batch prediction
├── app.py                    # Streamlit dashboard
└── requirements.txt          # Dependencies

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LightGBM | Gradient boosting classifier |
| SHAP | Model explainability |
| MLflow | Experiment tracking |
| Streamlit | Interactive dashboard |
| Plotly | Charts and visualisations |
| scikit-learn | Preprocessing and metrics |
| pandas | Data manipulation |

---

## Setup

**1. Clone the repository**
git clone https://github.com/nimitjain2908/churn-predictor.git
cd churn-predictor

**2. Install dependencies**
pip install -r requirements.txt

**3. Train the model**
python -m src.train

**4. Run the dashboard**
streamlit run app.py

---

## Results

| Metric | Score |
|---|---|
| AUC | 0.8358 |
| Accuracy | 80.24% |
| Test Set Size | 1,407 customers |

---

## Key Insights from SHAP

- **Contract type** is the strongest predictor — month-to-month customers churn significantly more
- **Tenure** is the second most important — newer customers are at higher risk
- **Online Security** acts as a retention anchor — customers without it are more likely to leave
- **Monthly Charges** — higher bills correlate with increased churn risk

---

## Built by

Nimit Jain · [LinkedIn](https://linkedin.com/in/nimitjain2908) · [GitHub](https://github.com/nimitjain2908)