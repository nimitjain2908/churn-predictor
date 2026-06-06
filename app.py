import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import os
from src.preprocess import load_and_preprocess
from src.explain import get_shap_values, plot_summary, plot_waterfall
from src.predict import predict_batch

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Customer Churn Prediction Dashboard")
st.caption("Powered by LightGBM + SHAP · Built by Nimit Jain")

# ── Load data and model ───────────────────────────────────────────────
@st.cache_data
def load_data():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()
    return X_test, y_test, feature_names

@st.cache_resource
def load_shap():
    return get_shap_values()

X_test, y_test, feature_names = load_data()
shap_values, _, _ = load_shap()

import mlflow

def load_metrics():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("churn-prediction")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs:
        metrics = runs[0].data.metrics
        return round(metrics.get("auc", 0), 4), round(metrics.get("accuracy", 0), 4)
    return None, None

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Overview", "Explainability", "Predict"])

# ── Tab 1: Overview ───────────────────────────────────────────────────
with tab1:
    st.subheader("Model Performance")

    auc, accuracy = load_metrics()

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC Score", auc if auc else "N/A")
    col2.metric("Accuracy", f"{round(accuracy * 100, 2)}%" if accuracy else "N/A")
    col3.metric("Test Set Size", f"{len(X_test)} customers")

    st.divider()

    results_df = predict_batch(X_test)

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Churn prediction distribution")
        fig1 = px.pie(
            results_df,
            names="prediction",
            color="prediction",
            color_discrete_map={"Churn": "#D85A30", "No Churn": "#1D9E75"}
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col5:
        st.subheader("Risk level distribution")
        risk_counts = results_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig2 = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            color="Risk Level",
            color_discrete_map={"High": "#D85A30", "Medium": "#BA7517", "Low": "#1D9E75"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    st.subheader("Churn probability distribution")
    fig3 = px.histogram(
        results_df,
        x="churn_probability",
        color="prediction",
        nbins=40,
        color_discrete_map={"Churn": "#D85A30", "No Churn": "#1D9E75"}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── Tab 2: Explainability ─────────────────────────────────────────────
with tab2:
    st.subheader("Global feature importance")
    st.caption("What drives churn across all customers?")

    summary_path = plot_summary(shap_values, X_test)
    st.image(summary_path, use_container_width=True)

    st.divider()

    st.subheader("Individual prediction explanation")
    st.caption("Why did the model make this prediction for a specific customer?")

    customer_index = st.slider("Select customer index", 0, len(X_test) - 1, 0)
    waterfall_path = plot_waterfall(shap_values, index=customer_index)
    st.image(waterfall_path, use_container_width=True)

    with st.expander("View customer details"):
        st.dataframe(X_test.iloc[[customer_index]], use_container_width=True)

# ── Tab 3: Predict ────────────────────────────────────────────────────
with tab3:
    st.subheader("Predict churn for a new customer")
    st.caption("Adjust the sliders and dropdowns to simulate a customer profile")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 65)
        total_charges = st.number_input("Total Charges ($)", value=float(tenure * monthly_charges))

    with col_b:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    with col_c:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("▶ Predict", type="primary"):
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
        payment_map = {
            "Electronic check": 0, "Mailed check": 1,
            "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
        }
        binary_map = {"No": 0, "Yes": 1, "No internet service": 2}

        customer = {f: 0 for f in feature_names}
        customer["tenure"] = tenure
        customer["MonthlyCharges"] = monthly_charges
        customer["TotalCharges"] = total_charges
        customer["Contract"] = contract_map[contract]
        customer["InternetService"] = internet_map[internet_service]
        customer["PaymentMethod"] = payment_map[payment_method]
        customer["OnlineSecurity"] = binary_map[online_security]
        customer["TechSupport"] = binary_map[tech_support]
        customer["PaperlessBilling"] = binary_map[paperless_billing]

        from src.predict import predict_single
        result = predict_single(customer)

        color = "#D85A30" if result["prediction"] == "Churn" else "#1D9E75"
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {color}; color: white; text-align: center;'>
            <h2>{result['prediction']}</h2>
            <p>Churn Probability: <b>{result['churn_probability']}</b></p>
            <p>Risk Level: <b>{result['risk_level']}</b></p>
        </div>
        """, unsafe_allow_html=True)