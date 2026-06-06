import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.preprocess import load_and_preprocess

def get_shap_values():
    model = joblib.load("outputs/churn_model.pkl")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    return shap_values, X_test, feature_names

def plot_summary(shap_values, X_test):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    return "outputs/shap_summary.png"

def plot_waterfall(shap_values, index=0):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    return "outputs/shap_waterfall.png"

if __name__ == "__main__":
    shap_values, X_test, feature_names = get_shap_values()
    plot_summary(shap_values, X_test)
    plot_waterfall(shap_values)
    print("SHAP plots saved to outputs/")