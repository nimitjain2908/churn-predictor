import joblib
import pandas as pd
from src.preprocess import load_and_preprocess

def load_model():
    model = joblib.load("outputs/churn_model.pkl")
    return model

def predict_single(customer_data: dict):
    model = load_model()
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

    input_df = pd.DataFrame([customer_data], columns=feature_names)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(prob >= 0.5)

    return {
        "churn_probability": round(float(prob), 4),
        "prediction": "Churn" if pred == 1 else "No Churn",
        "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    }

def predict_batch(df: pd.DataFrame):
    model = load_model()
    _, X_test, _, _, feature_names = load_and_preprocess()

    input_df = df.reindex(columns=feature_names, fill_value=0)
    probs = model.predict_proba(input_df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = df.copy()
    results["churn_probability"] = probs.round(4)
    results["prediction"] = ["Churn" if p == 1 else "No Churn" for p in preds]
    results["risk_level"] = pd.cut(
        probs,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return results

if __name__ == "__main__":
    _, X_test, _, _, feature_names = load_and_preprocess()
    sample = X_test.iloc[0].to_dict()
    result = predict_single(sample)
    print(result)