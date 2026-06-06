import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess(path="data/telco_churn.csv"):
    df = pd.read_csv(path)

    # Drop customerID — it's just an identifier, not a feature
    df.drop(columns=["customerID"], inplace=True)

    # TotalCharges has some blank strings — convert to numeric and drop those rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Encode target column: Yes → 1, No → 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Identify categorical columns and label encode them
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split into features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split — 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()