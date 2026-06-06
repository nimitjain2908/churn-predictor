import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
from src.preprocess import load_and_preprocess

def train():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess()

    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        params = {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 50,
            "max_depth": -1,
            "min_child_samples": 15,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }

        mlflow.log_params(params)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"AUC:      {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

        os.makedirs("outputs", exist_ok=True)
        joblib.dump(model, "outputs/churn_model.pkl")
        mlflow.lightgbm.log_model(model, "model")

        print("Model saved to outputs/churn_model.pkl")

if __name__ == "__main__":
    train()