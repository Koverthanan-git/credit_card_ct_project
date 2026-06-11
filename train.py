import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Connect to your local running MLflow instance
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Credit_Card_Fraud_CT")

def get_data_and_train():
    # Synthetic generator: Replace this line with your actual creditcard.csv download
    # e.g., df = pd.read_csv("creditcard.csv")
    np.random.seed(42)
    data = np.random.randn(1000, 30) # 30 standard credit card transaction features
    labels = np.random.choice([0, 1], size=1000, p=[0.98, 0.02]) # 2% fraud rate
    df = pd.DataFrame(data, columns=[f"V{i}" for i in range(1, 31)])
    df['Class'] = labels
    
    # Save a slice to act as reference data for future drift tracking
    df.to_csv("reference_data.csv", index=False)
    
    X = df.drop(columns=['Class'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # Train XGBoost Classifier
        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
        model.fit(X_train, y_train)
        
        # Log evaluation metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("max_depth", 4)
        
        # Log and register model inside the registry
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model",
            registered_model_name="CreditCardXGB"
        )
        print(f"Model successfully registered. Accuracy: {acc:.4f}")

if __name__ == "__main__":
    get_data_and_train()