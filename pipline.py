import xgboost as xgb
import pandas as pd
import numpy as np  # FIX Bug 4: np.random.choice() used in retrain_model() but numpy was never imported
import mlflow
from prefect import task, flow
from monitor import evaluate_production_drift

mlflow.set_tracking_uri("http://127.0.0.1:5000")

@task(name="Check Data Drift Metric")
def check_drift_status():
    drift_status, new_data = evaluate_production_drift()
    return drift_status, new_data

@task(name="Execute Automated Model Retraining")
def retrain_model(new_data_df: pd.DataFrame):
    print("⚠️ Retraining trigger initialized due to structural data shifts.")
    
    # Simulate collecting updated operational target labels for the drifted dataset
    new_data_df['Class'] = np.random.choice([0, 1], size=len(new_data_df), p=[0.95, 0.05])
    
    X = new_data_df.drop(columns=['Class'])
    y = new_data_df['Class']
    
    with mlflow.start_run():
        updated_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
        updated_model.fit(X, y)
        
        # Log and register updated weights to existing production deployment tags
        mlflow.xgboost.log_model(
            xgb_model=updated_model, 
            artifact_path="model",
            registered_model_name="CreditCardXGB"
        )
    print("✅ Retrained model version uploaded and registered to MLflow UI successfully.")

@flow(name="Self-Correcting Continuous Training Loop")
def continuous_training_flow():
    drift_flag, production_data = check_drift_status()
    
    if drift_flag:
        retrain_model(production_data)
    else:
        print("📊 Data profiles remain highly stable. Skipping update execution cycle.")

if __name__ == "__main__":
    # Execute the self-correcting loop locally
    continuous_training_flow()