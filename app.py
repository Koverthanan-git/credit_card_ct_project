import pandas as pd
from fastapi import FastAPI
import mlflow
import mlflow.xgboost
# FIX Bug 3: mlflow.pyfunc.load_model() returns a PyFuncModel which has no predict_proba().
# Using mlflow.xgboost.load_model() returns the native XGBClassifier with full sklearn API.

app = FastAPI(title="Credit Card Inference Gateway")
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def _load_latest_model() -> object:
    """Resolve the highest registered version of CreditCardXGB and load it.
    MLflow 3.x deprecated the 'latest' stage alias — we query the registry
    client directly to find the current max version number."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions("name='CreditCardXGB'")
    if not versions:
        raise RuntimeError("No versions of CreditCardXGB found in the registry.")
    latest_version = max(versions, key=lambda v: int(v.version))
    model_uri = f"models:/CreditCardXGB/{latest_version.version}"
    print(f"Loading model: {model_uri}")
    return mlflow.xgboost.load_model(model_uri)


# Load once at startup
model = _load_latest_model()


@app.post("/predict")
def predict(transaction: dict):
    """Accept a JSON transaction dict, return fraud prediction and probability."""
    # Convert incoming transaction JSON dictionary into a DataFrame format
    input_df = pd.DataFrame([transaction])

    # Generate prediction matrix using native XGBClassifier
    prediction  = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return {
        "is_fraud": int(prediction[0]),
        "fraud_probability": float(probability[0])
    }


if __name__ == "__main__":
    import uvicorn
    # Start your production server on port 8001
    uvicorn.run(app, host="127.0.0.1", port=8001)