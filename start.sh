#!/bin/zsh

echo "Starting Credit Card CT Project Services..."

# 1. Activate Virtual Environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# 2. Start MLflow
echo "Starting MLflow on port 5000..."
nohup mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts > logs/mlflow.log 2>&1 &

# Wait for MLflow to initialize
sleep 4

# 3. Start FastAPI
echo "Starting FastAPI on port 8001..."
nohup python app.py > logs/fastapi.log 2>&1 &

# Wait for FastAPI to initialize
sleep 2

# 4. Start Traefik
echo "Starting Traefik on port 8000..."
nohup traefik --configFile=config/traefik.yaml > logs/traefik.log 2>&1 &

echo ""
echo "✅ All services started successfully in the background!"
echo "Logs are available in the ./logs/ directory."
echo ""
echo "Useful endpoints:"
echo "- Gateway (Traefik):  http://127.0.0.1:8000"
echo "- Traefik Dashboard:  http://127.0.0.1:8080/dashboard/"
echo "- FastAPI Docs:       http://127.0.0.1:8001/docs"
echo "- MLflow UI:          http://127.0.0.1:5000"
