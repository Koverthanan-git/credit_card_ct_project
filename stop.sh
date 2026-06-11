#!/bin/zsh

echo "Stopping Credit Card CT Project Services..."

# Function to kill process by port
kill_port() {
  local port=$1
  local service_name=$2
  
  # Find PID listening on the port
  local pid=$(lsof -ti :$port 2>/dev/null)
  
  if [ -n "$pid" ]; then
    echo "Stopping $service_name (Port $port, PID $pid)..."
    # Kill the process and its children if any
    kill -9 $pid 2>/dev/null
    echo "✅ $service_name stopped."
  else
    echo "⚠️  $service_name is not running (Port $port is free)."
  fi
}

kill_port 5000 "MLflow"
kill_port 8001 "FastAPI"
kill_port 8000 "Traefik"

echo ""
echo "All services have been shut down."
