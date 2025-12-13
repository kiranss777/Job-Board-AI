#!/bin/bash

# Script to run FastAPI and Streamlit services
# Usage: ./run_services.sh [streamlit|fastapi|both]

# Activate virtual environment
source venv/bin/activate

# Get the service to run (default: both)
SERVICE=${1:-both}

if [ "$SERVICE" == "streamlit" ]; then
    echo "Starting Streamlit..."
    streamlit run app.py
elif [ "$SERVICE" == "fastapi" ]; then
    echo "Starting FastAPI..."
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
elif [ "$SERVICE" == "both" ]; then
    echo "Starting both services..."
    echo "FastAPI will run on http://localhost:8000"
    echo "Streamlit will run on http://localhost:8501"
    echo ""
    echo "Starting FastAPI in background..."
    uvicorn api:app --reload --host 0.0.0.0 --port 8000 &
    FASTAPI_PID=$!
    echo "FastAPI started with PID: $FASTAPI_PID"
    echo ""
    echo "Starting Streamlit..."
    streamlit run app.py
    # Kill FastAPI when Streamlit exits
    kill $FASTAPI_PID 2>/dev/null
else
    echo "Usage: ./run_services.sh [streamlit|fastapi|both]"
    exit 1
fi

