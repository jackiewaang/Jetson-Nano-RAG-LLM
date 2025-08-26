#!/bin/bash

echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

sleep 2

echo "Starting Streamlit frontend on port 8501..."
streamlit run app.py
