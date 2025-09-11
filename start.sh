#!/bin/bash

echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

sleep 2

echo "Starting Streamlit frontend on port 8501..."
#streamlit run app.py &

echo "Starting Llama.cpp server on port 8080..."
llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M --n-gpu-layers 99
