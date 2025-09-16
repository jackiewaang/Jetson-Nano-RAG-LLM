#!/bin/bash

# Prompt for model selection
echo "Select the model to use:"
echo "1) Qwen 2.5 0.5B Instruct GGUF"
echo "2) Llama 3.2 1B Instruct GGUF"
echo "3) Gemma 3 1B Instruct GGUF"
read -p "Enter choice [1-3]: " model_choice

case $model_choice in
  1)
    model_name="Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"
    offline_name="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    ;;
  2)
    model_name="MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF:Q4_K_M"
    offline_name="Llama-3.2-1B-Instruct.Q4_K_M.gguf"
    ;;
  3)
    model_name="MaziyarPanahi/gemma-3-1b-it-GGUF:Q4_K_M"
    offline_name="gemma-3-1b-it.Q4_K_M.gguf"
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

# Prompt for online/offline mode
echo "Select mode:"
echo "1) Online"
echo "2) Offline"
read -p "Enter choice [1-2]: " mode_choice

case $mode_choice in
  1)
    llama_arg="-hf $model_name"
    ;;
  2) llama_arg="-m ./models/$offline_name"
    ;;
  *)
    echo "Invalid choice."
    exit 1
    ;;
esac

# Start FastAPI backend
echo "Starting FastAPI backend on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

sleep 2

# Start Streamlit frontend
echo "Starting Streamlit frontend on port 8501..."
streamlit run app.py &

# Start Llama.cpp server
echo "Starting Llama.cpp server on port 8080 with model $llama_arg..."
llama-server $llama_arg --n-gpu-layers 99
