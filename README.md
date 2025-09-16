# On-Device RAG-Based Educational Assistant for Jetson Nano

This project implements a **Retrieval-Augmented Generation (RAG)** educational assistant that runs entirely **locally** on edge devices like Jetson Nano and on standard laptops. It provides an interactive interface for students to query uploaded educational materials while ensuring **privacy, reliability and feasibility in resource-constrained environments**.

---

## Features
- Local deployment: no cloud dependency
- Works on Jetson Nano and standard laptops
- RAG-based response generation grounded in user-provided documents
- Streamlit frontend for lightweight user interface
- FastAPI backend for API access to Llama.cpp server
- Multiple LLM models supported with quantization for edge deployment
- Fully offline: frameworks works offline with the downloaded models

--- 

## Requirements

- **Python**: 3.8 (Latest version supported by Jetson Nano)
- **JetPack**: 4.6.1 (Latest OS version for CUDA 10.2 support)
- **Llama.cpp**: build or install directly from [this gist](https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f)

---

### Models 

To run the project offline, download the following models from HuggingFace:

- **LLM Models**:
  - [Qwen 2.5 0.5B Instruct GGUF](https://huggingface.co/Qwen/Qwen-2.5-0.5B-Instruct-GGUF)
  - [Llama 3.2 1B Instruct GGUF](https://huggingface.co/MaziyarPanahi/Llama-3.2-1B-Instruct-GGUF)
  - [Gemma 3 1B IT GGUF](https://huggingface.co/MaziyarPanahi/gemma-3-1b-it-GGUF)
 
- **Embedding Model**:
  - [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

- **Reranker Model**:
  - [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
 
> After downloading, place the models in the `models/` folder

---

## Installation

1. **Clone the repository**
```bash
  git clone <https://github.com/jackiewaang/Jetson-Nano-RAG-LLM>
  cd <Jetson-Nano-RAG-LLM>
```
2. **Create a Python virtual environment and activate it**
```bash
  python3 -m venv venv
  source venv/bin/activate
```
3. **Install dependencies**
```bash
  pip install -r requirements.txt
```
4. **Ensure Llama.cpp server works**

---

## Running the Project

Start the project using the helper script:
```bash
  ./start.sh
```

The script will:
1. Prompt you to choose one of the available models:
   - Qwen 2.5
   - Llama 3.2
   - Gemma 3
2. Start the Streamlit frontend at http://localhost:8501
3. Start the FastAPI backend at http://localhost:8000
4. Start the Llama.cpp server at http://localhost:8080
