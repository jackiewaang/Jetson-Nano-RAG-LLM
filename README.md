Ensure Ollama is running locally at http://localhost:11434

Activate virtual environment
source venv/bin/activate

If first time then install dependencies
pip install -r requirements.txt

Run FastAPI backend locally at port 8000
uvicorn main:app --host 0.0.0.0 --port 8000

Run Streamlit frontend locally at port 8501
streamlit run app.py
