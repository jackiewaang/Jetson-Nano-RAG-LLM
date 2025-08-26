from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from typing import List
import requests

app = FastAPI()
rag = RAGPipeline()

ollama_endpoint = "http://localhost:11434/api/generate"

class Request(BaseModel):
    model: str
    prompt: str
    stream: bool

@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    pdf_paths = []
    for file in files:
        path = f"./uploads/{file.filename}"
        with open(path, "wb")as f:
            f.write(await file.read())
        pdf_paths.append(path)
    print("Loading pdfs...")
    docs = rag.load_pdfs(pdf_paths)
    print("Chunking documents...")
    chunks = rag.chunk_documents(docs)
    print("Building vector store...")
    rag.build_vector_store(chunks)
    print("Files uploaded successfully...")
    return {"message": f"{len(files)} PDFs uploaded and processed."}

@app.post("/")
def generate(req: Request):
    print("Retrieving similar text...")
    retrieved_text = rag.retrieve(req.prompt)
    print(retrieved_text)
    
    rag_prompt = f"""
    Use the retrieved context as your main source. Rephrase in your own words and combine relevant parts.
    If the context is incomplete, you may add general knowledge for obvious facts.
    If the answer isn't covered at all, say so.

    Retrieved Context: {retrieved_text}

    Question: {req.prompt}

    Answer:
    """
    print("Sending prompt to Ollama server...")
    r = requests.post(
            ollama_endpoint,
            json={
                "model": req.model,
                "prompt": rag_prompt,
                "stream": False
            }
    ) 
    print("Answer received...")
    print(r.json())
    return r.json()
