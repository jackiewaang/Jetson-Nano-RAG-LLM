from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from typing import List
import requests

app = FastAPI()
rag = RAGPipeline()

llama_endpoint = "http://localhost:8080/v1/chat/completions"

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float

class RAGRequest(ChatRequest):
    k: int
    n: int

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

@app.post("/chat")
def chat(req: ChatRequest):
    print("Sending prompt directly to Llama server...")
    r = requests.post(
            llama_endpoint,
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": req.prompt},
            ],
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        },
    )
    print("Answer received...")
    return r.json()

@app.post("/")
def generate(req: RAGRequest):
    print("Retrieving similar text...")
    retrieved_text = rag.retrieve(req.prompt, req.k, req.n)
    print(retrieved_text)
    
    system_prompt = """
        You are a helpful assistant. Use the retrieved context as your main source. Rephrase in your own words and combine relevant parts. If the context is incomplete, you may add general knowledge for obvious facts. If the answer isn't covered at all, say so.
    """
    
    user_prompt = f"""
    Retrieved Context: {retrieved_text}

    Question: {req.prompt}

    Answer:
    """
    print("Sending prompt to Llama server...")
    r = requests.post(
            llama_endpoint,
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            }
    ) 
    print("Answer received...")
    print(r.json())
    return r.json()
