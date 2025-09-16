from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from rag_pipeline import RAGPipeline
from typing import List
import requests

# Initialize FastAPI app and RAG pipeline
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

# Endpoint to upload and process PDF files
@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    pdf_paths = []
    # Save uploaded files to uploads directory
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

# Endpoint for direct chat without retrieval
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

# Endpoint for RAG-based generation
@app.post("/")
def generate(req: RAGRequest):
    print("Retrieving similar text...")
    retrieved_result = rag.retrieve(req.prompt, req.k, req.n)
    retrieved_text = retrieved_result["content"]
    sources_used = retrieved_result["sources"]

    print("Retrieved text:", retrieved_text)
    print("Sources used:", sources_used)

    # System prompt for Llama model
    system_prompt = """
        You are a helpful assistant. Use the retrieved context as your main source. Rephrase in your own words and combine relevant parts. If the context is incomplete, you may add general knowledge for obvious facts. If the answer isn't covered at all, say so.
    """
    
    # User prompt including retrieved context
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
    
    response_data = r.json()

    # Process sources to get unique filenames
    source_files = list(set([source["source"] for source in sources_used]))

    # Add sources used to the response
    response_data["sources_used"] = {
            "files": source_files,
            "detailed_sources": sources_used
    }

    print("Response with sources", response_data)
    return response_data
