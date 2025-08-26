from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import requests

print("Loading the Embedding model")
# Load Embedding Function
#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Loading the PDF document")
# Load PDF document
pdf_path = "/home/jetbot/rag2/uploads/TheRoleofLightweightLLMslikeGemmainEnhancingLow.pdf" 
loader = PyPDFLoader(pdf_path)
docs = loader.load()

print("Splitting document into chunks")
# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512)
chunk_docs = text_splitter.split_documents(docs)
documents = [doc.page_content for doc in chunk_docs]
ids = [f"doc{i}" for i in range(len(documents))]

print("Storing text into vector db")
# Store in Vector DB
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
        name="collection",
        embedding_function=emb_func
)

collection.upsert(
        documents=documents,
        ids=ids
)

query = "Can you summarize the main findings of the given PDF?"

print("Performing similarity search")
# Retrieve Top K from Vector DB
search_result = collection.query(
        query_texts=[query],
        n_results=10
)
retrieved_text = search_result["documents"][0]
context = "\n\n".join(retrieved_text)

rag_prompt = f"Answer the question based on the following information. \n\n{context}\n\nQuestion: {query}"

print(rag_prompt)

print("Sending RAG prompt to Ollama")

r = requests.post("http://localhost:11434/api/generate",
        json={
            "model": "gemma3-270m",
            "prompt": rag_prompt,
            "stream": False
        }
)

print(r.json())
