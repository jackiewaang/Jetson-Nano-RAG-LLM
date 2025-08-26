from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import re

class RAGPipeline:
    def __init__(self):
        # Load Embedding Function Model
        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Initialize ChromaDB with embedding function
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
                name="collection",
                embedding_function=self.embedding
        )
    
    # Clean and load text from given PDFs
    def load_pdfs(self, pdf_paths):
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        
        return documents

    # Clean document text by removing non-ASCII and whitespaces 
    def clean_text(self, text):
        text = re.sub(r'\s+',' ', text)
        text = text.encode('ascii', errors='ignore').decode()
        return text.strip()
    
    # Split documents content into chunks
    def chunk_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
        return text_splitter.split_documents(docs)

    # Store data into Vector DB
    def build_vector_store(self, chunk_docs):
        # Index docs for database
        documents = [doc.page_content for doc in chunk_docs]
        ids = [f"doc{i}" for i in range(len(documents))]

        self.collection.upsert(
                documents=documents,
                ids=ids
        )

    # Retrieve similar text from vector DB
    def retrieve(self, prompt):
        search_result = self.collection.query(
                query_texts=[prompt],
                n_results=3
        )
        return "\n\n".join(search_result["documents"][0])
