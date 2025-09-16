from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils import embedding_functions
import re

class RAGPipeline:
    def __init__(self, chroma_db_path="./chroma_db"):
        # Initialize embedding function and cross-encoder model
        self.embedding = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

        # Initialize ChromaDB client and collection for vector database
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="collection",
            embedding_function=self.embedding
        )
        self.doc_counter = 0 
    
    # Load PDF files and extract documents with metadata.
    def load_pdfs(self, pdf_paths):
        documents = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = path
                doc.metadata["page"] = doc.metadata.get("page", 0) + 1
            documents.extend(docs)
        
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        
        return documents

    # Clean document text by removing non-ASCII and whitespaces 
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.encode('ascii', errors='ignore').decode()
        return text.strip()

    # Split documents content into chunks
    def chunk_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, 
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(docs)

        # Add unique IDs and preserve metadata
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                chunk.metadata['chunk_id'] = i

        return chunks
    
    # Store data into Vector DB
    def build_vector_store(self, chunk_docs):
        # Index docs for database
        documents = []
        metadatas = []
        ids = []
        
        for i, doc in enumerate(chunk_docs):
            documents.append(doc.page_content)
            ids.append(f"doc_{self.doc_counter}_{i}")

            metadata = {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", i)
            }
            metadatas.append(metadata)

        self.doc_counter += len(documents)

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    # Rerank retrieved documents using cross-encoder
    def rerank(self, query, docs, metadatas):
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        scores = self.model.predict(pairs)
        ranked_results = sorted(
            zip(scores, docs, metadatas), 
            key=lambda x: x[0], 
            reverse=True
        )
        return ranked_results

    # Retrieve similar text chunks from vector DB and rerank them
    def retrieve(self, prompt, k=5, n=3):
        # Query database for top-k similar segments
        search_result = self.collection.query(
            query_texts=[prompt],
            n_results=k,
            include=["documents", "metadatas"]
        )

        candidate_docs = search_result["documents"][0] if search_result["documents"] else []
        candidate_metadatas = search_result["metadatas"][0] if search_result["metadatas"] else []
        distances = search_result["distances"][0] if search_result["distances"] else []

        # Rerank candidate documents
        reranked_docs = self.rerank(prompt, candidate_docs, candidate_metadatas)

        # Select top-n documents after reranking
        top_n_results = reranked_docs[:n] if reranked_docs else []

        top_docs = [doc for _, doc, _ in top_n_results]
        top_sources = [meta for _, _, meta in top_n_results]

        return {
            "content": "\n\n".join(top_docs),
            "sources": top_sources
        }