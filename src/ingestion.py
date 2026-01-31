"""Data Ingestion with Parent-Child Chunking."""

import json
import uuid
from typing import List, Optional
from pathlib import Path

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings

from .retrieval import get_qdrant_client


# Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Singleton embedding model (CPU) to save memory."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={"batch_size": 16, "normalize_embeddings": True}
        )
    return _embedding_model


def load_and_chunk_data(json_path: str) -> List[Document]:
    """Load JSONL corpus and apply Parent-Child chunking (1200/400 chars)."""
    print(f"--- LOADING DATA FROM {json_path} ---")
    
    data = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {json_path} not found.")
    
    raw_docs = []
    source_name = Path(json_path).stem
    for item in data:
        text = item.get("text", "").strip()
        if text:
            raw_docs.append(Document(
                page_content=text,
                metadata={
                    "doc_id": item.get("id", str(uuid.uuid4())),
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "source": source_name
                }
            ))
    print(f"Loaded {len(raw_docs)} documents.")
    
    # Parent-Child splitting
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    print("--- STARTING PARENT-CHILD SPLITTING ---")
    docs_to_index = []
    for parent_doc in raw_docs:
        for p_chunk in parent_splitter.split_documents([parent_doc]):
            parent_id = str(uuid.uuid4())
            for c_chunk in child_splitter.split_documents([p_chunk]):
                # Store parent content in child metadata
                c_chunk.metadata.update({
                    "parent_text": p_chunk.page_content,
                    "parent_title": parent_doc.metadata["title"],
                    "parent_url": parent_doc.metadata["url"],
                    "parent_id": parent_id
                })
                docs_to_index.append(c_chunk)
    
    return docs_to_index


def build_vector_store(docs: List[Document], persist_dir: str = "../qdrant_db", 
                       collection_name: str = "mtrag_collection") -> QdrantVectorStore:
    """Create and persist Qdrant vector store."""
    print(f"--- BUILDING VECTOR STORE: {collection_name} ---")
    
    embedding_model = _get_embedding_model()
    client = get_qdrant_client(persist_dir)
    
    # Recreate collection
    embedding_dim = len(embedding_model.embed_query("test"))
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    
    vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model)
    
    # Batch indexing
    batch_size = 64
    print(f"   Adding {len(docs)} documents in batches of {batch_size}...")
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
        vectorstore.add_documents(docs[i:i + batch_size])
    
    print("--- VECTOR STORE BUILT AND SAVED ---")
    return vectorstore