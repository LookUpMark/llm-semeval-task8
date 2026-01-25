"""
Data Ingestion Module for MTRAGEval.

This module handles loading mtRAG dataset JSON files and applies
Parent-Child Chunking strategy for optimal retrieval:
- Parent chunks: Large (1200 chars) for context
- Child chunks: Small (400 chars) for precise vector search

When a child chunk is retrieved, the parent content is returned to the LLM.
"""

import json
import uuid
from typing import List, Optional
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from src.retrieval import get_qdrant_client

# CONFIGURAZIONE
# BGE-M3 è SOTA per retrieval denso e funziona bene su CPU/GPU modeste.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Singleton instances
_embedding_model: Optional[HuggingFaceEmbeddings] = None


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Get or create singleton embedding model to save memory."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                "batch_size": 16,
                "normalize_embeddings": True
            }
        )
    return _embedding_model


def load_and_chunk_data(json_path: str):
    """
    Carica il dataset mtRAG e applica Parent-Child Chunking.
    """
    print(f"--- LOADING DATA FROM {json_path} ---") 
    # ESEMPIO: json_path potrebbe essere "./data/corpus.json" o simile.

    data = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {json_path} not found.")

    raw_docs = []
    # Adattamento alla struttura JSON di mtRAG
    for item in data:

        text = item.get("text", "").strip()

        if not text:
            continue

        source_name = Path(json_path).stem

        doc = Document(
            page_content=text,
            metadata={
                "doc_id": item.get("id", str(uuid.uuid4())),
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": source_name
            }
        )
        raw_docs.append(doc)
    
    print(f"Loaded {len(raw_docs)} documents.")

    # 1. Parent Splitter: Chunk grandi (1200 chars) per il contesto
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    
    # 2. Child Splitter: Chunk piccoli (400 chars) per la ricerca
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    docs_to_index = []

    print("--- STARTING PARENT-CHILD SPLITTING ---")
    for parent_doc in raw_docs:
        # Creiamo prima i blocchi grandi
        parent_chunks = parent_splitter.split_documents([parent_doc])
        
        for p_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            # Spezziamo il blocco grande in pezzi piccoli
            child_chunks = child_splitter.split_documents([p_chunk])
            
            for c_chunk in child_chunks:
                # TRUCCO: Salviamo il testo GRANDE nei metadati del piccolo
                c_chunk.metadata["parent_text"] = p_chunk.page_content
                c_chunk.metadata["parent_title"] = parent_doc.metadata["title"]
                c_chunk.metadata["parent_url"] = parent_doc.metadata["url"]
                c_chunk.metadata["parent_id"] = parent_id
                
                docs_to_index.append(c_chunk)
                
    return docs_to_index


def build_vector_store(docs: List[Document], persist_dir: str = "../qdrant_db", collection_name: str = "mtrag_collection"):
    """
    Crea e salva il database vettoriale Qdrant.
    Usa client singleton per evitare lock.
    """
    
    print(f"--- BUILDING VECTOR STORE: {collection_name} ---")
    
    # Singleton instances
    embedding_model = _get_embedding_model()
    # Use shared singleton client
    client = get_qdrant_client(persist_dir)

    # recreate = pulisce se esiste già
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_model.client.get_sentence_embedding_dimension(),
            distance=Distance.COSINE
        )
    )
    
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    batch_size = 64
    print(f"   Adding {len(docs)} documents in batches of {batch_size}...")
    
    from tqdm import tqdm
    for i in tqdm(range(0, len(docs), batch_size), desc="Indexing"):
        batch = docs[i : i + batch_size]
        vectorstore.add_documents(batch)

    print("--- VECTOR STORE BUILT AND SAVED ---")
    return vectorstore