"""
Retrieval Module for MTRAGEval.

Implements dense retrieval with cross-encoder reranking:
- Base Retrieval: Vector search returning top 20 candidates (high recall)
- Reranking: Cross-Encoder filters to top 5 (high precision)

Uses BGE-M3 for embeddings and BGE-Reranker-v2-M3 for reranking.
Hybrid dense+sparse retrieval is discussed theoretically, but not implemented due to framework constraints.
"""

from typing import List, Any
import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# CONFIGURATION
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_retriever() -> ContextualCompressionRetriever:
    """
    Restituisce un retriever avanzato: Dense Vector Search (Top 20) -> Rerank (Top 5).
    """
    # Qdrant connection
    client = QdrantClient(path="./qdrant_db")
    
    # Embedding model (BGE-M3 dense)
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # VectorStore: retrieval of top 20 documents (high recall)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="mtrag_collection",
        embedding=embedding_model
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # Cross-Encoder Reranker: filters top 5 documents (high precision)
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        model_kwargs={"device": DEVICE}
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    
    # Final pipeline: Retriever + Reranker
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )


def format_docs_for_gen(docs: List[Any]) -> str:
    """
    Estrae il PARENT CONTENT dai child chunks recuperati e deduplica.
    """
    unique_contents = set()
    final_context: List[str] = []

    for doc in docs:
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        if parent_content not in unique_contents:
            unique_contents.add(parent_content)
            final_context.append(parent_content)

    return "\n\n".join(final_context)
