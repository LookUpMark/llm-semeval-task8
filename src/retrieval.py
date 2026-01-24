"""
Retrieval Module for MTRAGEval.

Implements dense retrieval with cross-encoder reranking:
- Base Retrieval: Vector search returning top 20 candidates (high recall)
- Reranking: Cross-Encoder filters to top 5 (high precision)

Uses BGE-M3 for embeddings and a lightweight reranker for fast testing.
"""

from typing import List, Any, Optional
import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# CONFIGURATION
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Using lightweight reranker (~80MB) instead of BGE-Reranker-v2-M3 (2.2GB)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Use CPU for embeddings to save GPU memory for LLM
EMBEDDING_DEVICE = "cpu"
RERANKER_DEVICE = "cpu"

# Module-level singletons to avoid lock conflicts
_qdrant_client: Optional[QdrantClient] = None
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_cross_encoder: Optional[HuggingFaceCrossEncoder] = None


def get_qdrant_client(qdrant_path: str) -> QdrantClient:
    """Get or create singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=qdrant_path)
    return _qdrant_client


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Get or create singleton embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Get or create singleton cross-encoder."""
    global _cross_encoder
    if _cross_encoder is None:
        print(f"🔧 Loading reranker: {RERANKER_MODEL_NAME}")
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            model_kwargs={"device": RERANKER_DEVICE}
        )
    return _cross_encoder


def get_retriever(
    qdrant_path: str = "./qdrant_db",
    collection_name: str = "mtrag_collection",
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5
) -> ContextualCompressionRetriever:
    """
    Returns an advanced retriever: Dense Vector Search -> Cross-Encoder Rerank.
    
    Uses singleton clients to avoid Qdrant lock conflicts.
    
    Args:
        qdrant_path: Path to Qdrant database directory.
        collection_name: Name of the Qdrant collection.
        top_k_retrieve: Number of documents to retrieve before reranking (default 20).
        top_k_rerank: Number of documents to return after reranking (default 5).
    
    Returns:
        ContextualCompressionRetriever with reranking.
    """
    # Use singleton client (avoids lock issues)
    client = get_qdrant_client(qdrant_path)
    
    # Use singleton embedding model
    embedding_model = _get_embedding_model()
    
    # VectorStore: retrieval of top_k_retrieve documents (high recall)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
    
    # Use singleton cross-encoder
    cross_encoder = _get_cross_encoder()
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_k_rerank)
    
    # Final pipeline: Retriever + Reranker
    return ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )


def format_docs_for_gen(docs: List[Any]) -> str:
    """
    Extract PARENT TEXT from retrieved child chunks and deduplicate.
    
    The parent_text key is set by ingestion.py during Parent-Child chunking.
    """
    unique_contents = set()
    final_context: List[str] = []

    for doc in docs:
        # parent_text is set by ingestion.py (Parent-Child chunking)
        parent_text = doc.metadata.get("parent_text", doc.page_content)
        if parent_text not in unique_contents:
            unique_contents.add(parent_text)
            final_context.append(parent_text)

    return "\n\n".join(final_context)
