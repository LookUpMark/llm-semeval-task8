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


def get_retriever(
    qdrant_path: str = "./qdrant_db",
    collection_name: str = "mtrag_collection",
    top_k_retrieve: int = 20,
    top_k_rerank: int = 5
) -> ContextualCompressionRetriever:
    """
    Returns an advanced retriever: Dense Vector Search -> Cross-Encoder Rerank.
    
    Args:
        qdrant_path: Path to Qdrant database directory.
        collection_name: Name of the Qdrant collection.
        top_k_retrieve: Number of documents to retrieve before reranking (default 20).
        top_k_rerank: Number of documents to return after reranking (default 5).
    
    Returns:
        ContextualCompressionRetriever with reranking.
    """
    # Qdrant connection
    client = QdrantClient(path=qdrant_path)
    
    # Embedding model (BGE-M3 dense)
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # VectorStore: retrieval of top_k_retrieve documents (high recall)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model
    )
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})
    
    # Cross-Encoder Reranker: filters top_k_rerank documents (high precision)
    cross_encoder = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME,
        model_kwargs={"device": DEVICE}
    )
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
