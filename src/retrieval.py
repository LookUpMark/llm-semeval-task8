"""Hybrid Retrieval with Cross-Encoder Reranking."""

from typing import List, Any, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


# Configuration - CPU for embeddings/reranker to save GPU for LLM
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
EMBEDDING_DEVICE = "cpu"
RERANKER_DEVICE = "cpu"

# Singletons
_qdrant_client: Optional[QdrantClient] = None
_embedding_model: Optional[HuggingFaceEmbeddings] = None
_cross_encoder: Optional[HuggingFaceCrossEncoder] = None


def get_qdrant_client(qdrant_path: str) -> QdrantClient:
    """Singleton Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=qdrant_path)
    return _qdrant_client


def _get_embedding_model() -> HuggingFaceEmbeddings:
    """Singleton BGE-M3 embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embedding_model


def _get_cross_encoder() -> HuggingFaceCrossEncoder:
    """Singleton cross-encoder reranker."""
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading reranker: {RERANKER_MODEL_NAME}")
        _cross_encoder = HuggingFaceCrossEncoder(
            model_name=RERANKER_MODEL_NAME,
            model_kwargs={"device": RERANKER_DEVICE}
        )
    return _cross_encoder


def get_retriever(qdrant_path: str = "./qdrant_db", collection_name: str = "mtrag_collection",
                  top_k_retrieve: int = 20, top_k_rerank: int = 5, 
                  domain: str = None, use_filter: bool = True) -> ContextualCompressionRetriever:
    """
    Dense Vector Search -> Cross-Encoder Rerank pipeline.
    Retrieves top_k_retrieve candidates, reranks to top_k_rerank.
    If domain is specified and use_filter is True, filters results to that domain only.
    Uses metadata.source (set during ingestion) instead of metadata.domain.
    """
    client = get_qdrant_client(qdrant_path)
    embedding_model = _get_embedding_model()
    
    vectorstore = QdrantVectorStore(client=client, collection_name=collection_name, embedding=embedding_model)
    
    # Apply domain filter using metadata.source (from ingestion)
    search_kwargs = {"k": top_k_retrieve}
    if domain and use_filter:
        search_kwargs["filter"] = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=domain))]
        )
    
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    compressor = CrossEncoderReranker(model=_get_cross_encoder(), top_n=top_k_rerank)
    
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)


def format_docs_for_gen(docs: List[Any]) -> str:
    """Extract deduplicated parent texts from retrieved child chunks."""
    unique_contents = set()
    final_context = []
    
    for doc in docs:
        parent_text = doc.metadata.get("parent_text", doc.page_content)
        if parent_text not in unique_contents:
            unique_contents.add(parent_text)
            final_context.append(parent_text)
            
    return "\n\n".join(final_context)
