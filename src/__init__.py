"""
SemEval 2026 Task 8 - MTRAGEval: Multi-Turn RAG Agentic System

This package contains the core components of the Self-CRAG implementation:
- state: GraphState definition for the agentic workflow
- ingestion: Data loading and Parent-Child chunking
- retrieval: Hybrid retrieval with reranking
- generation: Llama 3.1 based generation and grading
- graph: LangGraph workflow orchestration
"""

# from src.state import GraphState
# from src.ingestion import load_and_chunk_data, build_vector_store
# from src.retrieval import get_retriever, format_docs_for_gen
# from src.generation import (
#     get_llama_pipeline,
#     query_rewriter,
#     generator,
#     retrieval_grader,
#     hallucination_grader
# )
# from src.graph import app

__all__ = [
    "GraphState",
    "load_and_chunk_data",
    "build_vector_store",
    "get_retriever",
    "format_docs_for_gen",
    "get_llama_pipeline",
    "query_rewriter",
    "generator",
    "retrieval_grader",
    "hallucination_grader",
    "app",
]
