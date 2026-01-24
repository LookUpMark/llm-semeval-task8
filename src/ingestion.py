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
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# CONFIGURATION
# BGE-M3 is SOTA for dense retrieval and works well on modest CPU/GPU.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"


def load_and_chunk_data(json_path: str) -> List[Document]:
    """
    Load the mtRAG dataset and apply Parent-Child Chunking.
    
    Args:
        json_path: Path to the JSON file containing mtRAG documents.
        
    Returns:
        List of child Documents with parent content in metadata.
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("load_and_chunk_data: Implement data loading and parent-child chunking logic")


def build_vector_store(docs: List[Document], persist_dir: str = "./qdrant_db") -> QdrantVectorStore:
    """
    Create and persist the Qdrant vector database.
    
    Args:
        docs: List of Documents to index.
        persist_dir: Directory where Qdrant DB will be persisted.
        
    Returns:
        QdrantVectorStore instance.
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("build_vector_store: Implement Qdrant vector store creation and persistence")
