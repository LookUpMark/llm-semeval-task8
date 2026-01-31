"""State definition for Self-CRAG RAG Graph."""

from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph state for Self-CRAG workflow."""
    
    # Dialog Context
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    standalone_question: str
    domain: str
    
    # RAG State
    documents: List[Any]  # List[Document]
    generation: str
    
    # Control Flags
    documents_relevant: str  # "yes" | "no"
    is_hallucination: str    # "yes" | "no"
    retry_count: int
    fallback_reason: str     # "irrelevant_docs" | "llm_refusal" | "hallucination_loop_exhausted" | "none"
