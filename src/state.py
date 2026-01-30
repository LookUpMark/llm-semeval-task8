"""State definition for Self-CRAG RAG Graph."""

from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph state for Self-CRAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    standalone_question: str
    documents: List[Any]
    generation: str
    documents_relevant: str
    is_hallucination: str
    domain: str
    retry_count: int
    fallback_reason: str  # irrelevant_docs | llm_refusal | hallucination_loop_exhausted | none
