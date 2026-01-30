"""State definition for Self-CRAG RAG Graph."""

from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    LangGraph state traveling through Self-CRAG workflow.
    
    - messages: Conversation history (add_messages reducer appends new messages)
    - question/standalone_question: Original and context-independent query
    - documents: Retrieved docs from vector store
    - generation: Final LLM response
    - documents_relevant/is_hallucination: Grading flags ('yes'/'no')
    - domain: Corpus domain (govt, clapnq, fiqa, cloud)
    - retry_count: Hallucination retry counter (max 2)
    """
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    standalone_question: str
    documents: List[Any]
    generation: str
    documents_relevant: str  # 'yes' or 'no'
    is_hallucination: str    # 'yes' or 'no'
    domain: str
    retry_count: int
