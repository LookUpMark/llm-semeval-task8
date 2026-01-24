"""
State definition for the Self-CRAG RAG Graph.

This module defines GraphState, the TypedDict that travels through
the LangGraph workflow. The key `messages` uses the `add_messages`
reducer to maintain conversation history across graph nodes.
"""

from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """
    Represents the state of our RAG graph.
    Each node receives this state as input and returns a dict
    with the keys it wants to update.
    
    Attributes:
        messages: The conversation history (HumanMessages + AIMessages).
                  Annotated[list, add_messages] means: "When a node returns 'messages',
                  append to the existing list instead of overwriting it".
        question: The user's original question in the current turn.
        standalone_question: The question "rewritten" by the system to be understandable
                             without context. E.g.: "How much does it cost?" -> "How much does IBM Cloud service cost?"
        documents: The list of documents retrieved from the Vector Store.
        generation: The final generated response.
        documents_relevant: Flag ('yes' or 'no') - Whether the grader says the retrieved documents are useful.
        is_hallucination: Flag ('yes' or 'no') - Whether the grader says the response is fabricated.
        retry_count: Counter to prevent infinite correction loops.
    """
    
    # The conversation history (HumanMessages + AIMessages).
    messages: Annotated[List[BaseMessage], add_messages]
    
    # The user's original question in the current turn.
    question: str
    
    # The question "rewritten" by the system to be understandable without context.
    standalone_question: str
    
    # The list of documents retrieved from the Vector Store.
    documents: List[Any]
    
    # The final generated response.
    generation: str
    
    # Boolean flags for graph decision logic.
    documents_relevant: str  # 'yes' or 'no'
    
    is_hallucination: str    # 'yes' or 'no'
    
    # Domain for corpus-specific retrieval (e.g., 'govt', 'clapnq', 'fiqa', 'cloud')
    domain: str
    
    # Counter to prevent infinite correction loops.
    retry_count: int

