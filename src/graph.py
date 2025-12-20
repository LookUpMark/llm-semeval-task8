"""
Graph Orchestration Module for MTRAGEval.

Self-CRAG workflow with Retry Loop:
When hallucination is detected, retries generation up to MAX_RETRIES times
before falling back to "I_DONT_KNOW".
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from src.state import GraphState
from src.generation import query_rewriter, generator, retrieval_grader, hallucination_grader
from src.retrieval import get_retriever, format_docs_for_gen

# Configuration
MAX_RETRIES = 2


# --- NODES ---

def rewrite_node(state: GraphState) -> dict:
    """Rewrites ambiguous queries to be standalone."""
    raise NotImplementedError("rewrite_node")


def retrieve_node(state: GraphState) -> dict:
    """Searches Qdrant and resets retry_count to 0."""
    raise NotImplementedError("retrieve_node")


def grade_documents_node(state: GraphState) -> dict:
    """Filters irrelevant documents. Sets documents_relevant flag."""
    raise NotImplementedError("grade_documents_node")


def generate_node(state: GraphState) -> dict:
    """Generates answer from filtered documents."""
    raise NotImplementedError("generate_node")


def hallucination_check_node(state: GraphState) -> dict:
    """Checks if generation is grounded in documents."""
    raise NotImplementedError("hallucination_check_node")


def increment_retry_node(state: GraphState) -> dict:
    """Increments retry_count before re-generation attempt."""
    raise NotImplementedError("increment_retry_node")


def fallback_node(state: GraphState) -> dict:
    """Returns I_DONT_KNOW when all else fails."""
    raise NotImplementedError("fallback_node")


# --- CONDITIONAL EDGES ---

def decide_to_generate(state: GraphState) -> str:
    """After grading: 'generate' if docs relevant, else 'fallback'."""
    raise NotImplementedError("decide_to_generate")


def decide_to_final(state: GraphState) -> str:
    """
    After hallucination check:
    - 'end' if grounded
    - 'increment_retry' if hallucinated but retries left
    - 'fallback' if max retries reached
    """
    raise NotImplementedError("decide_to_final")


# --- GRAPH CONSTRUCTION ---

def build_graph() -> StateGraph:
    """
    Builds Self-CRAG graph with retry loop.
    
    Flow:
    START -> rewrite -> retrieve -> grade_docs
    -> [generate if docs OK, else fallback]
    -> hallucination_check
    -> [end if grounded, increment_retry if retries left, fallback if max]
    -> increment_retry loops back to generate
    """
    raise NotImplementedError("build_graph")


# Compiled app placeholder
app = None


def initialize_graph():
    """Initializes the graph. Call once at startup."""
    global app
    app = build_graph()
