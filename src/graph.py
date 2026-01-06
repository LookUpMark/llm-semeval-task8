"""
Graph Orchestration Module for MTRAGEval.

Self-CRAG workflow with Retry Loop:
When hallucination is detected, retries generation up to MAX_RETRIES times
before falling back to "I_DONT_KNOW".
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from .state import GraphState
# from .retrieval import get_retriever, format_docs_for_gen
# from .generation import query_rewriter, generator, retrieval_grader, hallucination_grader

# Configuration
MAX_RETRIES = 2


# --- NODES ---

def rewrite_node(state: GraphState) -> dict:
    """Rewrites ambiguous queries to be standalone."""
    print("--- DEBUG: Execution rewrite_node ---")
    return {"standalone_question": state.get("question", "")}


def retrieve_node(state: GraphState) -> dict:
    """Searches Qdrant and resets retry_count to 0."""
    print("--- DEBUG: Execution retrieve_node ---")
    return {"documents": ["Dummy document 1", "Dummy document 2"], "retry_count": 0}


def grade_documents_node(state: GraphState) -> dict:
    """Filters irrelevant documents. Sets documents_relevant flag."""
    print("--- DEBUG: Execution grade_document_node ---")
    return {"documents_relevant": "yes"}


def generate_node(state: GraphState) -> dict:
    """Generates answer from filtered documents."""
    print("--- DEBUG: Execution generate_node ---")
    return {"generation": "Simulate response based on documents"}


def hallucination_check_node(state: GraphState) -> dict:
    """Checks if generation is grounded in documents."""
    print("--- DEBUG: Execution hallucination_check_node ---")
    return {"is_hallucination": "no"}


def increment_retry_node(state: GraphState) -> dict:
    """Increments retry_count before re-generation attempt."""
    print("--- DEBUG: Execution increment_retry_node ---")
    current_retry = state.get("retry_count", 0)
    return {"retry_count": current_retry + 1}

def fallback_node(state: GraphState) -> dict:
    """Returns I_DONT_KNOW when all else fails."""
    print("--- DEBUG: Execution fallback_node ---")
    return {"generation": "I_DONT_KNOW"}


# --- CONDITIONAL EDGES ---

def decide_to_generate(state: GraphState) -> str:
    """After grading: 'generate' if docs relevant, else 'fallback'."""
    relevant = state.get("documents_relevant", "no")
    if relevant == "yes":
        return "generate"
    return "fallback"


def decide_to_final(state: GraphState) -> str:
    """
    After hallucination check:
    - 'end' if grounded
    - 'increment_retry' if hallucinated but retries left
    - 'fallback' if max retries reached
    """
    is_hallucination = state.get("is_hallucination", "no")
    retry_count = state.get("retry_count", 0)
    
    if  is_hallucination == "no":
        return "end"
    
    if  retry_count < MAX_RETRIES:
        return "increment_retry"
    
    return "fallback"


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
    
    # 1. Init the graph with state
    workflow = StateGraph(GraphState)
    
    # 2. Registry the nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("fallback", fallback_node)
    
    # 3. Define initial linear flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    
    # 4. Add firs decision branch (CRAG)
    workflow.add_conditional_edges(
        "grade_docs", 
        decide_to_generate, 
        {
            "generate": "generate", 
            "fallback": "fallback"
        }
    )
    
    # 5. Add generation node and control
    workflow.add_edge("generate", "hallucination_check")
    
    # 6. Add second decision branch (Self-RAG with loop)
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_to_final, 
        {
            "end": END,
            "increment_retry": "increment_retry",
            "fallback": "fallback"
        }
    )
    
    # 7. Close the retry loop
    workflow.add_edge("increment_retry", "generate")
    workflow.add_edge("fallback", END)
    
    # 8. Compile the graph in an executable application
    return workflow.compile()


# Compiled app placeholder
app = None


def initialize_graph():
    """Initializes the graph. Call once at startup."""
    global app
    app = build_graph()
