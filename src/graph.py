"""
Graph Orchestration Module for MTRAGEval.

Self-CRAG workflow with Retry Loop:
When hallucination is detected, retries generation up to MAX_RETRIES times
before falling back to "I_DONT_KNOW".

This module integrates:
- Retrieval: Qdrant vector search with BGE-M3 + cross-encoder reranking
- Generation: Llama 3.1 8B Instruct (4-bit quantized)
- Grading: Document relevance and hallucination detection chains
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from .state import GraphState
from .retrieval import get_retriever, format_docs_for_gen
from .generation import create_generation_components

# Configuration
MAX_RETRIES = 2
import os
# Fix: Use absolute path for Qdrant to avoid issues when running from subdirectories (like tests/)
if os.path.exists("src"):
    PROJECT_ROOT = os.getcwd()
elif os.path.exists("llm-semeval-task8"):
    PROJECT_ROOT = os.path.join(os.getcwd(), "llm-semeval-task8")
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")

# Module-level components (initialized lazily)
_components = None
_retriever = None
_current_domain = None


def _get_components():
    """Lazy initialization of generation components."""
    global _components
    if _components is None:
        print("🔧 Loading LLM and generation components...")
        _components = create_generation_components()
    return _components


def _get_retriever_for_domain(domain: str):
    """Get or create retriever for specific domain."""
    global _retriever, _current_domain
    
    # SIMPLIFICATION: Use the UNIFIED collection for all domains
    # This matches All_Tasks_Pipeline.ipynb logic
    collection_name = "mtrag_unified"
    
    # Reuse if already initialized (regardless of domain, since it's unified)
    if _retriever is not None:
        return _retriever
    
    print(f"🔍 Initializing unified retriever for collection: {collection_name}")
    _retriever = get_retriever(
        qdrant_path=QDRANT_PATH,
        collection_name=collection_name,
        top_k_retrieve=20,
        top_k_rerank=5
    )
    # _current_domain is no longer needed since we use one collection
    return _retriever


# --- NODES ---

def rewrite_node(state: GraphState) -> dict:
    """Rewrites context-dependent questions to standalone form."""
    components = _get_components()
    question = state.get("question", "")
    messages = state.get("messages", [])
    
    # Format chat history for rewriter
    chat_history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            chat_history.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            chat_history.append(("ai", msg.content))
    
    try:
        result = components.query_rewriter.invoke({
            "question": question,
            "messages": chat_history
        })
        standalone = result if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"⚠️ Rewrite failed: {e}, using original question")
        standalone = question
    
    return {"standalone_question": standalone}


def retrieve_node(state: GraphState) -> dict:
    """Searches Qdrant for relevant documents."""
    question = state.get("standalone_question") or state.get("question", "")
    domain = state.get("domain", "govt")  # Default to govt if not specified
    
    try:
        retriever = _get_retriever_for_domain(domain)
        documents = retriever.invoke(question)
    except Exception as e:
        print(f"⚠️ Retrieval failed: {e}")
        documents = []
    
    return {"documents": documents, "retry_count": 0}


def grade_documents_node(state: GraphState) -> dict:
    """Filters irrelevant documents using CRAG grading."""
    components = _get_components()
    documents = state.get("documents", [])
    question = state.get("standalone_question") or state.get("question", "")
    
    if not documents:
        return {"documents_relevant": "no", "documents": []}
    
    # Grade each document
    relevant_docs = []
    for doc in documents:
        try:
            doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            result = components.retrieval_grader.invoke({
                "document": doc_content,
                "question": question
            })
            # Parse result
            score = result.get("binary_score", "no") if isinstance(result, dict) else "no"
            if score == "yes":
                relevant_docs.append(doc)
        except Exception as e:
            print(f"⚠️ Grading error: {e}")
            relevant_docs.append(doc)  # Keep on error
    
    has_relevant = "yes" if relevant_docs else "no"
    return {"documents_relevant": has_relevant, "documents": relevant_docs}


def generate_node(state: GraphState) -> dict:
    """Generates answer from documents using Llama."""
    components = _get_components()
    documents = state.get("documents", [])
    question = state.get("standalone_question") or state.get("question", "")
    
    # Format context from documents
    context = format_docs_for_gen(documents) if documents else ""
    
    try:
        result = components.generator.invoke({
            "context": context,
            "question": question
        })
        generation = result if isinstance(result, str) else str(result)
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        generation = "I_DONT_KNOW"
    
    return {"generation": generation}


def hallucination_check_node(state: GraphState) -> dict:
    """Checks if generation is grounded in documents."""
    components = _get_components()
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    
    if not documents or generation == "I_DONT_KNOW":
        return {"is_hallucination": "no"}
    
    # Format documents for grader
    doc_text = format_docs_for_gen(documents)
    
    try:
        result = components.hallucination_grader.invoke({
            "documents": doc_text,
            "generation": generation
        })
        # "yes" = grounded (not hallucinated), "no" = hallucinated
        score = result.get("binary_score", "yes") if isinstance(result, dict) else "yes"
        is_hallucination = "no" if score == "yes" else "yes"
    except Exception as e:
        print(f"⚠️ Hallucination check failed: {e}")
        is_hallucination = "no"  # Assume grounded on error
    
    return {"is_hallucination": is_hallucination}


def increment_retry_node(state: GraphState) -> dict:
    """Increments retry_count before re-generation attempt."""
    current_retry = state.get("retry_count", 0)
    return {"retry_count": current_retry + 1}


def fallback_node(state: GraphState) -> dict:
    """Returns I_DONT_KNOW when all else fails."""
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
    
    if is_hallucination == "no":
        return "end"
    
    if retry_count < MAX_RETRIES:
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
    
    workflow = StateGraph(GraphState)
    
    # Register nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("fallback", fallback_node)
    
    # Linear flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    
    # CRAG decision
    workflow.add_conditional_edges(
        "grade_docs", 
        decide_to_generate, 
        {"generate": "generate", "fallback": "fallback"}
    )
    
    workflow.add_edge("generate", "hallucination_check")
    
    # Self-RAG decision with retry loop
    workflow.add_conditional_edges(
        "hallucination_check",
        decide_to_final, 
        {"end": END, "increment_retry": "increment_retry", "fallback": "fallback"}
    )
    
    workflow.add_edge("increment_retry", "generate")
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


# Compiled app placeholder
app = None


def initialize_graph():
    """Initializes the graph. Call once at startup. Returns the compiled app."""
    global app
    app = build_graph()
    return app
