"""Self-CRAG Graph Orchestration with Retry Loop."""

import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage

from .state import GraphState
from .retrieval import get_retriever, format_docs_for_gen
from .generation import create_generation_components


# Configuration
MAX_RETRIES = 2

# Robust Project Root Logic
if os.path.exists("src"):
    PROJECT_ROOT = os.getcwd()
elif os.path.exists("llm-semeval-task8"):
    PROJECT_ROOT = os.path.join(os.getcwd(), "llm-semeval-task8")
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")


# Lazy-initialized singletons
_components = None
_retriever = None


def _get_components():
    """Lazy load generation components."""
    global _components
    if _components is None:
        print("Loading LLM and generation components...")
        _components = create_generation_components()
    return _components


def _get_retriever_for_domain(domain: str, use_filter: bool = True):
    """Returns retriever filtered by domain. Cached per-domain and filter setting."""
    global _retriever
    if _retriever is None:
        _retriever = {}
    
    cache_key = f"{domain}_{use_filter}"
    if cache_key not in _retriever:
        print(f"Initializing retriever for domain: {domain}, use_filter: {use_filter}")
        _retriever[cache_key] = get_retriever(
            qdrant_path=QDRANT_PATH, 
            collection_name="mtrag_unified", 
            top_k_retrieve=20, 
            top_k_rerank=5,
            domain=domain,
            use_filter=use_filter
        )
    return _retriever[cache_key]


# --- NODES ---

def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """Rewrites context-dependent questions to standalone form."""
    components = _get_components()
    question, messages = state.get("question", ""), state.get("messages", [])
    
    # Format history for prompt
    chat_history = [("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in messages]
    
    try:
        result = components.query_rewriter.invoke({"question": question, "messages": chat_history})
        return {"standalone_question": result if isinstance(result, str) else str(result)}
    except Exception as e:
        print(f"Rewrite failed: {e}, using original question")
        return {"standalone_question": question}


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Searches Qdrant for relevant documents with fallback to unfiltered search."""
    standalone_q = state.get("standalone_question") or state.get("question", "")
    original_q = state.get("question", "")
    domain = state.get("domain", "govt")
    
    documents = []
    
    # Try filtered retrieval first
    try:
        documents = _get_retriever_for_domain(domain, use_filter=True).invoke(standalone_q)
    except Exception as e:
        print(f"Filtered retrieval failed: {e}")
    
    # Fallback 1: if no docs, try with original question (filtered)
    if not documents and original_q != standalone_q:
        try:
            print("Fallback: trying original question with filter")
            documents = _get_retriever_for_domain(domain, use_filter=True).invoke(original_q)
        except Exception as e:
            print(f"Original question retrieval failed: {e}")
    
    # Fallback 2: if still no docs, try without domain filter
    if not documents:
        try:
            print("Fallback: trying without domain filter")
            documents = _get_retriever_for_domain(domain, use_filter=False).invoke(standalone_q)
        except Exception as e:
            print(f"Unfiltered retrieval failed: {e}")
        
    return {"documents": documents, "retry_count": 0}


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """CRAG: filters irrelevant documents."""
    components = _get_components()
    documents = state.get("documents", [])
    question = state.get("standalone_question") or state.get("question", "")
    
    if not documents:
        return {"documents_relevant": "no", "documents": [], "fallback_reason": "irrelevant_docs"}
    
    relevant_docs = []
    for doc in documents:
        try:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            result = components.retrieval_grader.invoke({"document": content, "question": question})
            score = result.get("binary_score", "no") if isinstance(result, dict) else "no"
            
            if score == "yes":
                relevant_docs.append(doc)
        except Exception as e:
            print(f"Grading error: {e}")
            relevant_docs.append(doc)  # Fail open on error
    
    if relevant_docs:
        return {"documents_relevant": "yes", "documents": relevant_docs, "fallback_reason": "none"}
    else:
        return {"documents_relevant": "no", "documents": [], "fallback_reason": "irrelevant_docs"}


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generates answer from documents using Llama."""
    components = _get_components()
    context = format_docs_for_gen(state.get("documents", []))
    question = state.get("standalone_question") or state.get("question", "")
    
    try:
        result = components.generator.invoke({"context": context, "question": question})
        generation = result if isinstance(result, str) else str(result)
        
        fallback_reason = "none"
        if "I_DONT_KNOW" in generation:
            fallback_reason = "llm_refusal"
            
        return {"generation": generation, "fallback_reason": fallback_reason}
    except Exception as e:
        print(f"Generation failed: {e}")
        return {"generation": "I_DONT_KNOW", "fallback_reason": "generation_error"}


def hallucination_check_node(state: GraphState) -> Dict[str, Any]:
    """Self-RAG: checks if generation is grounded in documents."""
    components = _get_components()
    documents = state.get("documents", [])
    generation = state.get("generation", "")
    
    if not documents or "I_DONT_KNOW" in generation:
        return {"is_hallucination": "no"}
    
    try:
        result = components.hallucination_grader.invoke({
            "documents": format_docs_for_gen(documents), 
            "generation": generation
        })
        score = result.get("binary_score", "yes") if isinstance(result, dict) else "yes"
        return {"is_hallucination": "no" if score == "yes" else "yes"}
    except Exception as e:
        print(f"Hallucination check failed: {e}")
        return {"is_hallucination": "no"}


def increment_retry_node(state: GraphState) -> Dict[str, Any]:
    return {"retry_count": state.get("retry_count", 0) + 1}


def fallback_node(state: GraphState) -> Dict[str, Any]:
    """Returns I_DONT_KNOW with appropriate reason."""
    reason = state.get("fallback_reason")
    if not reason or reason == "none":
        # Determine strict reason for generic fallback
        if state.get("is_hallucination") == "yes" and state.get("retry_count", 0) >= MAX_RETRIES:
            reason = "hallucination_loop_exhausted"
        else:
            reason = "fallback_triggered"
            
    return {"generation": "I_DONT_KNOW", "fallback_reason": reason}


# --- CONDITIONAL EDGES ---

def decide_to_generate(state: GraphState) -> str:
    return "generate" if state.get("documents_relevant", "no") == "yes" else "fallback"


def decide_to_final(state: GraphState) -> str:
    if state.get("is_hallucination", "no") == "no":
        return "end"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "increment_retry"
    return "fallback"


# --- GRAPH CONSTRUCTION ---

def build_graph() -> StateGraph:
    """Builds Self-CRAG graph: rewrite -> retrieve -> grade -> generate -> hallucination_check."""
    workflow = StateGraph(GraphState)
    
    # Register nodes
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_docs", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("fallback", fallback_node)
    
    # Define Flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_conditional_edges(
        "grade_docs", 
        decide_to_generate, 
        {"generate": "generate", "fallback": "fallback"}
    )
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check", 
        decide_to_final, 
        {"end": END, "increment_retry": "increment_retry", "fallback": "fallback"}
    )
    workflow.add_edge("increment_retry", "generate")
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


app = None

def initialize_graph():
    """Initializes the graph singleton."""
    global app
    app = build_graph()
    return app
