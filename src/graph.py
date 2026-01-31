"""Self-CRAG Graph Orchestration with Retry Loop for MTRAGEval.

This module implements a Self-CRAG (Corrective Retrieval Augmented Generation) system
that combines retrieval, validation, and generation techniques to produce accurate responses.

Main workflow:
1. Rewrite: reformulates context-dependent questions into standalone questions
2. Retrieve: retrieves relevant documents from Qdrant
3. Grade: validates document relevance (CRAG)
4. Generate: generates the answer using the documents
5. Hallucination Check: verifies the answer is grounded in documents (Self-RAG)
6. Retry Loop: in case of hallucination, retries up to MAX_RETRIES times
"""

import os
from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage

from .state import GraphState
from .retrieval import get_retriever, format_docs_for_gen
from .generation import create_generation_components

# ============================================================================
# CONFIGURATION
# ============================================================================

# Maximum number of retries in case of hallucination
MAX_RETRIES = 2
# Dynamic determination of project root
PROJECT_ROOT = (os.getcwd() if os.path.exists("src") else 
                os.path.join(os.getcwd(), "llm-semeval-task8") if os.path.exists("llm-semeval-task8") 
                else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Path to Qdrant vector database
QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")

# ============================================================================
# GLOBAL SINGLETONS (lazy loading for performance optimization)
# ============================================================================

_components = None
_retriever = None


def _get_components():
    """Lazy initializes and returns generation components (LLM, graders, etc.)."""
    global _components
    if _components is None:
        print("Loading LLM and generation components...")
        _components = create_generation_components()
    return _components


def _get_retriever_for_domain(domain: str):
    """Returns retriever filtered by domain. Cached per-domain."""
    global _retriever
    if _retriever is None:
        _retriever = {}
    
    if domain not in _retriever:
        print(f"Initializing retriever for domain: {domain}")
        _retriever[domain] = get_retriever(
            qdrant_path=QDRANT_PATH, 
            collection_name="mtrag_unified", 
            top_k_retrieve=20,  # Broad initial retrieval
            top_k_rerank=5,     # Reranking for better precision
            domain=domain
        )
    return _retriever[domain]


# ============================================================================
# GRAPH NODES
# ============================================================================

def rewrite_node(state: GraphState) -> Dict[str, Any]:
    """Rewrites context-dependent questions to standalone form."""
    components = _get_components()
    question, messages = state.get("question", ""), state.get("messages", [])
    
    # Convert messages to (role, content) format for the rewriter
    chat_history = [("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in messages]
    
    try:
        result = components.query_rewriter.invoke({"question": question, "messages": chat_history})
        return {"standalone_question": result if isinstance(result, str) else str(result)}
    except Exception as e:
        print(f"Rewrite failed: {e}, using original question")
        # Fallback: use original question on error
        return {"standalone_question": question}


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Searches Qdrant for relevant documents."""
    question = state.get("standalone_question") or state.get("question", "")
    domain = state.get("domain", "govt")
    
    try:
        documents = _get_retriever_for_domain(domain).invoke(question)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        documents = []  # Empty list on error
    return {"documents": documents, "retry_count": 0}


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """CRAG: filters irrelevant documents."""
    components = _get_components()
    documents = state.get("documents", [])
    question = state.get("standalone_question") or state.get("question", "")
    
    # Case 1: No documents retrieved from retrieval
    if not documents:
        return {"documents_relevant": "no", "documents": [], "fallback_reason": "irrelevant_docs"}
    
    # Case 2: Evaluate each document with the grader
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
            relevant_docs.append(doc)  # On error, keep document
    
    # Case 3: Return evaluation result
    if relevant_docs:
        return {"documents_relevant": "yes", "documents": relevant_docs, "fallback_reason": "none"}
    else:
        return {"documents_relevant": "no", "documents": [], "fallback_reason": "irrelevant_docs"}


def generate_node(state: GraphState) -> Dict[str, Any]:
    """Generates answer from documents using Llama."""
    components = _get_components()
    context = format_docs_for_gen(state.get("documents", []))  # Merges documents into single text
    question = state.get("standalone_question") or state.get("question", "")
    
    try:
        result = components.generator.invoke({"context": context, "question": question})
        generation = result if isinstance(result, str) else str(result)
        
        # Detect if LLM explicitly refused to answer
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
    
    # Skip check if no documents or if LLM already said "I don't know"
    if not documents or "I_DONT_KNOW" in generation:
        return {"is_hallucination": "no"}  # Cannot hallucinate if admitting not knowing
    
    try:
        result = components.hallucination_grader.invoke({
            "documents": format_docs_for_gen(documents), 
            "generation": generation
        })
        score = result.get("binary_score", "yes") if isinstance(result, dict) else "yes"
        # binary_score='yes' → grounded (not hallucination)
        # binary_score='no' → not grounded (is hallucination)
        return {"is_hallucination": "no" if score == "yes" else "yes"}
    except Exception as e:
        print(f"Hallucination check failed: {e}")
        return {"is_hallucination": "no"}


def increment_retry_node(state: GraphState) -> dict:
    """Increments the retry counter in the retry loop."""
    return {"retry_count": state.get("retry_count", 0) + 1}


def fallback_node(state: GraphState) -> Dict[str, Any]:
    """Returns I_DONT_KNOW with appropriate reason."""
    reason = state.get("fallback_reason")
    if not reason or reason == "none":
        # Determine reason based on state
        reason = "hallucination_loop_exhausted" if (state.get("is_hallucination") == "yes" and state.get("retry_count", 0) >= MAX_RETRIES) else "fallback_triggered"
    return {"generation": "I_DONT_KNOW", "fallback_reason": reason}


# ============================================================================
# CONDITIONAL EDGES (routing logic)
# ============================================================================

def decide_to_generate(state: GraphState) -> str:
    """Decides whether to proceed with generation or go to fallback."""
    return "generate" if state.get("documents_relevant", "no") == "yes" else "fallback"


def decide_to_final(state: GraphState) -> str:
    """Decides whether to terminate, retry, or go to fallback after hallucination check."""
    if state.get("is_hallucination", "no") == "no":
        return "end"
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "increment_retry"
    return "fallback"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

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
    
    # Conditional edges (dynamic routing)
    workflow.add_conditional_edges("grade_docs", decide_to_generate, 
                                   {"generate": "generate", "fallback": "fallback"})
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", decide_to_final, 
                                   {"end": END, "increment_retry": "increment_retry", "fallback": "fallback"})
    
    # Retry loop: from increment_retry back to generate
    workflow.add_edge("increment_retry", "generate")
    
    # Termination: fallback always leads to END
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================

# Global variable for compiled graph
app = None

def initialize_graph():
    """Initializes the graph singleton."""
    global app
    app = build_graph()
    return app
