"""Self-CRAG Graph Orchestration with Retry Loop for MTRAGEval."""

import os
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from .state import GraphState
from .retrieval import get_retriever, format_docs_for_gen
from .generation import create_generation_components

# Configuration
MAX_RETRIES = 2
PROJECT_ROOT = (os.getcwd() if os.path.exists("src") else 
                os.path.join(os.getcwd(), "llm-semeval-task8") if os.path.exists("llm-semeval-task8") 
                else os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")

# Lazy-initialized singletons
_components = None
_retriever = None


def _get_components():
    global _components
    if _components is None:
        print("Loading LLM and generation components...")
        _components = create_generation_components()
    return _components


def _get_retriever_for_domain(domain: str):
    """Returns unified retriever (single collection for all domains)."""
    global _retriever
    if _retriever is None:
        print("Initializing unified retriever for collection: mtrag_unified")
        _retriever = get_retriever(qdrant_path=QDRANT_PATH, collection_name="mtrag_unified", 
                                   top_k_retrieve=20, top_k_rerank=5)
    return _retriever


# --- NODES ---

def rewrite_node(state: GraphState) -> dict:
    """Rewrites context-dependent questions to standalone form."""
    components = _get_components()
    question, messages = state.get("question", ""), state.get("messages", [])
    chat_history = [("human" if isinstance(m, HumanMessage) else "ai", m.content) for m in messages]
    
    try:
        result = components.query_rewriter.invoke({"question": question, "messages": chat_history})
        return {"standalone_question": result if isinstance(result, str) else str(result)}
    except Exception as e:
        print(f"Rewrite failed: {e}, using original question")
        return {"standalone_question": question}


def retrieve_node(state: GraphState) -> dict:
    """Searches Qdrant for relevant documents."""
    question = state.get("standalone_question") or state.get("question", "")
    try:
        documents = _get_retriever_for_domain(state.get("domain", "govt")).invoke(question)
    except Exception as e:
        print(f"Retrieval failed: {e}")
        documents = []
    return {"documents": documents, "retry_count": 0}


def grade_documents_node(state: GraphState) -> dict:
    """CRAG: filters irrelevant documents."""
    components = _get_components()
    documents, question = state.get("documents", []), state.get("standalone_question") or state.get("question", "")
    
    if not documents:
        return {"documents_relevant": "no", "documents": []}
    
    relevant_docs = []
    for doc in documents:
        try:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            result = components.retrieval_grader.invoke({"document": content, "question": question})
            if (result.get("binary_score", "no") if isinstance(result, dict) else "no") == "yes":
                relevant_docs.append(doc)
        except Exception as e:
            print(f"Grading error: {e}")
            relevant_docs.append(doc)  # Keep on error
    
    return {"documents_relevant": "yes" if relevant_docs else "no", "documents": relevant_docs}


def generate_node(state: GraphState) -> dict:
    """Generates answer from documents using Llama."""
    components = _get_components()
    context = format_docs_for_gen(state.get("documents", []))
    question = state.get("standalone_question") or state.get("question", "")
    
    try:
        result = components.generator.invoke({"context": context, "question": question})
        return {"generation": result if isinstance(result, str) else str(result)}
    except Exception as e:
        print(f"Generation failed: {e}")
        return {"generation": "I_DONT_KNOW"}


def hallucination_check_node(state: GraphState) -> dict:
    """Self-RAG: checks if generation is grounded in documents."""
    components = _get_components()
    documents, generation = state.get("documents", []), state.get("generation", "")
    
    if not documents or generation == "I_DONT_KNOW":
        return {"is_hallucination": "no"}
    
    try:
        result = components.hallucination_grader.invoke({
            "documents": format_docs_for_gen(documents), "generation": generation
        })
        score = result.get("binary_score", "yes") if isinstance(result, dict) else "yes"
        return {"is_hallucination": "no" if score == "yes" else "yes"}
    except Exception as e:
        print(f"Hallucination check failed: {e}")
        return {"is_hallucination": "no"}  # Assume grounded on error


def increment_retry_node(state: GraphState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def fallback_node(state: GraphState) -> dict:
    return {"generation": "I_DONT_KNOW"}


# --- CONDITIONAL EDGES ---

def decide_to_generate(state: GraphState) -> str:
    return "generate" if state.get("documents_relevant", "no") == "yes" else "fallback"


def decide_to_final(state: GraphState) -> str:
    if state.get("is_hallucination", "no") == "no":
        return "end"
    return "increment_retry" if state.get("retry_count", 0) < MAX_RETRIES else "fallback"


# --- GRAPH CONSTRUCTION ---

def build_graph() -> StateGraph:
    """Builds Self-CRAG graph: rewrite → retrieve → grade → generate → hallucination_check (with retry)."""
    workflow = StateGraph(GraphState)
    
    # Register nodes
    for name, fn in [("rewrite", rewrite_node), ("retrieve", retrieve_node), 
                     ("grade_docs", grade_documents_node), ("generate", generate_node),
                     ("hallucination_check", hallucination_check_node),
                     ("increment_retry", increment_retry_node), ("fallback", fallback_node)]:
        workflow.add_node(name, fn)
    
    # Linear flow
    workflow.add_edge(START, "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "grade_docs")
    workflow.add_conditional_edges("grade_docs", decide_to_generate, {"generate": "generate", "fallback": "fallback"})
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_conditional_edges("hallucination_check", decide_to_final, 
                                   {"end": END, "increment_retry": "increment_retry", "fallback": "fallback"})
    workflow.add_edge("increment_retry", "generate")
    workflow.add_edge("fallback", END)
    
    return workflow.compile()


app = None  # Compiled app placeholder

def initialize_graph():
    """Initializes the graph. Call once at startup."""
    global app
    app = build_graph()
    return app
