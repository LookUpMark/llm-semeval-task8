"""
Evaluation Module for MTRAGEval.

This module provides evaluation utilities using RAGAS metrics:
- Faithfulness: How well is the answer grounded in the context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: How precise is the retrieved context?

Uses the local Llama 3.1 model as the judge (instead of OpenAI).
"""

from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_core.messages import HumanMessage
from src.graph import app
from src.generation import llm
from langchain_community.embeddings import HuggingFaceEmbeddings

# Embeddings configuration for RAGAS
ragas_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


def initialize_ragas_embeddings():
    """
    Initialize HuggingFace embeddings for RAGAS evaluation.
    
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("initialize_ragas_embeddings: Initialize embeddings for RAGAS")


def run_evaluation(test_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Run evaluation using Llama 3.1 as the Judge Model.
    Runs each question through the Self-CRAG pipeline and evaluates
    using RAGAS metrics: faithfulness, answer_relevancy, context_precision.
    
    Args:
        test_dataset: List of dicts with 'question' and 'ground_truth' keys.
        
    Returns:
        Dict containing RAGAS evaluation scores.
        
    Example:
        test_data = [
            {"question": "Who is the CEO of Apple?", "ground_truth": "Tim Cook"}
        ]
        scores = run_evaluation(test_data)
    """

    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("--- STARTING EVALUATION ---")
    for item in test_dataset:
        q = item['question']
        gt = item['ground_truth']
        
        inputs = {"question": q, "messages": [HumanMessage(content=q)]}
        output_state = app.invoke(inputs)
        
        generated_answer = output_state["generation"]
        # Estrai il testo puro dai documenti
        retrieved_docs_content = [doc.page_content for doc in output_state.get("documents", [])]
        
        results_data["question"].append(q)
        results_data["answer"].append(generated_answer)
        results_data["contexts"].append(retrieved_docs_content)
        results_data["ground_truth"].append(gt)

    hf_dataset = Dataset.from_dict(results_data)
    
    # Passiamo esplicitamente llm e embeddings a Ragas
    # NOTA: Usare un LLM da 8B come giudice può essere impreciso, 
    # ma è l'unica opzione in ambiente fully offline/open.
    scores = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm,
        embeddings=ragas_embeddings
    )
    
    print("--- EVALUATION RESULTS ---")
    print(scores)
    return scores



def load_test_data(json_path: str):
    """
    Carica il dataset di validazione per Task C (RAG).
    RICHIEDE: Il file presente in /dataset/val_data.json
    """
    import json
    
    # ESEMPIO: Adattare alla struttura reale del file JSON
    # ESEMPIO: Adattare alla struttura reale del file JSON
    # mtRAG usa JSONL (JSON Lines). Ogni riga è un task.
    # Struttura attesa da 'reference+RAG.jsonl':
    # { "turn_id": "...", "messages": [...], "reference_passages": [...] }
    
    test_data = []
    with open(json_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            
            # Estraiamo l'ultima domanda (user) e la risposta attesa (se presente)
            # O più semplicemente, per il Task C, usiamo le conversazioni fornite.
            # Qui semplifichiamo estraendo l'ultimo messaggio user come query.
            if "messages" in item:
                messages = item["messages"]
                last_user_msg = next((m for m in reversed(messages) if m['role'] == 'user'), None)
                last_agent_msg = next((m for m in reversed(messages) if m['role'] == 'assistant'), None) # O ground truth separata
                
                if last_user_msg:
                    test_data.append({
                        "question": last_user_msg['content'],
                        # In reference+RAG.jsonl la GT spesso è nel campo 'response' o va inferita.
                        # Per ora usiamo un placeholder o l'ultimo agent msg se presente nel dataset di training.
                        "ground_truth": last_agent_msg['content'] if last_agent_msg else "N/A"
                    })
        
    print(f"Caricati {len(test_data)} campioni di test.")
    return test_data



def create_evaluation_dataset(
    questions: List[str],
    answers: List[str], 
    contexts: List[List[str]],
    ground_truths: List[str]
) -> Dataset:
    """
    Creates a HuggingFace Dataset for RAGAS evaluation.
    
    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of retrieved context lists.
        ground_truths: List of ground truth answers.
        
    Returns:
        HuggingFace Dataset ready for RAGAS evaluation.
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("create_evaluation_dataset: Create HF Dataset from evaluation data")


def evaluate_single_turn(question: str, ground_truth: str) -> Dict[str, Any]:
    """
    Evaluate a single question-answer pair.
    
    Args:
        question: The input question.
        ground_truth: The expected answer.
        
    Returns:
        Dict with evaluation metrics for this single turn.
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("evaluate_single_turn: Implement single-turn evaluation")


if __name__ == "__main__":
    # Example usage
    test_data = [
        {"question": "Who is the CEO of Apple?", "ground_truth": "Tim Cook"}
    ]
    run_evaluation(test_data)
