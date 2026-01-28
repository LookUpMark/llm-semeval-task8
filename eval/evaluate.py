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
from src.generation import create_generation_components
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize LLM only once
_gen_components = None
def get_llm():
    global _gen_components
    if _gen_components is None:
        _gen_components = create_generation_components()
    return _gen_components.llm


# Embeddings configuration for RAGAS
RAGAS_EMBEDDING_MODEL = "BAAI/bge-m3"

ragas_embeddings = None


def initialize_ragas_embeddings():
    """
    Initialize HuggingFace embeddings for RAGAS evaluation.
    """
    global ragas_embeddings
    if ragas_embeddings is None:
        print(f"--- INITIALIZING EVALUATION EMBEDDINGS ({RAGAS_EMBEDDING_MODEL}) ---")
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=RAGAS_EMBEDDING_MODEL,
            model_kwargs={'device': 'cuda'} 
        )
    return ragas_embeddings



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
    """
    data = {
        "question": questions,
        "answer": answers,
        "context": contexts,
        "ground_truth": ground_truths
    }
    return Dataset.from_dict(data)



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

    eval_embeddings = initialize_ragas_embeddings()

    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("--- STARTING EVALUATION ---")
    
    # RAGAS expects a Dataset object with specific columns
    # We construct matching lists from the input data
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, item in enumerate(test_dataset):
        q = item['question']
        gt = item['ground_truth']
        ans = item.get('answer', "")
        ctx = item.get('contexts', [])
        
        # Ensure contexts is a list of strings
        if not isinstance(ctx, list):
            ctx = [str(ctx)] if ctx else []
            
        questions.append(q)
        answers.append(ans)
        contexts_list.append(ctx)
        ground_truths.append(gt)

    hf_dataset = create_evaluation_dataset(questions, answers, contexts_list, ground_truths)

    # Passiamo esplicitamente llm e embeddings a Ragas
    # NOTA: Usare un LLM da 8B come giudice può essere impreciso, 
    # ma è l'unica opzione in ambiente fully offline/open.
    scores = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=get_llm(),
        embeddings=eval_embeddings
    )
    
    print("--- EVALUATION RESULTS ---")
    print(scores)
    return scores



def load_and_evaluate(conversations_path: str, submission_path: str):
    """
    Loads conversations and submission file, links them via 'original_index',
    and runs evaluation.
    """
    import json
    import os
    
    print(f"Loading conversations from {conversations_path}...")
    with open(conversations_path, 'r') as f:
        all_conversations = json.load(f)
        
    print(f"Loading submission from {submission_path}...")
    submission_data = []
    with open(submission_path, 'r') as f:
        for line in f:
            if line.strip():
                submission_data.append(json.loads(line))
                
    print(f"Linking {len(submission_data)} generated results with ground truth...")
    
    eval_dataset = []
    
    for item in submission_data:
        # 1. Get original conversation
        idx = item.get("original_index")
        if idx is None:
            print(f"Warning: Item {item.get('conversation_id')} missing 'original_index'. Skipping.")
            continue
            
        if idx >= len(all_conversations):
            print(f"Warning: Index {idx} out of bounds (max {len(all_conversations)-1}). Skipping.")
            continue
            
        conv = all_conversations[idx]
        msgs = conv.get("messages", [])
        
        # 2. Extract Question and Ground Truth
        # Assumption: The 'question' was the last user message.
        # Assumption: The 'ground_truth' is the LAST assistant message in the history.
        # NOTE: This assumes the conversation log INCLUDES the correct answer.
        
        last_user_msg = next((m["text"] for m in reversed(msgs) if m.get("speaker") == "user"), None)
        last_agent_msg = next((m["text"] for m in reversed(msgs) if m.get("speaker") != "user"), None)
        
        if not last_user_msg:
             continue
             
        # 3. Extract Contexts (References)
        # Task C output has "references" which is list of strings or dicts?
        # In pipeline we saved: "references": contexts (list of strings)
        refs = item.get("references", [])
        # If it's Task B, we might not have references, or just "answer".
        # If references is missing, RAGAS context_precision will likely fail or score 0.
        
        eval_dataset.append({
            "question": last_user_msg,
            "answer": item.get("answer", ""),
            "contexts": refs,
            "ground_truth": last_agent_msg if last_agent_msg else "I don't know"
        })
        
    print(f"Prepared {len(eval_dataset)} items for evaluation.")
    return run_evaluation(eval_dataset)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG Submission")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission .jsonl file")
    parser.add_argument("--conversations", type=str, 
                        default="/home/marcantoniolopez/Documenti/github/projects/llm-semeval-task8/dataset/human/conversations/conversations.json",
                        help="Path to original conversations.json")
    
    args = parser.parse_args()
    
    results = load_and_evaluate(args.conversations, args.submission)

    # Stampa formattata per debug
    import pandas as pd
    df = pd.DataFrame(results)
    print("\nDetailed Results:")
    print(df[['faithfulness', 'answer_relevancy', 'context_precision']])
