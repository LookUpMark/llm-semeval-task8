"""RAGAS Evaluation Module for MTRAGEval using local Llama 3.1 as judge."""

import json
from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.generation import create_generation_components

# Lazy-initialized singletons
_gen_components, _ragas_embeddings = None, None
RAGAS_EMBEDDING_MODEL = "BAAI/bge-m3"


def get_llm():
    global _gen_components
    if _gen_components is None:
        _gen_components = create_generation_components()
    return _gen_components.llm


def initialize_ragas_embeddings():
    global _ragas_embeddings
    if _ragas_embeddings is None:
        print(f"--- INITIALIZING EVALUATION EMBEDDINGS ({RAGAS_EMBEDDING_MODEL}) ---")
        _ragas_embeddings = HuggingFaceEmbeddings(model_name=RAGAS_EMBEDDING_MODEL, model_kwargs={'device': 'cuda'})
    return _ragas_embeddings


def create_evaluation_dataset(questions: List[str], answers: List[str], 
                               contexts: List[List[str]], ground_truths: List[str]) -> Dataset:
    """Creates HuggingFace Dataset for RAGAS evaluation."""
    return Dataset.from_dict({
        "question": questions, "answer": answers, "context": contexts, "ground_truth": ground_truths
    })


def run_evaluation(test_dataset: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Runs RAGAS evaluation (faithfulness, answer_relevancy, context_precision).
    Uses local Llama 3.1 8B as judge - accuracy limited by model size.
    """
    eval_embeddings = initialize_ragas_embeddings()
    
    questions, answers, contexts_list, ground_truths = [], [], [], []
    for item in test_dataset:
        questions.append(item['question'])
        answers.append(item.get('answer', ""))
        ctx = item.get('contexts', [])
        contexts_list.append([str(ctx)] if not isinstance(ctx, list) else ctx)
        ground_truths.append(item['ground_truth'])

    hf_dataset = create_evaluation_dataset(questions, answers, contexts_list, ground_truths)
    
    print("--- RUNNING RAGAS EVALUATION ---")
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
    """Links submission file with original conversations via 'original_index' and evaluates."""
    print(f"Loading conversations from {conversations_path}...")
    with open(conversations_path, 'r') as f:
        all_conversations = json.load(f)
        
    print(f"Loading submission from {submission_path}...")
    submission_data = []
    with open(submission_path, 'r') as f:
        for line in f:
            if line.strip():
                submission_data.append(json.loads(line))
    
    print(f"Linking {len(submission_data)} results with ground truth...")
    eval_dataset = []
    
    for item in submission_data:
        idx = item.get("original_index")
        if idx is None:
            print(f"Warning: Item {item.get('conversation_id')} missing 'original_index'. Skipping.")
            continue
        if idx >= len(all_conversations):
            print(f"Warning: Index {idx} out of bounds (max {len(all_conversations)-1}). Skipping.")
            continue
            
        msgs = all_conversations[idx].get("messages", [])
        last_user = next((m["text"] for m in reversed(msgs) if m.get("speaker") == "user"), None)
        last_agent = next((m["text"] for m in reversed(msgs) if m.get("speaker") != "user"), None)
        
        if last_user:
            eval_dataset.append({
                "question": last_user,
                "answer": item.get("answer", ""),
                "contexts": item.get("references", []),
                "ground_truth": last_agent or "I don't know"
            })
        
    print(f"Prepared {len(eval_dataset)} items for evaluation.")
    return run_evaluation(eval_dataset)


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="Evaluate RAG Submission")
    parser.add_argument("--submission", type=str, required=True)
    parser.add_argument("--conversations", type=str, 
                        default="/home/marcantoniolopez/Documenti/github/projects/llm-semeval-task8/dataset/human/conversations/conversations.json")
    args = parser.parse_args()
    
    results = load_and_evaluate(args.conversations, args.submission)
    print("\nDetailed Results:")
    print(pd.DataFrame(results)[['faithfulness', 'answer_relevancy', 'context_precision']])
