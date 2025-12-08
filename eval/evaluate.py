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
RAGAS_EMBEDDING_MODEL = "BAAI/bge-m3"

# Placeholder for RAGAS embeddings
ragas_embeddings = None  # type: ignore


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
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError(
        "run_evaluation: Implement RAGAS evaluation pipeline with local Llama judge"
    )


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
