"""
Evaluation Package for MTRAGEval.

This package contains evaluation utilities using RAGAS metrics
with the local Llama 3.1 model as the judge.
"""

from eval.evaluate import run_evaluation

__all__ = ["run_evaluation"]
