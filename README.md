# Multi-Turn RAG Evaluation (MTRAGEval)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Framework](https://img.shields.io/badge/LangGraph-Self--CRAG-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-University_Project-purple.svg)

## Project Overview

This repository contains the implementation of a Self-Corrective Retrieval-Augmented Generation (Self-CRAG) system designed for the SemEval 2026 Task 8 challenge (Multi-Turn Retrieval-Augmented Generation). The project demonstrates a production-grade approach to complex question answering using local Large Language Models (LLMs) and advanced retrieval strategies.

### The "Sovereign" Strategy
Our solution is designed around three strict constraints:
1.  **Fully Offline**: No external API calls (e.g., GPT-4, Claude).
2.  **Open-Source Only**: All models are publicly available on HuggingFace.
3.  **Consumer Hardware**: Deployable on consumer-grade GPUs (e.g., NVIDIA T4) with limited VRAM.

These constraints drive every technical decision, including 4-bit quantization and CPU offloading for embeddings.

## Team Members

*   **Marc'Antonio Lopez**
*   **Filippo Simone Iannello**
*   **Nicolò Colle**
*   **Carmine Benvenuto**
*   **Elia Cola**

*Department of Control and Computer Engineering, Politecnico di Torino*

## System Architecture

The core architecture is built upon LangGraph, orchestrating a cyclic state machine that enables the system to reflect on its own outputs and correct them dynamically.

### Core Components

1.  **Self-CRAG Workflow**
    The system implements the Self-CRAG framework which integrates two critical feedback loops:
    *   **Corrective RAG (CRAG)**: Evaluates the relevance of retrieved documents before generation. If retrieved documents are deemed irrelevant, the system triggers a fallback mechanism.
    *   **Self-RAG**: Evaluates the generated answer for hallucinations and grounding. If the answer is not supported by the context, the system initiates a retry loop to regenerate the response.

2.  **Hybrid Retrieval Pipeline**
    To maximize recall and precision, the system employs a two-stage retrieval process:
    *   **Dense Retrieval**: Utilizes `BAAI/bge-m3` embeddings stored in a Qdrant vector database to retrieve the top-k candidates.
    *   **Reranking**: Applies a Cross-Encoder (`BAAI/bge-reranker-v2-m3`) to re-score and re-order the candidates, filtering out noise.

3.  **Parent-Child Chunking Strategy**
    To balance semantic search precision with contextual coherence for the LLM:
    *   **Child Chunks**: Small segments (400 characters) optimised for dense vector matching.
    *   **Parent Chunks**: Larger segments (1200 characters) retrieved via the child's metadata, providing the LLM with sufficient context to answer complex queries.

4.  **Local LLM Inference**
    The system uses **Llama 3.1 8B Instruct**, quantization to 4-bit (NF4) using `bitsandbytes` to ensure efficient inference on consumer-grade hardware (e.g., NVIDIA T4, 16GB VRAM), ensuring complete data privacy and offline capability.

## Project Structure

The codebase is organized as follows:

```
.
├── src/                    # Source Code
│   ├── graph.py            # Main application logic and LangGraph workflow implementation
│   ├── generation.py       # LLM initialization, prompt templates, and chain definitions
│   ├── retrieval.py        # Retrieval logic including embedding and reranking
│   ├── ingestion.py        # Data processing, chunking, and vector database indexing
│   └── state.py            # TypedDict definition for the graph state
├── notebooks/
│   └── All_Tasks_Pipeline.ipynb  # Primary entry point for executing the complete pipeline
├── tests/                  # Verification Notebooks
│   ├── verify_ingestion.ipynb    # Validates data parsing and chunking logic
│   ├── verify_retrieval.ipynb    # Tests retrieval metrics (Recall, Precision)
│   ├── verify_graph.py           # Validates graph state transitions
│   └── verify_self_crag.ipynb    # Tests the corrective feedback loops
├── eval/                   # Evaluation Utilities
│   └── evaluate.py         # RAGAS framework integration for metric calculation
├── data/
│   └── submissions/        # Output directory for generated results
└── requirements.txt        # Python dependencies
```

## Results

The system's performance was evaluated across three key tasks: Retrieval (Task A), Generation (Task B), and Multi-Turn RAG (Task C).



### Task C: RAG Performance (Self-CRAG)
Results on the Development Set (110 conversations), evaluated using the offline judge protocol.

**Overall Outcome:**
| Outcome | Count | Percentage |
| :--- | :---: | :---: |
| **Answered** | 70 | **63.6%** |
| **I_DONT_KNOW** | 40 | **36.4%** |
| **Total** | 110 | 100% |

**Fallback Analysis:**
| Reason | Count | Description |
| :--- | :---: | :--- |
| **irrelevant_docs** | 16 | All retrieved documents failed the relevance grader. |
| **hallucination_loop** | 12 | Generated answers failed grounding checks after retries. |
| **llm_refusal** | 12 | The LLM explicitly declined to answer. |

**Judge Assessment:**
*   **Hallucination Rate**: ~87% (High strictness mismatch between agent and judge).
*   **Refusal Validation**: ~40% of refusals were confirmed as valid data gaps (Context Missing), while others were false negatives (Agent too conservative).

### Hardware Impact & Limitations
The reported performance is directly influenced by the strict **hardware constraints** imposed by the evaluation environment (Single T4 GPU).
*   **Quantization Effects**: The use of **4-bit NF4 quantization** for Llama 3.1 8B introduces reasoning bottlenecks, particularly in multi-hop deduction and synonym matching, contributing to false negative refusals.
*   **Model Capacity**: The 8B parameter limit restricts the model's ability to maintain long-context coherence compared to larger models (e.g., 70B), necessitating aggressive self-correction loops that occasionally exhaust the retry limit.
*   **Judge vs. Agent Gap**: A significant portion of the "Hallucination" score stems from the capacity gap between the runtime agent (8B, quantized) and the offline judge (14B, unquantized), where the judge penalizes valid but simplified inferences.

---

## Evaluation Method

To rigorously validate the system without external APIs, we implemented a custom model-based evaluation framework.

*   **Judge LLM**: `Qwen/Qwen2.5-14B-Instruct`
*   **Metric**: Faithfulness and Refusal Justification (Model-Based)
*   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity

The evaluation logic is implemented in `eval/evaluate.ipynb` and focuses on:
1.  **Refusal Accuracy**: Whether `I_DONT_KNOW` responses are justified by context absence.
2.  **Faithfulness**: Whether generated answers are grounded in the retrieved documents (Self-RAG style).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.