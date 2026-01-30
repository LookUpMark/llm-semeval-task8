# Multi-Turn RAG Evaluation (MTRAGEval)

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Framework](https://img.shields.io/badge/LangGraph-Self--CRAG-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-University_Project-purple.svg)

This repository contains the implementation of a Self-Corrective Retrieval-Augmented Generation (Self-CRAG) system designed for the SemEval 2026 Task 8 challenge (Multi-Turn Retrieval-Augmented Generation). The project demonstrates a production-grade approach to complex question answering using local Large Language Models (LLMs) and advanced retrieval strategies.

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

## Getting Started

### Prerequisites

*   Python 3.12 or higher
*   NVIDIA GPU with at least 15GB VRAM (for 4-bit quantization)
*   CUDA Toolkit 12.x

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/llm-semeval-task8.git
    cd llm-semeval-task8
    ```

2.  **Set Up Virtual Environment**
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project is designed to be executed via a centralized Jupyter Notebook which handles the end-to-end workflow.

1.  **Launch Jupyter Lab**
    ```bash
    jupyter lab
    ```

2.  **Open the Main Pipeline**
    Navigate to `notebooks/All_Tasks_Pipeline.ipynb`. This notebook contains cells to:
    *   Load and ingest the corpus into Qdrant.
    *   Initialize the RAG graph and components.
    *   Run inference on the test dataset.
    *   Export results to `data/`.

### Verifying Components

To test individual modules without running the full pipeline, execute the notebooks located in the `tests/` directory.

*   **Ingestion**: `tests/verify_ingestion.ipynb` ensures that documents are indexed correctly with parent-child metadata.
*   **Retrieval**: `tests/verify_retrieval.ipynb` validates that the retriever returns relevant context for sample queries.
*   **Graph Logic**: `tests/verify_self_crag.ipynb` simulates various scenarios (e.g., hallucination detection) to verify the graph's routing logic.

## Evaluation

The project includes an evaluation script using RAGAS to compute metrics such as Faithfulness, Answer Relevancy, and Context Precision.

```bash
python eval/evaluate.py --submission data/submission_TaskC_Gbgers.jsonl
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.