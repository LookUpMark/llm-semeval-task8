# Guida Operativa Dettagliata per il Membro 4 (GenAI Engineer)
## Quantized LLM Inference, Prompt Engineering & Agentic Chains

Questa guida costituisce il riferimento definitivo per il tuo lavoro. A differenza di una check-list veloce, questo documento esplode ogni singolo passaggio per fornirti non solo il "cosa" e il "come", ma soprattutto il **"perché"** di ogni riga di codice.

Il tuo compito è fondamentale: sei il "cervello" del sistema. Devi rendere un modello LLM da 8 Miliardi di parametri (Llama 3.1) eseguibile su GPU consumer (T4) tramite quantizzazione, e orchestrarlo per funzionare come un Agente affidabile che non inventa mai.

### Deliverables del Membro 4

1.  **Modulo Generation** (`src/generation.py`)
2.  **Aggiornamento Dipendenze** (`requirements.txt`)
3.  **Notebook di Verifica** (Snippet di test per l'integrazione)

---

## 1. Aggiornamento delle Dipendenze

Prima di scrivere codice per l'LLM, dobbiamo preparare l'ambiente per supportare la quantizzazione a 4-bit e l'integrazione con LangChain.

**File da modificare:** `requirements.txt` (nella root del progetto).

### Spiegazione delle Librerie

Non stiamo aggiungendo librerie a caso. Ecco la ragione tecnica:

1.  **`bitsandbytes`**: La chiave di volta. Permette di caricare i pesi del modello in formato NF4 (Normal Float 4-bit). Senza di questo, Llama 3.1 richiederebbe 16GB di VRAM sol per i pesi, saturando istantaneamente una T4.
2.  **`accelerate`**: Gestisce il caricamento efficiente dei pesi su GPU ("Layer shuffling") e supporta `device_map="auto"` per distribuire il modello se avessi più GPU.
3.  **`langchain-huggingface`**: Il nuovo pacchetto standard (post-deprecazione di `LangChain` core) per integrare pipeline HF.
4.  **`langchain-core`**: Fornisce le primitive base (`ChatPromptTemplate`, `OutputParser`).

### Codice da Aggiungere

Assicurati che `requirements.txt` contenga queste righe specifiche:

```text
# Local LLM & Quantization (CRITICO PER GPU T4)
torch
transformers>=4.38.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
scipy
huggingface_hub

# Framework Agentico
langchain>=0.1.10
langchain-community>=0.0.25
langchain-core>=0.1.28
langchain-huggingface>=0.0.3
```

---

## 2. Architettura Tecnica: Llama 3.1 su Hardware Limitato

### 2.1 Il Problema della Memoria (VRAM)

Una GPU T4 ha **15.8 GB** di VRAM.
Llama 3.1 8B in precisione FP16 (Half Precision) richiede:
- $8 \times 10^9 \text{ params} \times 2 \text{ bytes (FP16)} \approx 16 \text{ GB}$

Siamo esattamente al limite. Aggiungendo il contesto (KV Cache) e gli embedding, andremmo in **OOM (Out Of Memory)** immediatamente.

### 2.2 La Soluzione: Quantizzazione NF4

Usando `bitsandbytes`, riduciamo ogni peso a **4 bit** (0.5 byte).
- $8 \times 10^9 \times 0.5 \text{ bytes} \approx 4 \text{ GB}$

Questo ci lascia ~10-11 GB liberi per:
1.  Contesto della chat (fino a 8k token)
2.  Embedding Model (caricato dal Membro 3)
3.  KV Cache
4.  Overhead di PyTorch

---

## 3. Implementazione `src/generation.py`

In questa sezione costruiremo il file passo dopo passo per capire ogni scelta.

### 3.1 Setup e Configurazione Quantizzazione

Iniziamo con gli import e la configurazione cruciale di `BitsAndBytesConfig`.

```python
import torch
from typing import Any, List, Tuple
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

# Identificativo esatto del modello su Hugging Face Hub
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_llama_pipeline(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct") -> HuggingFacePipeline:
    """
    Configura Llama 3.1 a 4-bit per GPU T4.
    Questa è la funzione più critica per la stabilità del sistema.
    """
    print("🚀 Loading Llama 3.1 8B Instruct (4-bit quantized)...")
    
    # 1. Quantization Configuration: NF4 (Normal Float 4)
    # NF4 is mathematically optimized for neural network weight distribution
    # compared to standard FP4.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Quantize quantization constants too (extra savings)
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16  # Calculation happens in FP16 for stability
    )

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    # CRITICAL FIX: Llama 3 does not have a default pad_token.
    # Without this, batch generation or chain usage might crash.
    tokenizer.pad_token = tokenizer.eos_token 
    
    # 3. Model Loading
    # device_map="auto" is essential: it handles GPU/CPU allocation automatically.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True
    )

    # 4. Generation Pipeline
    # Parameters optimized to reduce hallucinations and repetitions
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,      # Balance between response length and VRAM usage
        do_sample=True,
        temperature=0.01,          # Low temperature for deterministic outputs
        repetition_penalty=1.1,   # Slight penalty to avoid loops "The dog the dog the dog..."
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False    # Critical: Ensures we only get the generated answer, not the prompt + answer
    )

    # LangChain Wrapper
    return HuggingFacePipeline(pipeline=pipe)
```

> [!IMPORTANT]
> `temperature=0.01` è preferibile a `0.0` con `do_sample=True` per evitare edge-case in cui il modello entra in loop deterministici su token rari.

### 3.2 Modelli Pydantic per Output Strutturato

Poiché usiamo `JsonOutputParser`, dobbiamo definire lo schema atteso. Questo serve al parser per validare l'output dell'LLM (che a volte potrebbe essere sporco).

```python
# --- PYDANTIC MODELS ---

class GradeDocuments(BaseModel):
    """Schema for evaluating document relevance (CRAG)."""
    binary_score: str = Field(
        description="Is the document relevant to the question? Answer 'yes' or 'no'"
    )

class GradeHallucinations(BaseModel):
    """Schema for detecting hallucinations (Self-RAG)."""
    binary_score: str = Field(
        description="Is the answer supported by the documents? Answer 'yes' or 'no'"
    )
```

### 3.3 Prompt Engineering: Chains & Istruzioni

Llama 3 Instruct risponde molto meglio se usiamo i suoi token speciali: `<|begin_of_text|>`, `<|start_header_id|>`, etc. Li includeremo esplicitamente.

> [!WARNING]
> È fondamentale usare `PromptTemplate` (e non `ChatPromptTemplate`) quando si includono manualmente i token speciali Llama 3 (`<|start_header_id|>`). Altrimenti, LangChain potrebbe aggiungere prefissi non voluti come `System:` o `Human:`, confondendo il modello.

#### Helper per la Storia della Chat
Definiamo una funzione helper per formattare la storia della chat in una stringa leggibile:

```python
def _format_chat_history(messages: List[Tuple[str, str]]) -> str:
    """Helper to format chat history from list of tuples to string."""
    formatted = []
    for role, content in messages:
        role_name = "User" if role.lower() in ["user", "human"] else "Assistant"
        formatted.append(f"{role_name}: {content}")
    return "\n".join(formatted)
```

#### A. Query Rewriter Chain
Risolve le anafore ("E quanti anni ha?" → "Quanti anni ha Obama?"). Usa un `PromptTemplate` raw.

```python
def _create_query_rewriter(llm) -> Any:
    """Rewrites context-dependent questions into standalone questions."""
    
    rewrite_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in rewriting. 
Analyze the CHAT HISTORY and the LAST QUESTION.
Your task is to rewrite the last question so that it is understandable WITHOUT the history.

Examples:
Chat: "Who is Steve Jobs?" -> "Apple Founder."
User: "When was he born?"
Rewrite: "When was Steve Jobs born?"

If the question is already clear, return it unchanged.
Return ONLY the rewritten question, no preamble.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Chat History:
{messages}

Last Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    rewrite_prompt = PromptTemplate(
        template=rewrite_system,
        input_variables=["messages", "question"]
    )
    
    # Chain with formatting lambda
    chain = (
        {
            "messages": lambda x: _format_chat_history(x["messages"]),
            "question": lambda x: x["question"]
        }
        | rewrite_prompt
        | llm 
        | StrOutputParser()
    )
    
    return chain
```

#### B. Generator Chain (con Strict Refusal)
La parte più delicata: deve dire `I_DONT_KNOW` se non sa.

```python
def _create_generator(llm) -> Any:
    """Generates the final response with strict fallback."""
    
    gen_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict assistant for the mtRAG task.
Answer the user's question using EXCLUSIVELY the context provided below.

GOLDEN RULE:
If the context does not contain the necessary information to answer, YOU MUST answer with: "I_DONT_KNOW".
Do not try to guess. Do not use your internal knowledge.

CONTEXT:
{context}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    
    gen_prompt = PromptTemplate(
        template=gen_system,
        input_variables=["context", "question"]
    )
    
    return gen_prompt | llm | StrOutputParser()
```

#### C. Graders (CRAG & Self-RAG)
Usano `JsonOutputParser` per produrre output strutturati.

```python
def _create_retrieval_grader(llm) -> Any:
    """Evaluates if retrieved documents are relevant."""
    parser = JsonOutputParser(pydantic_object=GradeDocuments)
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

### EXAMPLES
Question: What is the capital of France?
Document: Paris is the capital city of France.
Output: {{"binary_score": "yes"}}

Question: How do I make pasta?
Document: The history of pasta starts in Italy.
Output: {{"binary_score": "yes"}}

Question: Who won the 2018 World Cup?
Document: The 2018 World Cup was hosted in Russia.
Output: {{"binary_score": "no"}}

Question: What is the weather in London?
Document: Apple Inc. released a new iPhone.
Output: {{"binary_score": "no"}}
### END EXAMPLES

Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Respond EXCLUSIVELY with a JSON object with key "binary_score".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
Document: {document}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"]
    )
    return prompt | llm | parser

def _create_hallucination_grader(llm) -> Any:
    """Verifies if the response is a hallucination."""
    parser = JsonOutputParser(pydantic_object=GradeHallucinations)
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is fully supported by the set of facts.
'no' means that the answer contains information that is not found in the documents (hallucination).

### EXAMPLES
Documents: Apple was founded in 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne.
Answer: Apple was founded in 1976.
Output: {{"binary_score": "yes"}}

Documents: Apple was founded in 1976.
Answer: Apple was founded in 1976 by Tim Cook.
Output: {{"binary_score": "no"}}

Documents: The sky is blue due to Rayleigh scattering.
Answer: The sky is blue.
Output: {{"binary_score": "yes"}}
### END EXAMPLES

Respond EXCLUSIVELY with a JSON object with key "binary_score".<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Documents: {documents}
Answer: {generation}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["documents", "generation"]
    )
    return prompt | llm | parser
```

---

## 4. Pattern di Configurazione con Dataclass

Invece di usare variabili globali (anti-pattern che rende difficile il testing e la manutenzione), utilizziamo un approccio più pulito con una **dataclass** come contenitore dei componenti e una **funzione factory** per l'inizializzazione.

```python
from dataclasses import dataclass

@dataclass
class GenerationComponents:
    """
    Container for all generation components.
    Encapsulates LLM and chains, avoiding global state.
    """
    llm: Any
    query_rewriter: Any
    generator: Any
    retrieval_grader: Any
    hallucination_grader: Any

def create_generation_components() -> GenerationComponents:
    """
    Factory function to create all generation components.
    Call this once at application startup and pass the result 
    to the components that need it.
    """
    print("Creating Generation Components...")
    llm = get_llama_pipeline()
    
    components = GenerationComponents(
        llm=llm,
        query_rewriter=_create_query_rewriter(llm),
        generator=_create_generator(llm),
        retrieval_grader=_create_retrieval_grader(llm),
        hallucination_grader=_create_hallucination_grader(llm)
    )
    
    print("Generation Components Ready.")
    return components
```

> [!TIP]
> Questo pattern permette di iniettare facilmente mock nei test, passando un `GenerationComponents` mock invece di quello reale.

---

## 5. Codice Completo (`src/generation.py`)

Copia questo codice nel file sorgente. Include tutte le sezioni discusse sopra.

```python
"""
Generation Module for MTRAGEval.

This module provides Llama 3.1 8B Instruct based generation and grading:
- Query Rewriter: Rewrites context-dependent questions to standalone form
- Generator: Produces answers with strict I_DONT_KNOW fallback
- Relevance Grader (CRAG): Evaluates document relevance
- Hallucination Grader (Self-RAG): Checks if generation is supported by documents

Uses 4-bit quantization (NF4) via bitsandbytes for T4 GPU compatibility.
"""

import torch
from typing import Any, List, Tuple
from dataclasses import dataclass
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline


# --- PYDANTIC MODELS FOR JSON PARSING ---

class GradeDocuments(BaseModel):
    """Schema for document relevance grading output."""
    binary_score: str = Field(
        description="Is the document relevant? 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Schema for hallucination detection output."""
    binary_score: str = Field(
        description="Is the answer supported? 'yes' or 'no'"
    )


# --- GENERATION COMPONENTS CONTAINER ---

@dataclass
class GenerationComponents:
    """
    Container for all generation components.
    
    Encapsulates the LLM pipeline and all chain instances,
    avoiding global state and improving testability.
    
    Attributes:
        llm: The HuggingFacePipeline instance (Llama 3.1 or Qwen 2.5).
        query_rewriter: Chain for rewriting context-dependent queries.
        generator: Chain for RAG generation with I_DONT_KNOW fallback.
        retrieval_grader: Chain for CRAG document relevance grading.
        hallucination_grader: Chain for Self-RAG hallucination detection.
    """
    llm: Any
    query_rewriter: Any
    generator: Any
    retrieval_grader: Any
    hallucination_grader: Any


def get_llama_pipeline(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct") -> HuggingFacePipeline:
    """
    Configure Llama 3.1 at 4-bit for T4 GPU.
    
    Sets up:
    - BitsAndBytesConfig for NF4 quantization
    - AutoTokenizer with padding fix
    - AutoModelForCausalLM with device_map="auto" for multi-GPU
    - HuggingFace text-generation pipeline
    
    Returns:
        HuggingFacePipeline wrapper for LangChain integration.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # return_full_text=False ensures we only get the generated answer,
    # not the prompt + answer.
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.01, # Low temperature for more deterministic outputs
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False 
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )

def _format_chat_history(messages: List[Tuple[str, str]]) -> str:
    """Helper to format chat history from list of tuples to string."""
    formatted = []
    for role, content in messages:
        role_name = "User" if role.lower() in ["user", "human"] else "Assistant"
        formatted.append(f"{role_name}: {content}")
    return "\n".join(formatted)

def _create_query_rewriter(llm) -> Any:
    """
    Creates the query rewriter chain.
    
    Analyzes chat history and rewrites context-dependent questions
    to standalone form.
    
    Returns:
        LangChain chain for query rewriting.
    """
    rewrite_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in rewriting. 
Analyze the CHAT HISTORY and the LAST QUESTION.
Your task is to rewrite the last question so that it is understandable WITHOUT the history.

Examples:
Chat: "Who is Steve Jobs?" -> "Apple Founder."
User: "When was he born?"
Rewrite: "When was Steve Jobs born?"

If the question is already clear, return it unchanged.
Return ONLY the rewritten question, no preamble.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Chat History:
{messages}

Last Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    rewrite_prompt = PromptTemplate(
        template=rewrite_system,
        input_variables=["messages", "question"]
    )

    # Chain that formats messages before passing to prompt
    chain = (
        {
            "messages": lambda x: _format_chat_history(x["messages"]),
            "question": lambda x: x["question"]
        }
        | rewrite_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

def _create_generator(llm) -> Any:
    """
    Creates the RAG generator chain.
    
    Generates answers using provided context with strict I_DONT_KNOW fallback
    when information is not available.
    
    Returns:
        LangChain chain for generation.
    """
    gen_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a strict assistant for the mtRAG task.
Answer the user's question using EXCLUSIVELY the context provided below.

GOLDEN RULE:
If the context does not contain the necessary information to answer, YOU MUST answer with: "I_DONT_KNOW".
Do not try to guess. Do not use your internal knowledge.

CONTEXT:
{context}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    gen_prompt = PromptTemplate(
        template=gen_system,
        input_variables=["context", "question"]
    )

    return gen_prompt | llm | StrOutputParser()

def _create_retrieval_grader(llm) -> Any:
    """
    Creates the document relevance grader (CRAG component).
    
    Evaluates if a document is relevant to the question.
    Returns JSON with binary_score: 'yes' or 'no'.
    
    Returns:
        LangChain chain for document grading.
    """
    parser = JsonOutputParser(
        pydantic_object=GradeDocuments
    )

    grade_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
Respond EXCLUSIVELY with a JSON object with key "binary_score". For example: {"binary_score": "yes"} or {"binary_score": "no"}.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
Document: {document}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    prompt = PromptTemplate(
        template=grade_system,
        input_variables=["question", "document"]
    )

    return prompt | llm | parser

def _create_hallucination_grader(llm) -> Any:
    """
    Creates the hallucination grader (Self-RAG component).
    
    Compares generation against source documents to detect hallucinations.
    Returns JSON with binary_score: 'yes' (supported) or 'no' (hallucinated).
    
    Returns:
        LangChain chain for hallucination detection.
    """
    parser = JsonOutputParser(
        pydantic_object=GradeHallucinations
    )

    hallucination_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'yes' means that the answer is fully supported by the set of facts.
'no' means that the answer contains information that is not found in the documents (hallucination).
Respond EXCLUSIVELY with a JSON object with key "binary_score". For example: {"binary_score": "yes"} or {"binary_score": "no"}.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Documents: {documents}
Answer: {generation}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    prompt = PromptTemplate(
        template=hallucination_system,
        input_variables=["documents", "generation"]
    )

    return prompt | llm | parser


def create_generation_components(model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> GenerationComponents:
    """
    Factory function to create and configure all generation components.
    
    Creates the quantized LLM pipeline and initializes all chains
    for query rewriting, generation, and grading.

    Args:
        model_id: Hugging Face model ID to load. Defaults to Llama 3.1 8B Instruct.
    
    Returns:
        GenerationComponents: Dataclass containing initialized LLM and chains.
    
    Example:
        >>> components = create_generation_components()
        >>> result = components.generator.invoke({"context": "...", "question": "..."})
    """
    print(f"Creating Generation Components with model: {model_id}...")
    llm = get_llama_pipeline(model_id=model_id)
    
    components = GenerationComponents(
        llm=llm,
        query_rewriter=_create_query_rewriter(llm),
        generator=_create_generator(llm),
        retrieval_grader=_create_retrieval_grader(llm),
        hallucination_grader=_create_hallucination_grader(llm)
    )
    
    print("Generation Components Ready.")
    return components
```

---

## 6. Verifica dell'Implementazione

Poiché non possiamo eseguire un training, la verifica si basa sul caricamento corretto dei pesi e sull'inferenza base.

Creare un file `tests/verify_generation.py` o eseguire questo snippet in un notebook:

```python
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation import create_generation_components

# CONFIGURATION
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Standard
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" # Light for local testing

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def print_header(title):
    tqdm.write(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
    tqdm.write(f"{Colors.HEADER}{Colors.BOLD} {title} {Colors.RESET}")
    tqdm.write(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}\n")

def print_section(title):
    tqdm.write(f"\n{Colors.CYAN}{Colors.BOLD}>>> {title} {Colors.RESET}")

def print_input(label, content):
    tqdm.write(f"{Colors.YELLOW}{Colors.BOLD}{label}:{Colors.RESET} {content}")

def print_output(output, expected):
    tqdm.write(f"{Colors.GREEN}{Colors.BOLD}Output:{Colors.RESET} {output}")
    tqdm.write(f"{Colors.BLUE}{Colors.BOLD}Expected:{Colors.RESET} {expected}")

if __name__ == "__main__":
    progress_bar = tqdm(
        total=100,
        desc="Running Tests",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        leave=True
    )

    print_header("GENERATION TESTS STARTED")
    
    # Initialization
    print_section("Initializing Components")
    tqdm.write(f"Model ID: {Colors.BOLD}{MODEL_ID}{Colors.RESET}")
    components = create_generation_components(model_id=MODEL_ID)
    progress_bar.update(10)

    # Test Generator (Positive)
    print_section("TEST: Generator (Positive)")
    context = "Tim Cook is the CEO of Apple."
    question = "Who is the CEO of Apple?"
    
    print_input("Context", context)
    print_input("Question", question)
    
    res = components.generator.invoke(
        {
            "context": context,
            "question": question
        }
    )
    print_output(res.strip(), "Something like 'Tim Cook'")
    progress_bar.update(50)

    # Test Generator (Negative)
    print_section("TEST: Generator (Negative)")
    context = "The sky is blue."
    question = "Who is the CEO of Apple?"
    
    print_input("Context", context)
    print_input("Question", question)
    
    res = components.generator.invoke(
        {
            "context": context,
            "question": question
        }
    )
    print_output(res.strip(), "'I_DONT_KNOW'")
    progress_bar.update(30)

    # Test Rewriter
    print_section("TEST: Query Rewriter")
    history = [
        ("human", "Who is the creator of Python?"),
        ('assistant', 'Guido van Rossum')
    ]
    question = "When was he born?"
    
    print_input("History", str(history))
    print_input("Question", question)
    
    res = components.query_rewriter.invoke(
        {
            "messages": history,
            "question": question
        }
    )
    print_output(res.strip(), "'When was Guido van Rossum born?'")
    progress_bar.update(10)
    
    progress_bar.close()

    print_header("GENERATION TESTS COMPLETED")
```

---

## 7. Troubleshooting

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `CUDA out of memory` | VRAM esaurita (T4 ha 16GB) | 1. Controlla che `load_in_4bit=True`. 2. Riduci `max_new_tokens`. 3. Assicurati che non ci siano altri processi GPU attivi. |
| `KeyError: 'pad_token'` | Tokenizer mal configurato | Assicurati di aver impostato `tokenizer.pad_token = tokenizer.eos_token`. |
| `ImportError: cannot import name 'BitsAndBytesConfig'` | `transformers` vecchio | Aggiorna con `pip install -U transformers`. |
| `ValueError: You need to pass a valid token` | Accesso Llama 3 negato | Esegui `huggingface-cli login` o setta `HUGGINGFACEHUB_API_TOKEN`. |
| `OutputParserException` | LLM genera JSON sporco | È normale con modelli piccoli/quantizzati. Il codice di orchestrazione (`graph.py`) deve gestire l'eccezione (fallback a 'no'). |

> [!TIP]
> **Monitoraggio GPU**: Tieni aperto un terminale con `watch -n 1 nvidia-smi` mentre carichi il modello. Dovresti vedere l'uso della memoria salire a circa 5-6 GB per il modello base.

---

## 8. Parte 2: Implementazione Qwen 2.5 14B (Alternative SOTA)

Se il progetto richiede capacità di reasoning superiori (specialmente in matematica o coding) o se si vuole testare un'alternativa SOTA a Llama 3, **Qwen 2.5 14B** è la scelta migliore. 
Nonostante sia più grande (14B vs 8B), grazie alla quantizzazione 4-bit, occupa circa **8-9 GB di VRAM**, rientrando ancora perfettamente nei limiti della T4 (15.8 GB).

### 8.1 Perché Qwen 2.5?
- **Finestra di contesto**: 128k nativa (contro 8k di Llama 3).
- **Reasoning**: Superiore in benchmark logici.
- **Supporto multilingua**: Eccellente supporto per l'italiano.

### 8.2 Configurazione `get_qwen_pipeline`

Aggiungi questa funzione alternativa in `src/generation.py`. 
Nota le differenze nel prompt template (ChatML).

```python
QWEN_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

def get_qwen_pipeline() -> HuggingFacePipeline:
    print(f"🚀 Loading {QWEN_MODEL_ID} (4-bit)...")
    
    # 1. Quantization Configuration (Identical)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # 2. Tokenizer Qwen
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    # Qwen handles padding differently, but eos_token is okay as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Model Loading
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 4. Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # Qwen supports longer outputs
        temperature=0.01,
        do_sample=True,
        repetition_penalty=1.05 # Qwen is less prone to repetitions
    )
    
    return HuggingFacePipeline(pipeline=pipe)
```

### 8.3 Prompting Qwen (ChatML)

Qwen usa il formato ChatML (`<|im_start|>system...`). LangChain di solito gestisce questo se si usano i `ChatPromptTemplate` corretti, ma per sicurezza ecco come adattare i prompt manuali.

Sostituisci i template in `src/generation.py` se usi Qwen:

**Query Rewriter (Qwen Style)**
```python
def _create_qwen_rewriter(llm) -> Any:
    # Qwen does not use <|begin_of_text|>, but standard ChatML
    system = """You are an assistant specialized in rewriting.
Analyze the history and the last question.
Rewrite the last question to make it independent of the context.
If it is already clear, return it unchanged.
Return ONLY the rewritten question."""
    
    # LangChain with HuggingFacePipeline and Qwen will automatically apply 
    # the correct chat template if we use messages.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("placeholder", "{messages}"),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()
```

### 8.4 Switch nel Codice
    
Per passare da Llama a Qwen, modifica la funzione `create_generation_components()`:
    
```python
def create_generation_components() -> GenerationComponents:
    print("Creating Generation Components (QWEN MODE)...")
    
    # --- MODEL SWITCH ---
    # Use Qwen instead of Llama
    llm = get_qwen_pipeline()
    # --------------------

    # Optional: Use Qwen-specific rewriter if defined
    # query_rewriter = _create_qwen_rewriter(llm)
    query_rewriter = _create_query_rewriter(llm)
    
    components = GenerationComponents(
        llm=llm,
        query_rewriter=query_rewriter,
        generator=_create_generator(llm),
        retrieval_grader=_create_retrieval_grader(llm),
        hallucination_grader=_create_hallucination_grader(llm)
    )
    
    print("Generation Components Ready (Qwen Mode).")
    return components
```
