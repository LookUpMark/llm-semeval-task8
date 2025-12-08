
# MANUALE OPERATIVO INTEGRALE: SEMEVAL 2026 TASK 8 (MTRAGEval) - Open Source Edition

Versione: 2.1 (Llama-3.1 T4 Optimized)

Stato: Esecutivo

Obiettivo: Creare un sistema RAG Multi-Turn "Agentico" (Self-CRAG) che massimizzi Faithfulness e Refusal Accuracy utilizzando esclusivamente LLM Open Source su hardware limitato (2x T4 GPU).

## 1. INTRODUZIONE E TEORIA DEL PROBLEMA

Prima di scrivere una riga di codice, dovete capire esattamente perché i RAG normali falliscono in questo task. Se non capite il problema, l'implementazione sarà inutile.

### 1.1 Il Nemico: Context Decay & Hallucination

Il benchmark mtRAG (Multi-Turn RAG) non è un test di "Ricerca e Risposta". È un test di "Memoria e Onestà".

-   **Context Decay (Decadimento del Contesto):** L'utente chiede: "Chi è il CEO di Apple?". Il sistema risponde: "Tim Cook". L'utente poi chiede: "E quanti anni ha?". Un sistema RAG classico cercherà nel database la stringa "E quanti anni ha?", non troverà nulla e allucinerà. Il sistema deve sapere che "ha" si riferisce a "Tim Cook".
    
-   **Trappola dell'Allucinazione (IDK):** Molte domande nel dataset non hanno risposta nei documenti forniti. I modelli LLM sono addestrati per essere "utili" e quindi tendono a inventare. In questo task, inventare è fatale. Il sistema deve dire "I_DONT_KNOW" se i documenti non supportano la risposta.
    

### 1.2 La Soluzione: Architettura Self-CRAG

Non useremo una "Chain" (catena lineare A -> B -> C), ma un Grafo di Stato (State Graph).

-   **CRAG (Corrective RAG):** Se il retrieval recupera documenti spazzatura, il sistema se ne accorge prima di generare la risposta e prova a cercare meglio (o si arrende).
    
-   **Self-RAG (Self-Reflective RAG):** Dopo aver generato la risposta, il sistema la rilegge. Se contiene fatti non presenti nei documenti, la scarta e ricomincia.
    

----------

## 2. PREPARAZIONE DELL'AMBIENTE (Tutti i Membri)

Poiché stiamo usando modelli locali pesanti, la gestione delle dipendenze è critica per non saturare la VRAM o andare in conflitto con CUDA.

### 2.1 File `requirements.txt`

Spiegazione Concettuale:

Rispetto alla versione standard, qui introduciamo bitsandbytes per la quantizzazione a 4-bit (essenziale per far entrare un modello da 8B parametri nella memoria della GPU T4) e accelerate per distribuire il carico sulle due GPU disponibili.

Dettaglio Tecnico:

Copiate questo contenuto in un file requirements.txt e lanciate pip install -r requirements.txt.

Plaintext

```
# Orchestrazione e Grafo
langchain==0.1.10
langchain-community==0.0.25
langchain-core==0.1.28
langchain-huggingface==0.0.3
langgraph==0.0.26

# Local LLM & Quantization (CRITICO PER KAGGLE T4)
torch
transformers
accelerate
bitsandbytes
scipy
huggingface_hub

# Vector Store e Embeddings
qdrant-client>=1.9.0
langchain-qdrant>=0.1.0
sentence-transformers==2.5.1
rank_bm25==0.2.2
faiss-cpu==1.8.0

# Gestione Dati e Processing
unstructured==0.12.5
tiktoken==0.6.0
pydantic==2.6.3
numpy==1.26.4
pandas==2.2.1

# Valutazione
ragas==0.1.4
datasets==2.18.0

# Utility
python-dotenv==1.0.1
tqdm==4.66.2

```

### 2.2 Repository Structure

Organize the folders exactly like this:

```
/semeval_task8
│
├── /data                    # mtRAG JSON files
├── /dataset                 # Additional dataset files
├── /qdrant_db               # Vector Store persisted data
├── /src                     # Core Python modules
│   ├── __init__.py          # Package exports
│   ├── state.py             # GraphState definition (Member 1)
│   ├── ingestion.py         # Data loading & chunking (Member 2)
│   ├── retrieval.py         # Hybrid retrieval & reranking (Member 3)
│   ├── generation.py        # Llama 3.1 chains & graders (Member 4)
│   └── graph.py             # LangGraph workflow orchestration (Member 1)
├── /eval                    # Evaluation code (Member 5)
│   ├── __init__.py
│   └── evaluate.py          # RAGAS evaluation pipeline
├── /notebooks               # Kaggle test notebooks
│   ├── 01_data_ingestion.ipynb    # Data loading & vector store
│   ├── 02_rag_pipeline.ipynb      # Self-CRAG workflow testing
│   └── 03_evaluation.ipynb        # RAGAS metrics evaluation
├── /docs                    # Documentation
│   └── implementation-guide.md
├── .env.example             # Environment template (copy to .env)
├── .gitignore               # Git ignore patterns
├── main.py                  # CLI entry point
└── requirements.txt         # Python dependencies

```

### 2.3 Kaggle Notebooks

For testing on Kaggle, use the notebooks in `/notebooks`:

| Notebook | Purpose |
|----------|---------|
| `01_data_ingestion.ipynb` | Load mtRAG data, apply Parent-Child chunking, build Qdrant vector store |
| `02_rag_pipeline.ipynb` | Initialize Llama 3.1 4-bit, test Self-CRAG workflow, run multi-turn conversations |
| `03_evaluation.ipynb` | Run RAGAS evaluation with local Llama judge, analyze IDK accuracy |

Each notebook includes:
- `!pip install` cells for dependencies
- GPU verification and VRAM monitoring
- Function skeletons with `raise NotImplementedError`
- Example code (commented) for execution

----------

## 3. IMPLEMENTAZIONE PER RUOLI

### MEMBRO 1: L'ARCHITETTO (Definizione dello Stato)

Spiegazione Concettuale:

Dobbiamo definire un oggetto dati che "viaggia" attraverso il grafo. Non possiamo passare variabili a caso. Definiamo uno schema rigido (TypedDict). L'aspetto più importante è la chiave messages con l'annotazione add_messages: questo dice al grafo che quando un nodo restituisce un messaggio, non deve cancellare la cronologia esistente, ma aggiungersi in coda. Questo crea la "memoria" della conversazione.

**Codice: `/src/state.py`**

Python

```
from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Rappresenta lo stato del nostro grafo RAG.
    Ogni nodo riceve questo stato in input e restituisce un dizionario
    con le chiavi che vuole aggiornare.
    """
    
    # La storia della conversazione (HumanMessages + AIMessages).
    # Annotated[list, add_messages] significa: "Quando un nodo ritorna 'messages',
    # fai un append alla lista esistente invece di sovrascriverla".
    messages: Annotated[List[BaseMessage], add_messages]
    
    # La domanda originale dell'utente nell'ultimo turno.
    question: str
    
    # La domanda "riscritta" dal sistema per essere comprensibile senza contesto.
    # Es: "Quanto costa?" -> "Quanto costa il servizio Cloud IBM?"
    standalone_question: str
    
    # La lista dei documenti recuperati dal Vector Store.
    documents: List[Any]
    
    # La risposta finale generata.
    generation: str
    
    # Flag booleani per la logica decisionale del grafo.
    # 'documents_relevant': Se il grader dice che i documenti trovati sono utili.
    documents_relevant: str  # 'yes' o 'no'
    
    # 'is_hallucination': Se il grader dice che la risposta è inventata.
    is_hallucination: str    # 'yes' o 'no'
    
    # Contatore per evitare loop infiniti di correzione.
    retry_count: int

```

**Dettaglio Tecnico:**

-   Importiamo `TypedDict` per il type hinting statico.
    
-   `add_messages` è una funzione reducer specifica di LangGraph. Senza di essa, ogni passaggio cancellerebbe la memoria precedente.
    
-   Inseriamo `documents` come `List[Any]` perché conterrà oggetti `Document` di LangChain.
    

----------

### MEMBRO 2: DATA ENGINEER (Ingestione Intelligente)

Spiegazione Concettuale:

Il problema principale dei RAG legali/normativi è che la risposta spesso dipende dal contesto "padre" (es. il titolo della legge), ma la ricerca vettoriale funziona meglio su paragrafi piccoli ("figli").

Implementiamo il Parent-Child Chunking:

1.  Spezziamo il testo in blocchi grandi (Parent) per mantenere il contesto.
    
2.  Spezziamo i Parent in blocchi piccoli (Child) per l'indicizzazione.
    
3.  Quando troviamo un Child, al LLM diamo il Parent.
    

**Codice: `/src/ingestion.py`**

Python

```
import json
import uuid
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# CONFIGURAZIONE
# BGE-M3 è SOTA per retrieval denso e funziona bene su CPU/GPU modeste.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def load_and_chunk_data(json_path: str):
    """
    Carica il dataset mtRAG e applica Parent-Child Chunking.
    """
    print(f"--- LOADING DATA FROM {json_path} ---")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    raw_docs = []
    # Adattamento alla struttura JSON di mtRAG
    for item in data:
        doc = Document(
            page_content=item.get('contents', ''),
            metadata={
                "source": item.get('domain', 'general'),
                "doc_id": item.get('id', str(uuid.uuid4()))
            }
        )
        raw_docs.append(doc)

    print(f"Caricati {len(raw_docs)} documenti grezzi.")

    # Configurazione Splitter (Parent e Child)
    # Parent: Chunk grandi (1200 chars) per dare contesto al LLM.
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    
    # Child: Chunk piccoli (400 chars) per la ricerca vettoriale precisa.
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    docs_to_index = []
    
    print("--- STARTING PARENT-CHILD SPLITTING ---")
    for parent_doc in raw_docs:
        parent_chunks = parent_splitter.split_documents([parent_doc])
        
        for p_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            child_chunks = child_splitter.split_documents([p_chunk])
            
            for c_chunk in child_chunks:
                # SALVATAGGIO CONTESTO PADRE
                # Nel metadata del Child, salviamo tutto il testo del Parent.
                c_chunk.metadata["parent_content"] = p_chunk.page_content
                c_chunk.metadata["parent_id"] = parent_id
                c_chunk.metadata["source"] = parent_doc.metadata["source"]
                
                docs_to_index.append(c_chunk)
                
    print(f"Creati {len(docs_to_index)} child chunks pronti per l'indicizzazione.")
    return docs_to_index

def build_vector_store(docs: List[Document], persist_dir: str = "./qdrant_db"):
    """
    Crea e salva il database vettoriale Qdrant.
    """
    print("--- BUILDING VECTOR STORE ---")
    # model_kwargs={'device': 'cuda'} forza l'uso della GPU per creare gli embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'} 
    )
    
    # Create Qdrant client with local persistence
    client = QdrantClient(path=persist_dir)
    
    vectorstore = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embedding_model,
        collection_name="mtrag_collection",
        client=client
    )
    print("--- VECTOR STORE BUILT AND SAVED ---")
    return vectorstore

```

**Dettaglio Tecnico:**

-   `RecursiveCharacterTextSplitter`: Divide il testo cercando di non rompere frasi o paragrafi.
    
-   `c_chunk.metadata["parent_content"]`: Questa è la chiave del successo. Indicizziamo il `c_chunk.page_content` (piccolo), ma nel retrieval estrarremo `parent_content` (grande).
    
-   `HuggingFaceEmbeddings`: Usa `sentence-transformers` in locale. Su Kaggle, `device='cuda'` accelera l'indicizzazione di 10x.
    

----------

### MEMBRO 3: RETRIEVAL SPECIALIST (Ricerca Ibrida & Reranking)

Spiegazione Concettuale:

La ricerca vettoriale ("Cos'è un bond?") a volte fallisce su termini specifici se l'embedding non è perfetto.

Usiamo una strategia a due stadi:

1.  **Retrieval:** Prendiamo 20 documenti che _potrebbero_ essere rilevanti (alta Recall).
    
2.  **Reranking:** Usiamo un modello Cross-Encoder (che legge Query e Documento insieme) per riordinare questi 20 e tenere solo i migliori 5 (alta Precision). Questo è computazionalmente costoso ma necessario per l'accuratezza.
    

**Codice: `/src/retrieval.py`**

Python

```
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Reranker multilingua molto potente ma leggero
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

def get_retriever():
    """
    Restituisce un retriever avanzato: Vector Search (Top 20) -> Rerank (Top 5).
    """
    # 1. Carica il DB esistente
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda'}
    )
    
    client = QdrantClient(path="./qdrant_db")
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="mtrag_collection",
        embedding=embedding_model
    )
    
    # 2. Base Retriever: Recupera molti documenti (Recall)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # 3. Reranker (Cross-Encoder): Filtra per precisione
    # model_kwargs={'device': 'cuda'} è essenziale per velocità su T4
    model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME, 
        model_kwargs={'device': 'cuda'}
    )
    
    # Il compressore filtra i 20 doc e tiene i migliori 5
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    # 4. Pipeline Completa
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever

def format_docs_for_gen(docs):
    """
    Estrae il PARENT CONTENT dai child chunks trovati.
    """
    unique_contents = set()
    final_context = []
    
    for doc in docs:
        # Recupera il contenuto padre salvato nei metadati
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        
        if parent_content not in unique_contents:
            unique_contents.add(parent_content)
            final_context.append(parent_content)
            
    return "\n\n".join(final_context)

```

**Dettaglio Tecnico:**

-   `HuggingFaceCrossEncoder`: A differenza dei modelli vettoriali (Bi-Encoder), il Cross-Encoder prende due input simultaneamente e sputa un punteggio di similarità (0-1). È più lento ma capisce sfumature sintattiche fini.
    
-   `top_n=5`: Riduciamo il rumore per l'LLM, fornendo solo il necessario.
    

----------

### MEMBRO 4: GENAI ENGINEER (Cervello e Controllo - LLAMA 3.1)

Spiegazione Concettuale:

Qui sostituiamo OpenAI con Llama 3.1 8B Instruct.

Poiché abbiamo solo 2 GPU T4 (circa 15GB VRAM ciascuna), non possiamo caricare il modello a piena precisione (FP16 richiederebbe ~16GB + overhead). Dobbiamo usare la Quantizzazione a 4-bit (NF4) tramite bitsandbytes. Questo riduce l'uso di VRAM a circa 6-8GB, permettendo al modello di girare comodamente e lasciando spazio per gli embeddings e il reranker.

Inoltre, Llama 3.1 open-source non ha una funzione with_structured_output affidabile come GPT-4. Useremo il Prompt Engineering per forzare l'output JSON e JsonOutputParser per decodificarlo.

**Codice: `/src/generation.py`**

Python

```
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

# --- CONFIGURAZIONE LLM LOCALE (LLAMA 3.1 8B QUANTIZZATO) ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_llama_pipeline():
    """Configura Llama 3.1 a 4-bit per GPU T4."""
    
    # Configurazione Quantizzazione 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Fix per Llama 3

    # Caricamento Modello
    # device_map="auto" distribuisce automaticamente il modello sulle 2 GPU T4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto", 
        trust_remote_code=True
    )

    # Creazione Pipeline Hugging Face
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512, # Lunghezza risposta
        temperature=0.01,   # Quasi deterministico
        do_sample=True,
        repetition_penalty=1.1 # Evita loop di ripetizione
    )

    return HuggingFacePipeline(pipeline=pipe)

# Inizializziamo l'LLM una volta sola
llm = get_llama_pipeline()

# --- 1. QUERY REWRITER ---
# Llama 3 ha bisogno di prompt chiari sui formati.
rewrite_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sei un assistente AI. Analizza la CRONOLOGIA della chat e l'ULTIMA DOMANDA dell'utente.
Se l'ultima domanda dipende dal contesto (es. "quanto costa?", "e lui?"), riscrivila in modo autonomo.
Se è già chiara, restituiscila invariata.
Restituisci SOLO la domanda riscritta, nient'altro.<|eot_id|>"""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("placeholder", "{messages}"),
    ("human", "{question}"),
])

query_rewriter = rewrite_prompt | llm | StrOutputParser()

# --- 2. GENERATOR (con strict IDK) ---
# Prompt specifico per Llama 3 con delimitatori speciali
gen_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sei un assistente rigoroso per mtRAG.
Usa ESCLUSIVAMENTE il contesto fornito per rispondere.
Se il contesto non contiene le informazioni esatte, RISPONDI SOLO: "I_DONT_KNOW".
Non inventare.

CONTESTO:
{context}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Domanda: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

gen_prompt = ChatPromptTemplate.from_template(gen_system)

generator = gen_prompt | llm | StrOutputParser()

# --- 3. RELEVANCE GRADER (CRAG) ---
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Il documento è rilevante? 'yes' o 'no'")

# Parser manuale per forzare JSON su modelli locali
parser_grade = JsonOutputParser(pydantic_object=GradeDocuments)

grade_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Sei un valutatore. Analizza se il documento è rilevante per la domanda.
Rispondi ESCLUSIVAMENTE con un JSON valido contenente la chiave 'binary_score' con valore 'yes' o 'no'.
Nessun preambolo.

Formato atteso: {{ "binary_score": "yes" }} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Domanda: {question}
Documento: {document}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

grade_prompt = ChatPromptTemplate.from_template(grade_system)

retrieval_grader = grade_prompt | llm | parser_grade

# --- 4. HALLUCINATION GRADER (Self-RAG) ---
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="La risposta è supportata? 'yes' o 'no'")

parser_hallucination = JsonOutputParser(pydantic_object=GradeHallucinations)

hallucination_system = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Confronta la RISPOSTA con i DOCUMENTI.
Se la risposta contiene fatti non presenti nei documenti, rispondi 'no'.
Se è supportata, rispondi 'yes'.
Rispondi ESCLUSIVAMENTE con un JSON: {{ "binary_score": "..." }} <|eot_id|>
<|start_header_id|>user<|end_header_id|>
Documenti: {documents}
Risposta: {generation}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

hallucination_prompt = ChatPromptTemplate.from_template(hallucination_system)

hallucination_grader = hallucination_prompt | llm | parser_hallucination

```

**Dettaglio Tecnico:**

-   `BitsAndBytesConfig`: `load_in_4bit=True` riduce il modello del 75% in memoria con perdita di qualità minima.
    
-   `tokenizer.pad_token = tokenizer.eos_token`: Llama 3 non ha un token di padding di default, bisogna impostarlo per evitare errori.
    
-   `device_map="auto"`: Cruciale su Kaggle. Divide i layer del modello tra le due GPU T4 automaticamente.
    
-   **Prompting Llama 3:** Ho inserito i token speciali `<|begin_of_text|><|start_header_id|>` ecc. nel prompt. Anche se LangChain gestisce parte di questo, esplicitarlo aiuta i modelli quantizzati a seguire meglio le istruzioni e a smettere di generare al momento giusto (`<|eot_id|>`).
    

----------

### MEMBRO 1 (RITORNO): ASSEMBLAGGIO DEL GRAFO

Spiegazione Concettuale:

Qui assembliamo i nodi. La logica è identica, ma dobbiamo gestire le eccezioni. Poiché stiamo usando un parser JSON manuale con un LLM locale, a volte l'LLM potrebbe fallire nel generare JSON valido. Se succede, assumiamo il caso peggiore (documento non rilevante o allucinazione) per sicurezza.

**Codice: `/src/graph.py`**

Python

```
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage, HumanMessage
from src.state import GraphState
from src.generation import query_rewriter, generator, retrieval_grader, hallucination_grader
from src.retrieval import get_retriever, format_docs_for_gen

# --- DEFINIZIONE DEI NODI ---

def rewrite_node(state: GraphState):
    """Nodo 1: Riscrive la query utente."""
    print("--- REWRITE QUERY ---")
    question = state["question"]
    
    # Invocazione del rewriter
    better_question = query_rewriter.invoke({
        "messages": state["messages"],
        "question": question
    })
    print(f"Original: {question} -> Rewritten: {better_question}")
    return {"standalone_question": better_question}

def retrieve_node(state: GraphState):
    """Nodo 2: Cerca nel Vector Store."""
    print("--- RETRIEVE ---")
    question = state["standalone_question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents}

def grade_documents_node(state: GraphState):
    """Nodo 3: Filtra i documenti irrilevanti."""
    print("--- GRADE DOCUMENTS ---")
    question = state["standalone_question"]
    documents = state["documents"]
    
    filtered_docs = []
    relevant_found = False
    
    for d in documents:
        try:
            score = retrieval_grader.invoke({
                "question": question,
                "document": d.page_content
            })
            grade = score.get("binary_score", "no")
        except Exception as e:
            print(f"JSON Parsing Error in Grading: {e}")
            grade = "no"

        if grade == "yes":
            filtered_docs.append(d)
            relevant_found = True
            
    relevance_status = "yes" if relevant_found else "no"
    
    return {
        "documents": filtered_docs, 
        "documents_relevant": relevance_status
    }

def generate_node(state: GraphState):
    """Nodo 4: Genera la risposta."""
    print("--- GENERATE ---")
    question = state["standalone_question"]
    documents = state["documents"]
    
    # Se non ci sono documenti, passiamo contesto vuoto -> IDK
    context = format_docs_for_gen(documents)
    
    generation = generator.invoke({"context": context, "question": question})
    
    return {"generation": generation}

def hallucination_check_node(state: GraphState):
    """Nodo 5: Verifica post-generazione."""
    print("--- CHECK HALLUCINATIONS ---")
    documents = state["documents"]
    generation = state["generation"]
    
    if "I_DONT_KNOW" in generation:
        return {"is_hallucination": "no"}
        
    context = format_docs_for_gen(documents)
    
    try:
        score = hallucination_grader.invoke({
            "documents": context,
            "generation": generation
        })
        grade = score.get("binary_score", "yes") # Default ottimista o pessimista a scelta
    except Exception:
        grade = "no" # In caso di errore, scartiamo per sicurezza
    
    return {"is_hallucination": grade}

def fallback_node(state: GraphState):
    """Nodo Fallback."""
    print("--- FALLBACK ---")
    return {
        "generation": "I_DONT_KNOW",
        "messages": []
    }

# --- ARCHI CONDIZIONALI ---

def decide_to_generate(state: GraphState):
    if state["documents_relevant"] == "yes":
        return "generate"
    else:
        return "fallback"

def decide_to_final(state: GraphState):
    if state["is_hallucination"] == "yes":
        # Se è supportato dai doc (yes) -> ok
        print("DECISION: GENERATION ACCEPTED")
        return "end"
    else:
        # Se non è supportato (no) -> fallback
        print("DECISION: HALLUCINATION -> FALLBACK")
        return "fallback"

# --- COSTRUZIONE DEL GRAFO ---
workflow = StateGraph(GraphState)

workflow.add_node("rewrite", rewrite_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_check_node)
workflow.add_node("fallback", fallback_node)

workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "grade_docs")

workflow.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)

workflow.add_edge("generate", "hallucination_check")

workflow.add_conditional_edges(
    "hallucination_check",
    decide_to_final,
    {
        "end": END,
        "fallback": "fallback"
    }
)

workflow.add_edge("fallback", END)

app = workflow.compile()

```

**Dettaglio Tecnico:**

-   Blocchi `try-except`: Fondamentali con LLM locali e output JSON. Se Llama sbaglia una virgola nel JSON, il parser esplode. Noi catturiamo l'errore e forziamo un valore di default sicuro ("no"), preferendo scartare un buon documento piuttosto che includerne uno cattivo (o crashare).
    

----------

### MEMBRO 5: EVALUATION ENGINEER (Il giudice)

Spiegazione Concettuale:

Per valutare senza OpenAI, dobbiamo dire a Ragas di usare il nostro LLM Llama locale e i nostri embeddings HuggingFace. Di default Ragas cerca OpenAI e fallisce.

**Codice: `/eval/evaluate.py`**

Python

```
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_core.messages import HumanMessage
from src.graph import app
from src.generation import llm # Importiamo il nostro LLM
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configurazione Embeddings per Ragas
ragas_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def run_evaluation(test_dataset):
    """
    Esegue la valutazione usando Llama 3.1 come giudice (Judge Model).
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

if __name__ == "__main__":
    test_data = [
        {"question": "Chi è il CEO di Apple?", "ground_truth": "Tim Cook"}
    ]
    run_evaluation(test_data)

```

**Dettaglio Tecnico:**

-   `llm=llm`: Passiamo l'oggetto `HuggingFacePipeline` creato in `generation.py`.
    
-   `embeddings=ragas_embeddings`: Assicura che Ragas non cerchi API key di OpenAI per calcolare le similarità vettoriali.
    

----------

## 4. GUIDA ALL'USO (MAIN ENTRY POINT)

Spiegazione Concettuale:

Un semplice loop CLI per interagire con il sistema. Poiché il caricamento del modello Llama 3.1 su T4 richiede tempo (1-2 minuti), il main deve inizializzare tutto prima di chiedere l'input.

**Codice: `main.py`**

Python

```
import os
from dotenv import load_dotenv
from src.graph import app
from langchain_core.messages import HumanMessage, AIMessage

# Carica token HuggingFace per scaricare i modelli la prima volta
load_dotenv()

def main():
    print("=== SEMEVAL 2026 TASK 8: MTRAGEval SYSTEM (LLAMA 3.1 LOCAL) ===")
    print("Attendi il caricamento del modello e dei pesi su GPU...\n")
    
    # Simuliamo chat history locale
    chat_history = []
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                break
                
            chat_history.append(HumanMessage(content=user_input))
            
            inputs = {
                "question": user_input,
                "messages": chat_history
            }
            
            final_output = None
            # Stream dell'esecuzione del grafo
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"  [Node: {key}] processing...")
                    final_output = value
            
            if final_output and "generation" in final_output:
                ai_response = final_output["generation"]
            else:
                ai_response = "Error: No generation produced."
                
            print(f"AI: {ai_response}\n")
            
            chat_history.append(AIMessage(content=ai_response))
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Errore critico: {e}")

if __name__ == "__main__":
    main()

```

**Dettaglio Tecnico:**

-   Il caricamento delle librerie in `src.graph` e `src.generation` avverrà alla prima esecuzione di `import`, allocando la VRAM.
    
-   Il loop mantiene la `chat_history` in RAM python.
    

----------

## 5. CHECKLIST FINALE E STRATEGIA DI GARA (Versione Kaggle)

1.  **Monitoraggio VRAM:** Usate `nvidia-smi` nel notebook Kaggle. Se saturate i 15GB di una T4, riducete `max_new_tokens` in `generation.py` o diminuite il `batch_size` negli embeddings.
    
2.  **Quantizzazione:** Assicuratevi che `bitsandbytes` sia installato correttamente. Se dà errori di CUDA, provate a reinstallarlo o verificare la versione di `torch`.
    
3.  **Prompt Sensitivity:** Llama 3.1 è molto sensibile ai prompt. Se vedete che il nodo "Rewriter" chiacchiera troppo invece di dare solo la domanda, modificate il `rewrite_system` aggiungendo esempi (Few-Shot Prompting).
    
4.  **Parsing JSON:** I parser JSON sui modelli piccoli sono il punto debole. Se falliscono spesso, considerate di usare un parser regex (Regular Expression) come fallback invece di scartare tutto.
