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
    


### 1.3 Perché questa architettura specifica? (Rationale)

Potreste chiedervi: *"Perché non usare un semplice script Python con un `if`?"* o *"Perché LangGraph e non LangChain standard?"*.

1.  **Ciclicità vs Linearità:** Una Chain standard (es. `RetrievalQA`) è una linea retta: Input -> Retrieve -> Gen -> Output. Se il retrieval sbaglia, l'output è sbagliato. Punto. In un **Grafo**, possiamo creare dei *loop*: se la risposta non piace al Judge, torniamo indietro e cerchiamo ancora. Questo è l'unico modo per aumentare l'accuratezza senza cambiare modello.
2.  **State Management:** In un dialogo multi-turn, dobbiamo portarci dietro la cronologia. LangGraph gestisce lo *Stato* in modo nativo, permettendo a ogni nodo di leggere e scrivere nella memoria condivisa senza passare argomenti giganti ovunque.
3.  **Debuggabilità:** Ogni "Nodo" è una funzione isolata. Se il Retrieval si rompe, fissate solo `retrieve_node`. In una catena monolitica, il debug è un incubo.

----------

## 2. PREPARAZIONE DELL'AMBIENTE (Tutti i Membri)

Poiché stiamo usando modelli locali pesanti, la gestione delle dipendenze è critica per non saturare la VRAM o andare in conflitto con CUDA.

### 2.0 SETUP DATASET (CRITICO - PRIMO PASSO)

Prima di installare qualsiasi cosa, **dovete procurarvi i dati**. Senza dati, Member 2 e Member 5 sono bloccati.

**Fonte Ufficiale:** [IBM mt-rag-benchmark GitHub](https://github.com/IBM/mt-rag-benchmark/)

**Procedura di Setup Manuale:**
1.  **Download:** Scaricate il repository come ZIP o clonate `git clone https://github.com/IBM/mt-rag-benchmark.git`.
2.  **Individuazione File (Corpora - Task A):** Nel repo scaricato, andate in `corpora/passage_level/`. Troverete 4 file zip: `clapnq.jsonl.zip`, `cloud.json.zip`, `fiqa.jsonl.zip`, `govt.jsonl.zip`.
3.  **Individuazione File (Evaluation - Task C):** Andate in `human/generation_tasks/`. Troverete `reference+RAG.jsonl` (Target per Task C).
4.  **Posizionamento & Unzip:**
    *   **Per l'Ingestione (Membro 2):** I file `.jsonl` estratti devono trovarsi in `/dataset/corpora/` (folder `passage_level`).
    *   **Per la Valutazione (Membro 5):** Copiate `human/generation_tasks/reference+RAG.jsonl` in `/dataset/human/generation_tasks/`.
5.  **Verifica:** Assicuratevi che `ls dataset/corpora/passage_level/` mostri i file JSONL.

> **NOTA IMPORTANTE:** Per iniziare, consigliamo di usare solo **uno** dei domini (es. `govt.jsonl`) per evitare tempi di indicizzazione biblici sulla T4.

> **NOTA:** L'assignment menziona "Task C: Generation with Retrieved Passages (RAG)" come target principale (`reference+RAG.jsonl` o simile). Assicuratevi di avere questo file per la valutazione.

### 2.1 File `requirements.txt`

Spiegazione Concettuale:

Rispetto alla versione standard, qui introduciamo bitsandbytes per la quantizzazione a 4-bit (essenziale per far entrare un modello da 8B parametri nella memoria della GPU T4) e accelerate per distribuire il carico sulle due GPU disponibili.

Dettaglio Tecnico:

Copiate questo contenuto in un file requirements.txt e lanciate pip install -r requirements.txt.

```
# Orchestration & Grafo
langchain
langchain-community
langchain-core
langchain-huggingface
langgraph

# Local LLM & Quantization (CRITICO PER KAGGLE T4)
torch
transformers
accelerate
bitsandbytes
scipy
huggingface_hub

# Vector Store e Embeddings
qdrant-client
langchain-qdrant
sentence-transformers
rank_bm25
faiss-cpu

# Gestione Dati e Processing
unstructured
tiktoken
pydantic
numpy
pandas

# Valutazione
ragas
datasets

# Utility
python-dotenv
tqdm
```

**Perché queste librerie specifiche?**
-   **`bitsandbytes`**: Non è una semplice libreria di compressione. Implementa il data type **NF4 (Normal Float 4)**. A differenza della quantizzazione intera (INT4), NF4 è ottimizzato per la distribuzione dei pesi delle reti neurali (che è normalmente distribuita), preservando molta più "intelligenza" del modello a parità di bit.
-   **`accelerate`**: Gestisce il caricamento del modello su più device. Con `device_map="auto"`, *accelerate* calcola automaticamente quali layer del modello mettere sulla GPU 0 e quali sulla GPU 1 per bilanciare la VRAM. Fondamentale per il setup 2x T4.
-   **`scipy`**: Spesso dipendenza nascosta per algoritmi di distanza avanzati.

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
├── /notebooks               # Kaggle test notebooks
│   ├── TaskA_Retrieval.ipynb      # Retrieval testing (Task A)
│   ├── TaskB_Generation.ipynb     # Generation testing (Task B)
│   ├── TaskC_RAG.ipynb            # Full RAG Pipeline (Task C)
│   └── All_Tasks_Pipeline.ipynb   # Combined Workflow
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
| Notebook | Purpose |
|----------|---------|
| `TaskA_Retrieval.ipynb` | Load mtRAG data, apply Parent-Child chunking, build Qdrant vector store & test Retrieval |
| `TaskB_Generation.ipynb` | Test Generation Logic (LLM, Rewriter, Graders) |
| `TaskC_RAG.ipynb` | Full Self-CRAG workflow testing (Task C) |

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

```python
from typing import TypedDict, List, Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Rappresenta lo stato del nostro grafo RAG.
    """
    # La storia della conversazione. add_messages evita di sovrascrivere.
    messages: Annotated[List[BaseMessage], add_messages]
    
    # La domanda originale dell'utente.
    question: str
```

**Nota:** `TypedDict` garantisce che sappiamo sempre quali chiavi esistono. `add_messages` è il riduttore magico di LangGraph.

```python
    # La domanda "riscritta" per essere autonoma (es. "Quanto costa?" -> "Quanto costa X?")
    standalone_question: str
    
    # La lista dei documenti recuperati (oggetti Document)
    documents: List[Any]
    
    # La risposta generata dall'LLM
    generation: str
```

**Nota:** `standalone_question` è fondamentale per il retrieval. Se cerchiamo solo "Quanto costa?", non troveremo nulla.

```python
    # Flag decisionali per il grafo
    documents_relevant: str  # 'yes' o 'no'
    is_hallucination: str    # 'yes' o 'no'
    
    # Contatore per evitare loop infiniti (es. max 3 tentativi)
    retry_count: int
```

**Dettaglio Tecnico:**

-   Importiamo `TypedDict` per il type hinting statico.
    
-   `add_messages` è una funzione reducer specifica di LangGraph. Senza di essa, ogni passaggio cancellerebbe la memoria precedente.
    
-   Inseriamo `documents` come `List[Any]` perché conterrà oggetti `Document` di LangChain.
    

----------

### APPROFONDIMENTO: Perché `add_messages`?
In Python, se avete un dizionario `state = {"messages": [A, B]}` e una funzione ritorna `{"messages": [C]}`, un update normale sovrascriverebbe tutto, risultando in `state = {"messages": [C]}`. Avreste perso la memoria.
L'annotazione `Annotated[List, add_messages]` istruisce LangGraph a usare una logica di **append**: prende la lista vecchia `[A, B]` e ci attacca quella nuova `[C]`, risultando in `[A, B, C]`. Questo dettaglio implementativo è il cuore del supporto Multi-Turn.

### MEMBRO 2: DATA ENGINEER (Ingestione Intelligente)

Spiegazione Concettuale:

Il problema principale dei RAG legali/normativi è che la risposta spesso dipende dal contesto "padre" (es. il titolo della legge), ma la ricerca vettoriale funziona meglio su paragrafi piccoli ("figli").

Implementiamo il Parent-Child Chunking:

1.  Spezziamo il testo in blocchi grandi (Parent) per mantenere il contesto.
    
2.  Spezziamo i Parent in blocchi piccoli (Child) per l'indicizzazione.
    
3.  Quando troviamo un Child, al LLM diamo il Parent.
    

**Codice: `/src/ingestion.py`**

```python
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
    RICHIEDE: Il file corpus JSON presente in /data.
    """
    print(f"--- LOADING DATA FROM {json_path} ---") 
    # ESEMPIO: json_path potrebbe essere "./data/corpus.json" o simile.
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ERRORE CRITICO: Il file {json_path} non esiste. Hai seguito il passo 2.0 SETUP DATASET?")
    
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

    # 1. Parent Splitter: Chunk grandi (1200 chars) per il contesto
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    
    # 2. Child Splitter: Chunk piccoli (400 chars) per la ricerca
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    docs_to_index = []
```

**Nota:** La discrepanza tra 1200 (lettura) e 400 (ricerca) è voluta. 400 caratteri sono un vettore denso e preciso. 1200 caratteri sono un paragrafo leggibile.

```python
    print("--- STARTING PARENT-CHILD SPLITTING ---")
    for parent_doc in raw_docs:
        # Creiamo prima i blocchi grandi
        parent_chunks = parent_splitter.split_documents([parent_doc])
        
        for p_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())
            
            # Spezziamo il blocco grande in pezzi piccoli
            child_chunks = child_splitter.split_documents([p_chunk])
            
            for c_chunk in child_chunks:
                # TRUCCO: Salviamo il testo GRANDE nei metadati del piccolo
                c_chunk.metadata["parent_content"] = p_chunk.page_content
                c_chunk.metadata["parent_id"] = parent_id
                c_chunk.metadata["source"] = parent_doc.metadata["source"]
                
                docs_to_index.append(c_chunk)
                
    return docs_to_index
```

**Nota:** La riga chiave è `c_chunk.metadata["parent_content"] = p_chunk.page_content`. Stiamo "nascondendo" la risposta completa dentro il frammento di ricerca.

```python
def build_vector_store(docs: List[Document], persist_dir: str = "./qdrant_db"):
    """
    Crea e salva il database vettoriale Qdrant.
    """
    # ... (codice Qdrant standard)
    
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
    

**Perché Parent-Child Chunking? La metafora dell'Avvocato.**
Immaginate un avvocato che cerca una legge.
-   Se indicizziamo l'intero codice civile (Parent), la ricerca vettoriale si diluisce e non trova nulla.
-   Se indicizziamo solo le singole frasi (Child), troviamo la frase "È vietato fumare", ma perdiamo il contesto ("...nelle raffinerie di petrolio").
Il Parent-Child Chunking ci dà il meglio dei due mondi: "Troviamo la frase piccola (Child), ma all'LLM diamo la pagina intera (Parent)". Per mtRAG, dove le risposte dipendono da dettagli sottili ma richiedono contesto ampio, questa è l'unica strategia vincente.

### MEMBRO 3: RETRIEVAL SPECIALIST (Ricerca Ibrida & Reranking)

Spiegazione Concettuale:

La ricerca vettoriale ("Cos'è un bond?") a volte fallisce su termini specifici se l'embedding non è perfetto.

Usiamo una strategia a due stadi:

1.  **Retrieval:** Prendiamo 20 documenti che _potrebbero_ essere rilevanti (alta Recall).
    
2.  **Reranking:** Usiamo un modello Cross-Encoder (che legge Query e Documento insieme) per riordinare questi 20 e tenere solo i migliori 5 (alta Precision). Questo è computazionalmente costoso ma necessario per l'accuratezza.
    

**Codice: `/src/retrieval.py`**

```python
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
    # Qui chiediamo 20 documenti. È una rete a strascico.
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    
    # 3. Reranker (Cross-Encoder): Filtra per precisione
    # Questo modello è lento ma intelligente. Legge la coppia (query, doc).
    model = HuggingFaceCrossEncoder(
        model_name=RERANKER_MODEL_NAME, 
        model_kwargs={'device': 'cuda'}
    )
```

**Nota:** Usiamo `top_n=5` nel prossimo step. Da 20 candidati mediocri, estraiamo 5 gemme.

```python
    # Il compressore filtra i 20 doc e tiene i migliori 5
    compressor = CrossEncoderReranker(model=model, top_n=5)
    
    # 4. Pipeline Completa: Retriever + Reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever
```

**Metodo helper per estrarre il Parent Content:**

```python
def format_docs_for_gen(docs):
    """
    Estrae il PARENT CONTENT dai child chunks trovati.
    """
    unique_contents = set()
    final_context = []
    
    for doc in docs:
        # MAGIA: Non usiamo doc.page_content (piccolo), ma il parent (grande)
        parent_content = doc.metadata.get("parent_content", doc.page_content)
        
        # Deduplicazione (se due child puntano allo stesso parent)
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

```python
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

    # Nota su Temperature = 0.01:
    # Non usiamo 0.0 assoluto perché alcuni backend di campionamento (sampling) potrebbero comportarsi in modo anomalo
    # o degenerare in greedy search pura che a volte si "blocca" su token ripetitivi. 
    # 0.01 è un compromesso sicuro per avere quasi-determinismo con stabilità.
    
    # Nota su Repetition Penalty = 1.1:
    # I modelli Llama (specialmente quantizzati) tendono a entrare in loop ("Il costo è 10. Il costo è 10. Il costo è 10.").
    # Una penalità leggera (1.1 o 1.15) penalizza i token appena usati, forzando il modello a variare e proseguire il discorso.

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

---

### CONFRONTO LLM: Llama 3.1 8B vs Qwen 2.5 14B (Ablation Study)

Per validare scientificamente la scelta del modello LLM, è necessario eseguire un **confronto sistematico** tra almeno due modelli. Questo confronto sarà parte della fase di valutazione finale e permetterà di giustificare la scelta del modello nella submission.

**Obiettivo del confronto:**
Misurare l'impatto della scelta del modello LLM sulle metriche chiave del task (Faithfulness, Answer Relevancy, IDK Accuracy) mantenendo costanti tutti gli altri componenti (retriever, reranker, chunking strategy).

**Modelli da confrontare:**

| Aspetto | Llama 3.1 8B | Qwen 2.5 14B |
|---------|--------------|--------------|
| **Parametri** | 8B | 14B |
| **VRAM (4-bit)** | ~6GB | ~9-10GB |
| **Capacità di ragionamento** | Buona | Eccellente (migliore su task complessi) |
| **Velocità di inferenza** | Più veloce | Più lento (~30-40% in più) |
| **Supporto multilingua** | Ottimo | Eccellente (particolarmente forte su cinese) |
| **JSON output** | Richiede prompt espliciti | Migliore aderenza al formato richiesto |

**Ipotesi da testare:**
1. Qwen 2.5 14B dovrebbe ottenere punteggi più alti su *Faithfulness* grazie alla maggiore capacità di ragionamento.
2. Qwen 2.5 14B dovrebbe produrre meno errori di parsing JSON nei grader.
3. Llama 3.1 8B dovrebbe essere significativamente più veloce (metrica: tokens/secondo).

**Implementazione per il confronto: `/src/generation_qwen.py`**

Creare un file separato con la configurazione Qwen per poter switchare facilmente tra i due modelli.

```python
import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

# --- CONFIGURAZIONE LLM LOCALE (QWEN 2.5 14B QUANTIZZATO) ---
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

def get_qwen_pipeline():
    """Configura Qwen 2.5 14B a 4-bit per GPU T4."""
    
    # Configurazione Quantizzazione 4-bit (NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # Qwen performa meglio con bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.01,
        do_sample=True,
        repetition_penalty=1.05  # Qwen è più stabile, serve meno penalità
    )

    return HuggingFacePipeline(pipeline=pipe)

llm = get_qwen_pipeline()
```

**Nota su `bfloat16`:** Qwen 2.5 è stato addestrato con bfloat16, quindi usare questo dtype per il compute garantisce migliore qualità rispetto a float16. Le T4 supportano bfloat16 via software emulation (leggermente più lento ma funzionale).

**Prompt per Qwen 2.5 (ChatML format):**

Qwen 2.5 usa il formato ChatML, diverso da Llama 3. Ecco i prompt adattati:

```python
# --- QUERY REWRITER (Qwen) ---
rewrite_system_qwen = """<|im_start|>system
Sei un assistente AI. Analizza la CRONOLOGIA della chat e l'ULTIMA DOMANDA dell'utente.
Se l'ultima domanda dipende dal contesto (es. "quanto costa?", "e lui?"), riscrivila in modo autonomo.
Se è già chiara, restituiscila invariata.
Restituisci SOLO la domanda riscritta, nient'altro.<|im_end|>"""

rewrite_prompt_qwen = ChatPromptTemplate.from_messages([
    ("system", rewrite_system_qwen),
    ("placeholder", "{messages}"),
    ("human", "{question}"),
])

query_rewriter = rewrite_prompt_qwen | llm | StrOutputParser()

# --- GENERATOR (Qwen) ---
gen_system_qwen = """<|im_start|>system
Sei un assistente rigoroso per mtRAG.
Usa ESCLUSIVAMENTE il contesto fornito per rispondere.
Se il contesto non contiene le informazioni esatte, RISPONDI SOLO: "I_DONT_KNOW".
Non inventare.

CONTESTO:
{context}<|im_end|>
<|im_start|>user
Domanda: {question}<|im_end|>
<|im_start|>assistant"""

gen_prompt_qwen = ChatPromptTemplate.from_template(gen_system_qwen)
generator = gen_prompt_qwen | llm | StrOutputParser()

# --- RELEVANCE GRADER (Qwen) ---
grade_system_qwen = """<|im_start|>system
Sei un valutatore. Analizza se il documento è rilevante per la domanda.
Rispondi ESCLUSIVAMENTE con un JSON valido contenente la chiave 'binary_score' con valore 'yes' o 'no'.
Nessun preambolo.

Formato atteso: {{ "binary_score": "yes" }}<|im_end|>
<|im_start|>user
Domanda: {question}
Documento: {document}<|im_end|>
<|im_start|>assistant"""

grade_prompt_qwen = ChatPromptTemplate.from_template(grade_system_qwen)
retrieval_grader = grade_prompt_qwen | llm | parser_grade

# --- HALLUCINATION GRADER (Qwen) ---
hallucination_system_qwen = """<|im_start|>system
Confronta la RISPOSTA con i DOCUMENTI.
Se la risposta contiene fatti non presenti nei documenti, rispondi 'no'.
Se è supportata, rispondi 'yes'.
Rispondi ESCLUSIVAMENTE con un JSON: {{ "binary_score": "..." }}<|im_end|>
<|im_start|>user
Documenti: {documents}
Risposta: {generation}<|im_end|>
<|im_start|>assistant"""

hallucination_prompt_qwen = ChatPromptTemplate.from_template(hallucination_system_qwen)
hallucination_grader = hallucination_prompt_qwen | llm | parser_hallucination
```

**Protocollo di confronto:**

1. **Setup:** Preparare due notebook identici, uno con `from src.generation import llm` e uno con `from src.generation_qwen import llm`.
2. **Dataset:** Usare lo stesso subset di test (es. 100 domande multi-turn) per entrambi i modelli.
3. **Metriche da raccogliere:**
   - Faithfulness (RAGAS)
   - Answer Relevancy (RAGAS)
   - Context Precision (RAGAS)
   - IDK Accuracy (% di risposte "I_DONT_KNOW" quando la risposta corretta è IDK)
   - JSON Parsing Error Rate (% di fallimenti nei grader)
   - Latenza media per query (secondi)
   - Picco VRAM (GB)
4. **Output:** Tabella comparativa da includere nel paper/report finale.

**Risultato atteso:**
Il modello con il miglior trade-off tra accuratezza e risorse sarà selezionato per la submission finale. Se Qwen 2.5 14B supera Llama 3.1 8B di almeno 5 punti percentuali su Faithfulness senza saturare la VRAM, sarà la scelta raccomandata.

----------

### MEMBRO 1 (RITORNO): ASSEMBLAGGIO DEL GRAFO

Spiegazione Concettuale:

Qui assembliamo i nodi. La logica è identica, ma dobbiamo gestire le eccezioni. Poiché stiamo usando un parser JSON manuale con un LLM locale, a volte l'LLM potrebbe fallire nel generare JSON valido. Se succede, assumiamo il caso peggiore (documento non rilevante o allucinazione) per sicurezza.

**Codice: `/src/graph.py`**

```python
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

def increment_retry_node(state: GraphState):
    """Incrementa retry_count prima di ritentare."""
    print("--- INCREMENT RETRY ---")
    return {"retry_count": state.get("retry_count", 0) + 1}

def decide_to_final(state: GraphState):
    """
    Post-hallucination: decide accettare, ritentare o fallback.
    - Se grounded (yes) -> end
    - Se allucinazione + retry disponibili -> increment_retry
    - Se allucinazione + max retry -> fallback
    """
    is_grounded = state["is_hallucination"] == "yes"
    retry_count = state.get("retry_count", 0)
    
    if is_grounded:
        print("DECISION: ACCEPTED")
        return "end"
    elif retry_count < MAX_RETRIES:
        print(f"DECISION: RETRY ({retry_count + 1}/{MAX_RETRIES})")
        return "increment_retry"
    else:
        print("DECISION: MAX RETRIES -> FALLBACK")
        return "fallback"


# --- COSTRUZIONE DEL GRAFO ---
workflow = StateGraph(GraphState)

# Aggiunta Nodi
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_docs", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("hallucination_check", hallucination_check_node)
workflow.add_node("increment_retry", increment_retry_node)  # NUOVO: Retry Loop
workflow.add_node("fallback", fallback_node)

```

**Flusso Lineare Iniziale:**

```python
workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "grade_docs")
```

**Bivio Decisionale 1 (Post-Grading):**

```python
# Se i documenti sono buoni -> genera. Se no -> fallback.
workflow.add_conditional_edges(
    "grade_docs",
    decide_to_generate,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)

workflow.add_edge("generate", "hallucination_check")
```

**Bivio Decisionale 2 (Post-Generation):**

```python
# Se la risposta è fedele -> END. Se allucinazione + retry left -> increment_retry. Se max retry -> fallback.
workflow.add_conditional_edges(
    "hallucination_check",
    decide_to_final,
    {
        "end": END,
        "increment_retry": "increment_retry",  # NUOVO: loop per ritentare
        "fallback": "fallback"
    }
)

# NUOVO: Dopo increment_retry, torniamo a generate (crea il loop)
workflow.add_edge("increment_retry", "generate")

workflow.add_edge("fallback", END)

app = workflow.compile()

```

**Dettaglio Tecnico:**

-   Blocchi `try-except`: Fondamentali con LLM locali e output JSON. Se Llama sbaglia una virgola nel JSON, il parser esplode. Noi catturiamo l'errore e forziamo un valore di default sicuro ("no"), preferendo scartare un buon documento piuttosto che includerne uno cattivo (o crashare).
    

----------

### VISIONE D'INSIEME DEL GRAFO: Perché questa topologia?
Avete notato che il grafo non è un cerchio perfetto, ma ha dei rami morti (Ending).
1.  **Early Stopping (Relevance Check):** Se il Retrieval non trova nulla di buono (`grade_docs` -> `no`), non sprechiamo tempo a generare. Andiamo subito al Fallback. Questo risparmia secondi preziosi di GPU e riduce le allucinazioni (perché forzare l'LLM a rispondere su spazzatura genera spazzatura).
2.  **Safety Check (Hallucination):** Anche se i documenti erano buoni, l'LLM potrebbe aver capito male. Il nodo finale controlla *la risposta*. È un "doppio controllo" costoso ma necessario per la metrica di *Faithfulness*.

### MEMBRO 5: EVALUATION ENGINEER (Il giudice)

Spiegazione Concettuale:

Per valutare senza OpenAI, dobbiamo dire a Ragas di usare il nostro LLM Llama locale e i nostri embeddings HuggingFace. Di default Ragas cerca OpenAI e fallisce.

**Codice: `/eval/evaluate.py`**

```python
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

if __name__ == "__main__":
    # PUNTO DI ATTENZIONE:
    # Per il test rapido usate la lista dummy.
    # Per la valutazione reale, scommentate load_test_data e puntate al file scaricato al passo 2.0.
    
    # test_data = load_test_data("./dataset/reference+RAG.jsonl") 
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

```python
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
