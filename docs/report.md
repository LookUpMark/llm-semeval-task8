# Rapporto Tecnico Strategico: Ottimizzazione Architetturale e Roadmap Operativa per SemEval 2026 Task 8 (Open Source Edition)

**Stato:** Esecutivo (Llama-3.1 T4 Optimized)

## 1. Introduzione e Analisi del Contesto Strategico

### 1.1 Definizione dell'Obiettivo e Ambito del Task
Il presente rapporto tecnico definisce il piano di esecuzione per il SemEval 2026 Task 8: MTRAGEval. A differenza delle iterazioni precedenti, questo task richiede la gestione di un dialogo multi-turno su hardware limitato (2x T4 GPU). L'obiettivo primario è massimizzare la *Faithfulness* e la *Refusal Accuracy*, evitando allucinazioni tramite un'architettura che non dipende da API proprietarie (OpenAI) ma sfrutta esclusivamente modelli Open Source locali quantizzati.

L'analisi del dataset mtRAG evidenzia che i sistemi tradizionali falliscono nel *Context Decay* (perdita di riferimenti anaforici come "e quanti anni ha?") e nell'inventare risposte quando i documenti non contengono le informazioni. La strategia tecnica adotta quindi un'architettura **Self-CRAG (Corrective RAG + Self-RAG)** implementata tramite LangGraph, progettata per operare entro i 15GB di VRAM disponibili.

### 1.2 Risoluzione delle Incongruenze
Rispetto ai draft iniziali, il progetto si allinea ai vincoli operativi reali:
* **Hardware (Il Vincolo T4):** L'architettura deve girare su GPU T4 (architettura Turing, 16GB VRAM teorici, ~15GB utilizzabili). Questo impone un vincolo ferreo. Un modello Llama 3 8B in FP16 occupa circa 16GB solo di pesi, saturando la memoria senza lasciare spazio per il contesto (KV Cache) o gli attivazioni. L'unica via è `bitsandbytes` NF4 (Normal Float 4), che riduce il modello a ~5-6GB, lasciando ~10GB per il contesto lungo del RAG.
* **Dipendenze (L'approccio Sovereign):** Viene eliminata la dipendenza da API esterne. Questo non è solo per risparmiare costi, ma per garantire la *riproducibilità* scientifica e la privacy dei dati. Un giudice basato su GPT-4 cambia nel tempo (OpenAI aggiorna i modelli); un giudice Llama 3 locale è congelato e garantisce che i benchmark di oggi siano validi anche tra sei mesi. Tutto lo stack è "air-gapped" friendly.

### 1.3 Visione Architetturale di Alto Livello
La soluzione proposta è un **Grafo di Stato (State Graph)** gestito da LangGraph. Il flusso non è lineare ma ciclico:
1.  **Rewrite:** Riscrive la query utente per renderla indipendente dal contesto.
2.  **Retrieve & Rerank:** Recupera documenti da Qdrant e li filtra con un Cross-Encoder.
3.  **Grade:** Un nodo valutatore decide se i documenti sono rilevanti.
4.  **Generate:** Llama 3.1 genera la risposta.
5.  **Reflect:** Il sistema verifica post-generazione se ha allucinato. In caso positivo, scarta e riprova o va in fallback.

---

## 2. Analisi del Benchmark mtRAG e Strategia dei Dati

### 2.1 Strategia di Ingestione
Il dataset mtRAG richiede una gestione intelligente del contesto. L'ingestione utilizzerà `RecursiveCharacterTextSplitter` per gestire la struttura dei documenti JSON forniti.

### 2.2 Strategie Avanzate di Chunking: Parent-Child
Per risolvere il problema della perdita di contesto nei chunk piccoli, adotteremo il **Parent-Child Chunking** implementato in `src/ingestion.py`:
1.  **Parent Chunk:** Blocchi grandi (1200 caratteri) che mantengono il contesto completo.
2.  **Child Chunk:** Blocchi piccoli (400 caratteri) ottimizzati per la ricerca vettoriale.
3.  **Indicizzazione:** Si indicizza il *Child*, ma si salva il contenuto del *Parent* nei metadati (`c_chunk.metadata["parent_content"]`). Al momento del retrieval, l'LLM riceve il contenuto padre.

**Rationale del Parent-Child:**
Il problema fondamentale del RAG su documenti complessi è il "Context Fragmentation". Se l'utente chiede "Quali sono le eccezioni all'articolo 5?", un chunk da 500 caratteri potrebbe contenere "L'eccezione è se il soggetto è minorenne", ma *senza* contenere il testo "Articolo 5" che era nel paragrafo precedente. Il sistema vettoriale non troverebbe mai quel chunk perché manca la keyword "Articolo 5".
Con il Parent-Child, il chunk piccolo "puzza" semanticamente del contesto genitore grazie all'overlap e alla vicinanza, permettendo il recupero, ma l'LLM vede il blocco grande che contiene *sia* "Articolo 5" *sia* l'eccezione, permettendogli di collegare i punti.

### 2.3 Gestione delle Domande "Non Rispondibili" (IDK)
Il sistema è istruito per essere rigoroso. Il prompt del generatore include delimitatori speciali e l'istruzione: *Se il contesto non contiene le informazioni esatte, RISPONDI SOLO: "I_DONT_KNOW"*. Non sono previste soglie statistiche complesse, ma un controllo semantico diretto guidato dal prompt engineering su Llama 3.1.

---

## 3. Architettura Tecnica: Implementazione LangGraph

### 3.1 Definizione dello Stato del Grafo (GraphState)
Lo stato, definito in `src/state.py`, utilizza `TypedDict` e include:
* `messages`: Lista con annotazione `add_messages` per mantenere la memoria della conversazione.
* `standalone_question`: La query riscritta.
* `documents`: Lista di oggetti Document recuperati.
* `generation`: La risposta generata.
* `documents_relevant` & `is_hallucination`: Flag decisionali ('yes'/'no') prodotti dai grader.
* `retry_count`: Per evitare loop infiniti.

### 3.2 Descrizione dei Nodi Funzionali
* **Nodo 1: Query Rewriter.** Utilizza Llama 3.1 per analizzare la `chat_history` e riscrivere l'ultima domanda se dipende dal contesto (es. "Quanto costa?" -> "Quanto costa il servizio Cloud IBM?").
* **Nodo 2: Retrieval.** Interroga **Qdrant** (`mtrag_collection`) usando embeddings BGE-M3 su GPU.
* **Nodo 3: Grade Documents.** Un validatore basato su LLM analizza ogni documento recuperato. Utilizza un parser JSON manuale (`JsonOutputParser`) con gestione degli errori `try-except` per robustezza contro i fallimenti di formato dei modelli locali.
* **Nodo 4: Generate.** Utilizza **Llama 3.1 8B Instruct (4-bit)**. Il prompt forza l'uso esclusivo del contesto recuperato.
* **Nodo 5: Hallucination Check.** Confronta la generazione con i documenti. Se rileva fatti non supportati, imposta `is_hallucination: "yes"`. 
* **Nodo Fallback.** Se i documenti non sono rilevanti o l'allucinazione persiste, il nodo restituisce staticamente "I_DONT_KNOW". 

**Perché un nodo Fallback esplicito?** 
Molti sistemi RAG lasciano che sia l'LLM a decidere di dire "non lo so". Ma gli LLM Instruction-Tuned hanno un bias verso l'essere *utili*, quindi spesso provano a indovinare. Forzando un nodo strutturale nel grafo che sovrascrive l'output con una stringa hardcoded ("I_DONT_KNOW"), eliminiamo l'ambiguità. Se il sistema decide di non rispondere, la risposta è garantita essere pulita.

### 3.3 Flusso del Grafo
Il grafo implementato in `src/graph.py` prevede:
1.  Archi condizionali post-grading: Se `documents_relevant == "no"`, si va direttamente al **Fallback** (nessuna Web Search prevista).
2.  Archi condizionali post-generazione: Se `is_hallucination == "yes"`, il sistema può tentare di rigenerare o andare in fallback.

---

## 4. Specifiche Tecniche dei Componenti di Retrieval

### 4.1 Embedding Model: BGE-M3
Si utilizza `BAAI/bge-m3` tramite `HuggingFaceEmbeddings`. L'esecuzione è forzata su device CUDA (`model_kwargs={'device': 'cuda'}`) per garantire tempi di indicizzazione accettabili su T4.

### 4.2 Reranker: Cross-Encoder
Per raffinare il retrieval, si implementa una strategia a due stadi in `src/retrieval.py`:
1.  **Base Retriever:** Recupera i top-20 documenti da Qdrant.
2.  **Compressor:** Usa `BAAI/bge-reranker-v2-m3` (Cross-Encoder) per riordinare e selezionare solo i top-5 documenti più rilevanti (`top_n=5`).

**Cross-Encoder vs Bi-Encoder:** la differenza chiave è l'Attenzione.
- Un Bi-Encoder (Retrieval veloce) calcola il vettore della domanda e del documento separatamente. Non "vedono" l'uno l'altro fino al calcolo del coseno. Perde sfumature.
- Un Cross-Encoder (Reranker) prende in input `[CLS] Domanda [SEP] Documento`. L'attenzione del Transformer può confrontare ogni parola della domanda con ogni parola del documento. È 100 volte più lento, ma capisce se "banca" si riferisce a un fiume o a un edificio in base al contesto. Usarlo solo su 20 documenti è il trucco per avere questa precisione senza pagare il costo computazionale su tutto il DB.

### 4.3 Database Vettoriale
La scelta definitiva è **Qdrant**. Viene utilizzato `QdrantVectorStore` con persistenza locale su file (`path="./qdrant_db"`), eliminando la necessità di server containerizzati complessi per la fase di sviluppo base.

---

## 5. Strategie di Generazione e Mitigazione

### 5.1 Configurazione LLM (Llama 3.1 4-bit)
Per mitigare i limiti di memoria, il "GenAI Engineer" implementerà in `src/generation.py`:
* **Quantizzazione:** Utilizzo di `BitsAndBytesConfig` con `load_in_4bit=True` e tipo `nf4`.
* **Tokenizzazione:** Fix esplicito per il `pad_token` di Llama 3 (`tokenizer.pad_token = tokenizer.eos_token`).
* **Pipeline:** Configurazione con `repetition_penalty=1.1` e `temperature=0.01` per output quasi deterministici.

### 5.2 Parsing e Formattazione
Data la difficoltà dei modelli quantizzati nel produrre JSON perfetti, si utilizzeranno parser Pydantic (`JsonOutputParser`) all'interno di blocchi `try-except`. In caso di errore di parsing nel grading, il sistema assumerà un atteggiamento conservativo (es. scartare il documento).

---

## 6. Piano Operativo e Roadmap

La roadmap è strutturata sulla creazione dei moduli Python in `src/`.

### Settimana 1: Setup e Ingestione
* **Attività:** Setup di `requirements.txt` (inclusi `accelerate`, `bitsandbytes`).
* **Output:** Script `src/ingestion.py` funzionante che produce la cartella `./qdrant_db` popolata con chunks Parent-Child.

### Settimana 2: Pipeline RAG Base
* **Attività:** Implementazione di `src/retrieval.py` con Cross-Encoder e `src/generation.py` con caricamento Llama 4-bit.
* **Output:** Notebook `02_rag_pipeline.ipynb` capace di eseguire una chat single-turn.

### Settimana 3: Orchestrazione Graph
* **Attività:** Sviluppo di `src/graph.py` e `src/state.py`. Collegamento dei nodi Rewrite, Grade e Hallucination Check.
* **Output:** CLI `main.py` funzionante con gestione della memoria chat.

### Settimana 4: Valutazione e Tuning
* **Attività:** Esecuzione di `src/eval/evaluate.py`. Monitoraggio VRAM con `nvidia-smi`.
* **Output:** Metriche Ragas calcolate localmente.

---

## 7. Valutazione e Quality Assurance (QA)

### 7.1 Framework di Valutazione: Ragas Locale
Non potendo usare GPT-4 come giudice, la valutazione in `eval/evaluate.py` utilizzerà lo stesso **Llama 3.1 locale** come giudice.
* Il codice passa esplicitamente `llm=llm` e `embeddings=ragas_embeddings` alla funzione `evaluate()` di Ragas per evitare chiamate OpenAI.
* Metriche monitorate: *Faithfulness*, *Answer Relevancy*, *Context Precision*.

### 7.2 Monitoraggio Risorse
Una parte critica del QA sarà il monitoraggio dell'occupazione VRAM. Se si saturano i 15GB della T4, il piano prevede di ridurre il `max_new_tokens` (attualmente 512) o il batch size degli embeddings.

---

## 8. Conclusioni
Questa architettura rappresenta l'adattamento necessario per competere nel Task 8 con risorse limitate. L'uso di **Llama 3.1 a 4-bit**, combinato con la precisione del **Parent-Child Chunking** su Qdrant e la logica di controllo **Self-CRAG** in LangGraph, permette di soddisfare i requisiti di multi-turn e robustezza alle allucinazioni senza dipendenze cloud costose.