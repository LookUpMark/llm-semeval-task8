# System Description Paper Guideline — SemEval 2026 Task 8

Considerando che si tratta di un **System Description Paper** per una challenge specifica (**SemEval**) e basandosi sulla natura fortemente **ingegneristica** e **resource-constrained** del progetto (2× NVIDIA T4 GPU), la struttura del paper deve mettere in evidenza non solo **cosa è stato fatto**, ma soprattutto **perché è stato fatto in quel modo**, evidenziando il continuo compromesso tra **efficienza ed efficacia**.

Di seguito viene proposta la **struttura ottimale del paper**, suddivisa per sezioni, con indicazione esplicita del contributo di ciascun membro del team e del collegamento tra i vari moduli.

---

## Titolo suggerito (bozza)

**TeamName at SemEval 2026 Task 8:  
A Resource-Constrained Self-CRAG Architecture for Multi-Turn RAG using Open Source LLMs**

---

## 1. Introduction

In questa sezione va definito il problema e la filosofia generale della soluzione proposta.

### The Task
- Breve descrizione del **SemEval 2026 Task 8**.
- Focus sulla gestione del **dialogo multi-turn**.
- Rischio di **hallucinations** nei sistemi RAG conversazionali.

### Challenges
- Problema del **Context Decay**, ovvero il deterioramento del contesto informativo lungo il dialogo.
- Necessità di alta accuratezza nei casi di rifiuto, con risposte corrette di tipo **"I_DONT_KNOW"**.

### Proposed Approach — *The “Sovereign” Strategy*
- La soluzione è progettata per operare:
  - interamente **offline**,
  - esclusivamente con **modelli open-source**,
  - su **hardware consumer (2× T4 GPU)**.
- Nessun utilizzo di API proprietarie (es. GPT-4).
- Questo vincolo giustifica tutte le scelte tecniche successive:
  - quantizzazione,
  - modelli di dimensioni contenute ma accuratamente selezionati,
  - architettura modulare e ciclica.

---

## 2. System Architecture

Questa sezione fornisce una panoramica ad alto livello del sistema prima di entrare nei dettagli implementativi.

### Workflow Overview
- Adozione di un **grafo ciclico agentico (LangGraph)** invece di una pipeline lineare.
- Descrizione dei nodi principali:
  - **Rewrite**
  - **Retrieve**
  - **Grade**
  - **Generate**
  - **Reflect**

L’approccio ciclico consente:
- controllo esplicito del flusso,
- auto-correzione,
- riduzione delle allucinazioni.

---

## 3. Methodology

La sezione centrale del paper, in cui ogni membro descrive il proprio modulo.

---

### 3.1 Data Ingestion and Indexing Strategy  
**(Membro 2)**

Questa sezione descrive la strategia di preparazione e indicizzazione del corpus documentale.

#### Corpus Processing
- Pulizia dei documenti.
- Normalizzazione dei testi.
- Gestione e preservazione dei **metadati**.

#### Parent–Child Chunking
- Descrizione del problema della **Context Fragmentation** causato dal chunking standard.
- Introduzione della strategia a doppio livello:
  - **Parent chunks**: ~1200 caratteri (contesto semantico globale).
  - **Child chunks**: ~400 caratteri (granularità per retrieval).
- Meccanismo di recupero dei child con ricostruzione del parent durante la generazione.

#### Vector Storage
- Scelta di **Qdrant** come vector database.
- Utilizzo della **Cosine Distance**.
- Gestione dei payload per collegare chunk figli e padri.

---

### 3.2 Hybrid Retrieval Module  
**(Membro 3)**

#### Embedding Model
- Utilizzo di **BGE-M3** per:
  - supporto multi-lingua,
  - multi-granularità semantica,
  - compatibilità con retrieval ibrido.

#### Two-Stage Retrieval
1. **Broad Retrieval**
   - Recupero iniziale Top-20 documenti via embedding.
2. **Reranking**
   - Cross-Encoder per il riordinamento semantico.
   - Selezione finale dei **Top-5** documenti.

---

### 3.3 Generative Pipeline and Flow Control  
**(Membri 1 & 4)**

#### LLM Configuration
- **Llama 3.1 8B**.
- Quantizzazione **4-bit (NF4)** per rientrare nei limiti di VRAM delle T4.

#### Self-Correction Mechanisms
- **Grading Node**:
  - verifica della rilevanza dei documenti recuperati.
- **Hallucination Check**:
  - controllo post-generazione per validare la risposta.

#### Query Rewriting
- Gestione della memoria conversazionale.
- Trasformazione di query ambigue:
  - *“Quanto costa?”*  
  → *“Quanto costa [Oggetto X]?”*

---

## 4. Experimental Setup

Sezione dedicata alla riproducibilità.

### Hardware Constraints
- Utilizzo di **2× NVIDIA T4 GPU**.
- Impiego di:
  - `accelerate`
  - `bitsandbytes`
per abilitare l’esecuzione dei modelli.

### Software Stack
- LangGraph
- Qdrant
- HuggingFace Transformers

### Dataset
- Descrizione del dataset **mtRAG** utilizzato nella challenge.

---

## 5. Evaluation and Results  
**(Membro 5)**

### Evaluation Protocol
- Utilizzo di un **Local Judge** (Llama 3 stesso).
- Nessun uso di GPT-4, per coerenza con l’approccio open-source.

### Metrics
- Faithfulness
- Answer Relevancy
- Context Precision
- Refusal Accuracy (capacità di rispondere correttamente *I_DONT_KNOW*)

### Quantitative Results
- Tabelle riassuntive dei risultati ottenuti.

---

## 6. Ablation Studies / Discussion

Sezione opzionale ma fortemente apprezzata.

### Impact of Reranking
- Confronto tra retrieval con e senza Cross-Encoder.

### Impact of Parent–Child Chunking  
*(Tua opportunità diretta)*

- Discussione su come la strategia Parent–Child:
  - abbia migliorato il recupero contestuale,
  - ridotto la frammentazione informativa,
  - prodotto risposte più coerenti rispetto al chunking standard.

---

## 7. Conclusion

Conclusione del lavoro:

- Dimostrazione che un’architettura **agentica e ciclica (Self-CRAG)**,
- se adeguatamente ingegnerizzata,
- può competere su task complessi di multi-turn RAG anche con:
  - hardware limitato,
  - modelli open-source,

grazie a:
- strategie intelligenti di gestione dei dati,
- controllo esplicito del flusso,
- meccanismi di auto-correzione.

---
