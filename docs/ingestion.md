Ecco la trascrizione del documento fornito in formato Markdown, organizzata per sezioni linguistiche come presente nel file originale.

# Bozza del Progetto RAG (English Version)

## 1. Introduction

The transition of Retrieval-Augmented Generation (RAG) toward conversational paradigms, driven by recent developments in Large Language Models (LLMs), has introduced new challenges related to context persistence that are often unmanageable for legacy systems focused on single-turn interactions. This transition, however, has highlighted the inadequacy of conventional RAG architectures in handling the complexity of multi-turn dialogues. It is in this scenario that our project is situated, aiming to measure the effectiveness of models in preserving contextual coherence and ensuring the accuracy of information retrieval during prolonged interactions.

### 1.1 The Challenge: Context Decay and Hallucination

The central problem addressed in this work is not merely information retrieval, but rather the management of the system's "Memory and Honesty". Conventional RAG systems, based on linear chains, suffer from two main pathologies:

* 
**Context Decay:** The loss of anaphoric references during dialogue. For example, a question such as "And how old is he?" becomes incomprehensible to a retriever if it is not correctly linked to the entity mentioned in previous turns.


* 
**Hallucinations and Refusal Accuracy:** The intrinsic tendency of LLMs, trained to be "helpful," to invent plausible answers even when retrieved documents do not contain the requested information. In this task, the ability to respond with "I_DONT_KNOW" (Refusal Accuracy) is as crucial as the precision of the answer itself.



### 1.2 Resource Constraints and the "Sovereign" Approach

Unlike approaches that rely on proprietary models via API (e.g., GPT-4), our participation distinguishes itself through a rigorous operational constraint: the entire system is designed to operate in a "Sovereign" and Open Source environment, utilizing exclusively local resources on limited hardware. This constraint mandates the abandonment of massive or full-precision (FP16) models in favor of aggressive optimization strategies, such as 4-bit NF4 quantization, without sacrificing the reasoning capabilities necessary for the task.

### 1.3 Proposed Solution: A Self-CRAG Architecture

To overcome the limitations of linearity and hardware restrictions, we propose a Self-Corrective RAG (Self-CRAG) architecture implemented via a State Graph instead of a sequential chain. Our system integrates a quantized Llama-3.1-8B LLM that orchestrates a cyclic flow of Query Rewriting, Retrieval, Grading, and Self-Reflection.

## X. Data Ingestion and Context-Aware Indexing Strategy

To address the challenges of Multi-Turn RAG, specifically Context Decay and Hallucination in resource-constrained environments, we designed a specialized ingestion pipeline. Our approach prioritizes the preservation of semantic context over simple keyword matching. This section details our implementation of Parent-Child Chunking and our vector storage strategy.

### X.1 Corpus Processing and Metadata Preservation

The mtRAG benchmark provides data in JSONL format, containing diverse domains (e.g., finance, government, cloud). The initial phase of our pipeline, implemented in the `load_and_chunk_data` module, focuses on data integrity and metadata extraction. Unlike standard RAG pipelines that often discard metadata, we explicitly extract and preserve the title, url, and source fields for every document. This metadata is not merely archival; it is injected into the retrieval context to allow the generation model (Llama 3.1) to cite sources accurately, thereby improving the Faithfulness metric.

### X.2 Addressing Context Fragmentation: The Parent-Child Chunking Strategy

A critical failure mode in standard RAG systems is "Context Fragmentation": when documents are split into small chunks (e.g., 200-400 tokens) to optimize vector search precision, crucial context informations, such as the subject of a sentence or the antecedent of a pronoun, is often lost in the cut. For example, a chunk might contain the text "He signed the act," but the identity of "He" (e.g., "The President") resides in the previous chunk. This makes the chunk invisible to vector searches querying "The President".

In order to address this problem without saturating the limited VRAM with massive context windows, we implemented a Parent-Child Chunking strategy using `RecursiveCharacterTextSplitter`. This involves a dual-layer approach:

1. 
**Parent Chunks (Context Layer):** The raw text is first split into large "Parent" chunks of 1200 characters (with 100 char overlap). This size was selected to ensure that a coherent narrative or logical argument is fully contained within a single block, facilitating the LLM's reasoning and anaphora resolution capabilities.


2. 
**Child Chunks (Search Layer):** Each Parent chunk is further subdivided into smaller "Child" chunks of 400 characters (with 50 char overlap). These smaller units are optimized for dense embedding models, which perform best on focused, semantically dense text segments rather than long, diluted paragraphs.



**Indexing Mechanism:** Crucially, we index the Child Chunk vectors for retrieval but store the Parent Chunk text in the metadata (`c_chunk.metadata["parent_text"]`). During the retrieval phase, when a Child chunk is identified as relevant via vector similarity, the system retrieves and feeds the full Parent Content to the LLM. This hybrid approach combines the high precision of small-chunk retrieval with the high context of large-chunk generation.

### X.3 Vector Representation and Efficient Storage

Given the hardware constraints, efficient resource management was paramount.

* 
**Embedding Model:** We selected `BAAI/bge-m3` as our embedding model. This model was chosen for its state-of-the-art performance in dense retrieval tasks and its ability to handle multi-granularity inputs efficiently. More in detail, this capability is critical for our architecture, as the system must bridge the gap between user queries which are often brief and synthetic (e.g., 'Who is the CEO?'), and the longer, more discursive 400-character 'Child Chunks' created during the ingestion module. To prevent VRAM saturation during the ingestion phase, we implemented a singleton pattern, ensuring only one instance of the model is loaded into memory, and utilized CPU offloading for the encoding process where necessary.


* 
**Vector Store:** We utilized Qdrant as our vector database. Qdrant was chosen for its ability to handle payload filtering efficiently and its support for local file persistence, which eliminated the overhead of maintaining a separate Docker container during the development phase. The collection uses Cosine Similarity to measure the semantic orientation between query and document vectors.



By decoupling the search unit (Child) from the generation unit (Parent), our ingestion strategy directly mitigates the risk of retrieving out-of-context information, laying the foundation for the "Self-CRAG" architecture to operate with high accuracy.

## X. Hybrid Retrieval and Reranking Strategy

While the ingestion phase ensures data integrity via Parent-Child chunking, the retrieval phase must address the semantic ambiguity typical of multi-turn dialogues and distinguish subtle nuances in regulatory or technical queries. Simple vector search often fails in these contexts as it "compresses" meaning based on terminological overlap: vectors for opposing concepts such as "tax exemption" and "tax application" appear mathematically nearly identical (high cosine similarity), leading to the retrieval of incorrect documents. To overcome this limitation, we implemented a two-stage "Retrieve-and-Rerank" pipeline designed to maximize Recall in the first stage and Precision in the second.

### X.2.1 Stage 1: Dense Candidate Generation (High Recall)

The first stage acts as a broad filter. We utilize the `BAAI/bge-m3` embedding model to query the Qdrant vector store. Instead of retrieving only the few strictly relevant documents, we configured the retriever to fetch a broader set of Top-20 candidates (`top_k_retrieve=20`). This "High Recall" strategy is deliberate: in complex queries, the "correct" answer might not have the highest cosine similarity score due to vocabulary mismatches. By casting a wider net, we ensure that the relevant document is included in the initial pool, even if it ranks 15th or 20th purely based on vector distance.

### X.2.2 Stage 2: Cross-Encoder Refinement (High Precision)

To filter the noise from the top 20 candidates, we employ a Cross-Encoder Reranker using the `BAAI/bge-reranker-v2-m3` model. Unlike Bi-Encoders, such as the vector search performed by Qdrant (which is fast but "blind" as it processes query and document independently), the Cross-Encoder processes the query and document simultaneously within the transformer's self-attention mechanism. This allows the model to capture deep semantic dependencies, such as negation or causality, that vector similarity often misses: the model distinguishes phrases like "The user can access" from "The user cannot access", where the mere presence of the particle "not" reverses relevance despite the high lexical similarity of the two sentences. The reranker assigns a relevance score to each of the 20 candidates and selects only the Top-5 (`top_k_rerank=5`) for the generation phase. This reduces the usage of the LLM's context window while guaranteeing that the input context is of the highest quality.

### X.2.3 Context Reconstruction and Parent-Swap

A critical component of our module, implemented in `format_docs_for_gen`, is the reconstruction of the full context. Although retrieval and reranking are performed on granular Child Chunks (400 chars), feeding these fragments to the LLM would result in disjointed answers. Therefore, before generation, the module performs a "Parent-Swap": it reads the `parent_text` metadata from the retrieved chunks and deduplicates them. This ensures that the Llama-3.1 model receives the full and coherent Parent Context (1200 chars), enabling it to answer complex questions that span across multiple sentences.

### X.2.4 Resource Optimization: The "CPU Offload" Strategy

Operating within the strict constraints of the provided hardware (2x T4 GPUs), we adopted a resource-aware allocation strategy. In our implementation, both the Embedding Model and the Reranker are explicitly offloaded to the CPU (`device="cpu"`). While this introduces a marginal latency overhead compared to GPU execution, it reserves the entirety of the VRAM for the generation model Llama-3.1 (heavy due to quantization) and its KV-cache. Additionally, we implemented a Singleton Pattern for the Qdrant client and models to prevent memory leaks and lock conflicts during graph execution.

---

# TRADUZIONE (Italian Version)

## 1. Introduzione

La transizione del Retrieval-Augmented Generation (RAG) verso paradigmi conversazionali, spinta dai recenti sviluppi degli LLM, ha introdotto nuove sfide legate alla persistenza del contesto, spesso ingestibili per i sistemi legacy focalizzati su interazioni singole. Tale transizione, tuttavia, ha messo in luce l'inadeguatezza delle architetture RAG convenzionali nel gestire la complessità dei dialoghi multi-turno. È in questo scenario che si inserisce il nostro progetto, che cerca misurare l'efficacia dei modelli nel preservare la coerenza contestuale e garantire l'accuratezza del recupero delle informazioni durante interazioni prolungate.

### 1.1 The Challenge: Context Decay and Hallucination

Il problema centrale affrontato in questo lavoro non è meramente il recupero dell'informazione, quanto la gestione della "Memoria e Onestà" del sistema. I sistemi RAG convenzionali, basati su catene lineari (Linear Chains), soffrono di due patologie principali:

* 3. Context Decay (Decadimento del Contesto): La perdita di riferimenti anaforici durante il dialogo. Ad esempio, una domanda come "E quanti anni ha?" diventa incomprensibile per un retriever se non viene correttamente collegata all'entità menzionata nei turni precedenti.


* 4. Allucinazioni e Refusal Accuracy: La tendenza intrinseca degli LLM, addestrati per essere "utili", a inventare risposte plausibili anche quando i documenti recuperati non contengono le informazioni richieste. In questo task, la capacità di rispondere "I_DONT_KNOW" (Refusal Accuracy) è tanto cruciale quanto la precisione della risposta stessa.



### 1.2 Resource Constraints and the "Sovereign" Approach

A differenza di approcci che si affidano a modelli proprietari via API (es. GPT-4), la nostra partecipazione si distingue per un rigoroso vincolo operativo: l'intero sistema è progettato per operare in un ambiente "Sovereign" e Open Source, utilizzando esclusivamente risorse locali su hardware limitato. Questo vincolo impone l'abbandono di modelli massivi o a precisione piena (FP16) in favore di strategie di ottimizzazione aggressive, come la quantizzazione NF4 a 4-bit, senza sacrificare le capacità di ragionamento necessarie per il task.

### 1.3 Proposed Solution: A Self-CRAG Architecture

Per superare i limiti della linearità e le restrizioni hardware, proponiamo un'architettura Self-Corrective RAG (Self-CRAG) implementata tramite un Grafo di Stato (State Graph) invece di una catena sequenziale. Il nostro sistema integra un LLM Llama-3.1-8B quantizzato che orchestra un flusso ciclico di Query Rewriting, Retrieval, Grading e Self-Reflection.

## X. Metodologia: Strategia di Ingestione Dati e Indicizzazione

Per affrontare le sfide del Multi-Turn RAG, nello specifico il Decadimento del Contesto (Context Decay) e le Allucinazioni in ambienti a risorse limitate, abbiamo progettato una pipeline di ingestione specializzata. Il nostro approccio privilegia la conservazione del contesto semantico rispetto alla semplice corrispondenza per parole chiave. Questa sezione descrive in dettaglio l'implementazione del Parent-Child Chunking e la nostra strategia di archiviazione vettoriale.

### X.1 Elaborazione del Corpus e Conservazione dei Metadati

Il benchmark mtRAG fornisce dati in formato JSONL, comprendenti diversi domini (es. finanza, governo, cloud). La fase iniziale della nostra pipeline, implementata nel modulo `load_and_chunk_data`, si concentra sull'integrità dei dati e sull'estrazione dei metadati. A differenza delle pipeline RAG standard che spesso scartano i metadati, noi estraiamo e preserviamo esplicitamente i campi title, url e source per ogni documento. Questi metadati non hanno solo scopo di archiviazione; vengono iniettati nel contesto di retrieval per consentire al modello di generazione (Llama 3.1) di citare accuratamente le fonti, migliorando così la metrica di Faithfulness.

### X.2 Risoluzione della Frammentazione del Contesto: La Strategia Parent-Child Chunking

Una modalità di fallimento critica nei sistemi RAG standard è la "Frammentazione del Contesto": quando i documenti vengono suddivisi in piccoli chunk (es. 200-400 token) per ottimizzare la precisione della ricerca vettoriale, informazioni contestuali cruciali, come il soggetto di una frase o l'antecedente di un pronome, vengono spesso perse nel taglio. Ad esempio, un chunk potrebbe contenere il testo "Egli ha firmato l'atto", ma l'identità di "Egli" (es. "Il Presidente") risiede nel chunk precedente. Ciò rende il chunk invisibile alle ricerche vettoriali che interrogano "Il Presidente".

Per risolvere questo problema senza saturare la VRAM limitata con finestre di contesto massive, abbiamo implementato una strategia di Parent-Child Chunking utilizzando `RecursiveCharacterTextSplitter`. Questo comporta un approccio a doppio livello:

1. 
**Parent Chunks (Livello di Contesto):** Il testo grezzo viene prima suddiviso in grandi chunk "Genitori" (Parent) di 1200 caratteri (con sovrapposizione di 100 caratteri). Questa dimensione è stata selezionata per garantire che una narrazione coerente o un argomento logico sia interamente contenuto in un singolo blocco, facilitando le capacità di ragionamento e di risoluzione delle anafore dell'LLM.


2. 
**Child Chunks (Livello di Ricerca):** Ogni Parent chunk viene ulteriormente suddiviso in "Figli" (Child) più piccoli di 400 caratteri (con sovrapposizione di 50 caratteri). Queste unità più piccole sono ottimizzate per i modelli di embedding densi, che performano meglio su segmenti di testo focalizzati e semanticamente densi piuttosto che su lunghi paragrafi diluiti.



**Meccanismo di Indicizzazione:** In modo cruciale, indicizziamo i vettori dei Child Chunk per il retrieval, ma salviamo il testo del Parent Chunk nei metadati (`c_chunk.metadata["parent_text"]`). Durante la fase di retrieval, quando un Child chunk viene identificato come rilevante tramite similarità vettoriale, il sistema recupera e fornisce l'intero Parent Content all'LLM. Questo approccio ibrido combina l'alta precisione del retrieval a chunk piccoli con l'alto contesto della generazione a chunk grandi.

### X.3 Rappresentazione Vettoriale e Archiviazione Efficiente

Dati i vincoli hardware, la gestione efficiente delle risorse è stata fondamentale.

* 
**Modello di Embedding:** Abbiamo selezionato `BAAI/bge-m3` come nostro modello di embedding. Questo modello è stato scelto per le sue prestazioni allo stato dell'arte nei task di retrieval denso e per la sua capacità di gestire efficientemente input multi-granularità. Più nel dettaglio, questa capacità è critica per la nostra architettura, in quanto il sistema deve colmare il divario tra le query utente, che sono spesso brevi e sintetiche (es. "Chi è il CEO?"), e i "Child Chunks" di 400 caratteri, più lunghi e discorsivi, creati durante il modulo di ingestione. Per prevenire la saturazione della VRAM durante la fase di ingestione, abbiamo implementato un pattern singleton, assicurando che solo un'istanza del modello venga caricata in memoria, e utilizzato l'offloading su CPU per il processo di encoding dove necessario.


* 
**Vector Store:** Abbiamo utilizzato Qdrant come database vettoriale. Qdrant è stato scelto per la sua capacità di gestire efficientemente il filtraggio dei payload e il supporto alla persistenza su file locale, che ha eliminato l'overhead di mantenere un container Docker separato durante la fase di sviluppo. La collezione utilizza la Similarità del Coseno per misurare l'orientamento semantico tra i vettori della query e del documento.



Disaccoppiando l'unità di ricerca (Child) dall'unità di generazione (Parent), la nostra strategia di ingestione mitiga direttamente il rischio di recuperare informazioni fuori contesto, ponendo le basi affinché l'architettura "Self-CRAG" operi con elevata accuratezza.

## X.2 Modulo di Retrieval Ibrido e Strategia di Reranking

Mentre la fase di ingestione garantisce l'integrità dei dati tramite il Parent-Child chunking, la fase di retrieval deve affrontare l'ambiguità semantica tipica dei dialoghi multi-turno ed essere capace di distinguere sfumature sottili in query normative o tecniche. Una semplice ricerca vettoriale spesso fallisce in questi contesti poiché "comprime" il significato basandosi sulla sovrapposizione terminologica: vettori per concetti opposti come "esenzione tasse" e "applicazione tasse" risultano matematicamente quasi identici (alta similarità del coseno), portando al recupero di documenti errati. Per superare questo limite, abbiamo implementato una pipeline a due stadi "Retrieve-and-Rerank" progettata per massimizzare il Recall (richiamo) nel primo stadio e la Precision (precisione) nel secondo.

### X.2.1 Stadio 1: Generazione Densa dei Candidati (Alto Recall)

Il primo stadio agisce come un filtro ampio. Utilizziamo il modello di embedding `BAAI/bge-m3` per interrogare il vector store Qdrant. Invece di recuperare solo i pochi documenti strettamente rilevanti, abbiamo configurato il retriever per estrarre un set più ampio di Top-20 candidati (`top_k_retrieve = 20`). Questa strategia di "Alto Recall" è intenzionale: in query complesse, la risposta "corretta" potrebbe non avere il punteggio di cosine similarity più alto a causa di discrepanze nel vocabolario. Gettando una rete più ampia, ci assicuriamo che il documento rilevante sia incluso nel pool iniziale, anche se si classifica 15° o 20° basandosi puramente sulla distanza vettoriale.

### X.2.2 Stadio 2: Raffinamento con Cross-Encoder (Alta Precisione)

Per filtrare il rumore dai primi 20 candidati, impieghiamo un Cross-Encoder Reranker utilizzando il modello `BAAI/bge-reranker-v2-m3`. A differenza dei Bi-Encoder, come la ricerca vettoriale che fa Qdrant (veloce ma "cieca" in quanto processa query e documento indipendentemente), il Cross-Encoder elabora la query e il documento simultaneamente all'interno del meccanismo di self-attention del transformer. Ciò consente al modello di catturare dipendenze semantiche profonde, come la negazione o la causalità, che la similarità vettoriale spesso perde: il modello distingue frasi come "L'utente può accedere" da "L'utente non può accedere", dove la sola presenza della particella "non" inverte la rilevanza nonostante l'alta similarità lessicale delle due frasi. Il reranker assegna un punteggio di rilevanza a ciascuno dei 20 candidati e seleziona solo i Top-5 (`top_k_rerank=5`) per la fase di generazione. Questo riduce l'utilizzo della context window per l'LLM garantendo al contempo che il contesto in input sia della massima qualità.

### X.2.3 Ricostruzione del Contesto e "Parent-Swap"

Un componente critico del nostro modulo, implementato in `format_docs_for_gen`, è la ricostruzione del contesto completo. Sebbene il retrieval e il reranking siano eseguiti sui granulari Child Chunks (400 car), fornire questi frammenti all'LLM risulterebbe in risposte sconnesse. Pertanto, prima della generazione, il modulo esegue un "Parent-Swap" (scambio col genitore): legge i metadati `parent_text` dai chunk recuperati ed elimina i duplicati. Questo assicura che il modello Llama-3.1 riceva il Parent Context completo e coerente (1200 car), permettendogli di rispondere a domande complesse che si estendono su più frasi.

### X.2.4 Ottimizzazione delle Risorse: La Strategia "CPU Offload"

Operando entro i rigidi vincoli dell'hardware fornito (2x GPU T4), abbiamo adottato una strategia di allocazione consapevole delle risorse. Nella nostra implementazione, sia il Modello di Embedding che il Reranker sono esplicitamente offloadati sulla CPU (`device="cpu"`). Sebbene ciò introduca un marginale overhead di latenza rispetto all'esecuzione su GPU, riserva l'intera VRAM per il modello di generazione Llama-3.1 (pesante a causa della quantizzazione) e la sua KV-cache. Inoltre, abbiamo implementato un Pattern Singleton per il client Qdrant e i modelli per prevenire memory leak e conflitti di lock durante l'esecuzione del grafo.

---