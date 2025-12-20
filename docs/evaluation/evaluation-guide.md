# Manuale Completo: SemEval 2026 Task 8 - Valutazione e Sottomissione

Questa è la guida definitiva e onnicomprensiva per partecipare alla competizione. Questo documento non è una semplice lista di comandi, ma un manuale operativo dettagliato che spiega la logica, le motivazioni e i dettagli tecnici dietro ogni singola operazione. Copre ogni aspetto del ciclo di vita del progetto: dalla preparazione teorica e pratica dei dati, all'esecuzione ottimizzata della pipeline, fino alla verifica analitica dei risultati e alla sottomissione finale. Include inoltre sezioni approfondite sul funzionamento interno del sistema RAG e dell'architettura LangGraph.

---

## 📚 Indice Dettagliato
1.  [Fase 0: Verifica dei Dati (Knowledge Base)](#fase-0-verifica-dei-dati-knowledge-base)
2.  [Fase 1: Preparazione dell'Ambiente Operativo](#fase-1-preparazione-dellambiente-operativo)
3.  [Fase 2: Esecuzione della Pipeline (Generazione Risultati)](#fase-2-esecuzione-della-pipeline-generazione-risultati)
4.  [Fase 3: Validazione Formale (Format Checker)](#fase-3-validazione-formale-format-checker)
5.  [Fase 4: Valutazione Locale (Calcolo Metriche)](#fase-4-valutazione-locale-calcolo-metriche)
6.  [Fase 5: Sottomissione Ufficiale](#fase-5-sottomissione-ufficiale)
7.  [Approfondimento Tecnico: Il Motore del Sistema](#approfondimento-tecnico-il-motore-del-sistema)
8.  [Architettura LangGraph Pipeline: Flusso Decisionale](#architettura-langgraph-pipeline-flusso-decisionale)

---

## Fase 0: Verifica dei Dati (Knowledge Base)

Prima ancora di scrivere o eseguire una riga di codice, dobbiamo stabilire le fondamenta del progetto: i dati. Un sistema RAG (Retrieval-Augmented Generation) è, per definizione, dipendente dalla sua base di conoscenza. Se questa manca o è corrotta, il sistema non può funzionare, indipendentemente dalla bontà del codice.

Il primo passo fondamentale è quindi un audit del filesystem per assicurarsi che tutte le risorse necessarie siano presenti e correttamente posizionate. È essenziale verificare la presenza della cartella `dataset/human`. Questa cartella deve contenere i "compiti" che l'AI dovrà svolgere, ovvero i file JSONL con le domande. In particolare, durante la fase di sviluppo, il file cruciale è `dataset/human/generation_tasks/reference.jsonl`. Questo file contiene coppie domanda-risposta verificate da esseri umani e serve come "Gold Standard" per misurare le prestazioni del sistema prima della gara.

Parallelamente, è vitale controllare la cartella `dataset/corpora`. Questa non è una semplice cartella, ma rappresenta la "memoria a lungo termine" del nostro assistente. Deve ospitare i documenti originali della Knowledge Base, che sono stati suddivisi dagli organizzatori in quattro domini specifici: `clapnq` (fatti generali), `cloud` (manualistica tecnica), `fiqa` (finanza), e `govt` (documenti legali). Assicurati che all'interno ci siano i file decompressi o le sottocartelle corrispondenti.

> [!WARNING]
> **Attenzione Critica**: Se la cartella `dataset/corpora` dovesse risultare vuota o parziale, il modulo di Retrieval (utilizzato obbligatoriamente nei Task A e C) cercherà invano. Il risultato sarà una lista di documenti vuota, che a sua volta porterà il generatore a non avere contesto, causando risposte "I don't know" o allucinazioni. Questo scenario garantisce un punteggio vicino allo zero nella valutazione ufficiale.

---

## Fase 1: Preparazione dell'Ambiente Operativo

La riproducibilità è chiave nella scienza dei dati. L'ambiente di lavoro deve essere configurato in modo deterministico, con tutte le dipendenze corrette installate prima di eseguire qualsiasi script. Questo previene errori oscuri legati a versioni incompatibili di librerie.

Inizia aprendo il terminale nella root del progetto (`llm-semeval-task8/`). Il primo comando deve essere l'attivazione del tuo virtual environment (ad esempio con `source .venv/bin/activate` o `conda activate ...`). Un indicatore visivo del successo sarà la comparsa del nome dell'ambiente tra parentesi nel prompt del terminale. Non sottovalutare questo passaggio: eseguire script installando pacchetti nell'ambiente globale del sistema operativo è una pratica rischiosa che può compromettere l'intero sistema.

Successivamente, dobbiamo occuparci delle dipendenze specifiche per la valutazione. Gli script ufficiali forniti da IBM non si limitano alle librerie standard; richiedono strumenti specifici per il calcolo delle metriche NLP, come `ragas` (per la valutazione RAG), `scikit-learn` (per metriche statistiche) o `rouge_score`. Esegui il comando `pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt`. L'uso del file `constraints.txt` è importante perché forza l'installazione di versioni specifiche testate dagli organizzatori, garantendo che i tuoi risultati siano comparabili con quelli ufficiali. Se durante l'installazione riscontri problemi con `torch` (spesso dovuto a conflitti con CUDA), ti consigliamo di installarlo manualmente (`pip install torch==2.1.2`) prima di procedere, isolando così il problema.

---

## Fase 2: Esecuzione della Pipeline (Generazione Risultati)

Una volta che i dati sono pronti e l'ambiente è stabile, passiamo all'azione: generare le risposte del modello. Invece di eseguire script separati e frammentati, utilizzeremo il notebook unificato **`notebooks/All_Tasks_Pipeline.ipynb`**. Questo strumento è stato ingegnerizzato per ottimizzare il flusso di lavoro, eseguendo la RAG Pipeline una sola volta per ogni domanda e distribuendo poi i risultati ai vari formati di output richiesti dai Task A, B e C. Questo approccio "Single Pass" riduce drasticamente i tempi di calcolo e i costi (se si usano API a pagamento).

Apri il notebook e portati alla seconda cella per configurare le variabili principali. Dovrai impostare `TEAM_NAME` con il nome ufficiale del tuo team; questo identificativo verrà usato per nominare i file di output, facilitando la gestione delle versioni. Verifica attentamente che `INPUT_FILE` punti al file corretto: usa `dataset/human/generation_tasks/reference.jsonl` per i test locali di sviluppo. Quando inizierà la competizione, dovrai solo cambiare questo percorso per puntare al nuovo file di test cieco.

Una volta configurato, avvia l'esecuzione di tutte le celle ("Run All"). Il notebook non è una scatola nera: ti mostrerà una barra di avanzamento mentre processa le domande. Per ogni input, invocherà la funzione `app.invoke` del grafo LangGraph. Al termine dell'elaborazione, estrarrà chirurgicamente i dati necessari: prenderà solo i contesti recuperati per il file del Task A, solo il testo generato per il file del Task B, ed entrambi per il Task C. Infine, salverà tre file distinti con estensione `.jsonl` nella cartella `data/submissions/`, pronti per la fase successiva.

---

## Fase 3: Validazione Formale (Format Checker)

Nel contesto di una competizione internazionale, la forma è sostanza. La generazione dei file è solo metà dell'opera; è assolutamente cruciale che questi rispettino il formato rigoroso imposto dagli organizzatori fino all'ultima virgola. Un campo mancante, un nome sbagliato o una struttura JSON non valida possono portare all'invalidazione immediata della sottomissione, rendendo inutile tutto il lavoro svolto.

Per evitare questo rischio, utilizza sempre lo script ufficiale `format_checker.py` per validare ogni file generato. Non fidarti della "ispezione visiva". Esegui i comandi validazione dal terminale per ogni task (Task A, B, C), assicurandoti di puntare ai file appena creati in `data/submissions/`. Lo script analizza la struttura del JSONL riga per riga.

Se l'output conferma "Validation Successful!", puoi tirare un sospiro di sollievo: i file sono sintatticamente corretti. In caso di errori, lo script è molto verboso e ti indicherà la riga esatta e la natura del problema (es. "Campo 'score' mancante nell'oggetto context alla riga 42"). Questo ti permette di tornare al notebook, correggere la logica di estrazione dei dati e rigenerare i file finché non sono perfetti.

---

## Fase 4: Valutazione Locale (Calcolo Metriche)

Fino alla data di inizio della competizione (10 Gennaio), non avrai feedback esterni. La tua bussola sarà la valutazione locale sul dataset di sviluppo. Questo ti permette di iterare velocemente: modifica il prompt, rigenera, valuta, ripeti.

Per il **Task A (Retrieval)**, esegui `run_retrieval_eval.py`. Questo script confronta i documenti trovati dal tuo sistema con la lista dei documenti "Gold" definiti dagli esperti. Calcolerà metriche come la **Recall@5** (che misura la completezza: quanti dei documenti rilevanti totali sono stati trovati nei primi 5?) e il **nDCG@5** (che misura l'ordine: i documenti rilevanti sono in cima alla lista, dove l'utente guarda per primo, o in fondo?). Un punteggio nDCG alto indica un motore di ricerca di alta qualità.

Per i **Task B e C (Generation)**, la valutazione è più complessa perché non esiste una sola risposta esatta in linguaggio naturale. Richiede quindi un "LLM Giudice" che agisca come un professore umano. Esegui `run_generation_eval.py` specificando un modello locale potente (come `ibm-granite/granite-3.3-8b-instruct`) o un provider API. Le metriche prodotte sono sofisticate: la **Faithfulness** misura se il modello ha inventato informazioni non presenti nei documenti (allucinazioni); l'**Answer Relevance** misura se la risposta è pertinente alla domanda posta; la **Correctness** verifica l'accuratezza fattuale rispetto alla risposta di riferimento.

---

## Fase 5: Sottomissione Ufficiale

A partire dal 10 Gennaio 2026, la procedura cambierà leggermente per la gara ufficiale. In quella data, gli organizzatori rilasceranno il **Test Set Ufficiale**. Questo file conterrà solo le domande, senza le risposte o i documenti di riferimento.

<<<<<<< HEAD
Il workflow sarà:
1. Scaricare il nuovo file.
2. Aggiornare la variabile `INPUT_FILE` nel notebook `All_Tasks_Pipeline.ipynb` per puntare a questo nuovo file.
3. Eseguire la pipeline *senza* guardare i risultati (perché non avrai la ground truth).
4. Validare con il `format_checker.py`.
5. Caricare i file sulla piattaforma ufficiale.
=======
Il workflow sarà: scaricare il nuovo file, aggiornare la variabile `INPUT_FILE` nel notebook `All_Tasks_Pipeline.ipynb` per puntare ad esso, eseguire la pipeline senza guardare i risultati (perché non avrai la ground truth), validare con il `format_checker.py`, e caricare i file sulla piattaforma ufficiale.
>>>>>>> main

Ricorda il principio fondamentale del Machine Learning: non modificare mai il sistema basandoti sui risultati del Test Set (overfitting). La valutazione finale avverrà su dati mai visti prima, quindi un sistema robusto e generalizzabile vincerà su uno iper-ottimizzato per il training set.

---

## Approfondimento Tecnico: Il Motore del Sistema

Per comprendere a fondo il funzionamento del sistema e poterlo migliorare, dobbiamo aprire il cofano e analizzare i tre componenti principali: la strategia di Retrieval, la natura della Knowledge Base e il processo Generativo.

### 1. Il Modulo Retrieval (`src/retrieval.py`): Come troviamo l'ago nel pagliaio
<<<<<<< HEAD
Questo modulo è il cuore pulsante della pipeline RAG. Il processo non è una semplice ricerca per parole chiave (come Ctrl+F), ma una ricerca semantica avanzata.
Innanzitutto, la domanda dell'utente viene trasformata in un vettore numerico multi-dimensionale (embedding) tramite il modello **`BAAI/bge-m3`**. Questo permette di cercare per significato, non solo per keyword esatte.
Successivamente, Qdrant (il database vettoriale) trova i documenti più simili. Per raffinare questa lista grezza, utilizziamo un modello Cross-Encoder molto preciso, il **`BAAI/bge-reranker-v2-m3`**, che riordina i risultati assegnando un punteggio di qualità. Solo i top-5 passano al generatore.
=======
Questo modulo è il cuore pulsante della pipeline RAG. Il processo non è una semplice ricerca per parole chiave (come Ctrl+F), ma una ricerca semantica avanzata. Innanzitutto, la domanda dell'utente viene trasformata in un vettore numerico multi-dimensionale (embedding) tramite il modello **`BAAI/bge-m3`**. Questo permette di cercare per significato, non solo per keyword esatte. Successivamente, Qdrant (il database vettoriale) trova i documenti più simili. Per raffinare questa lista grezza, utilizziamo un modello Cross-Encoder molto preciso, il **`BAAI/bge-reranker-v2-m3`**, che riordina i risultati assegnando un punteggio di qualità. Solo i top-5 passano al generatore.
>>>>>>> main

### 2. I Documenti Recuperabili (`dataset/corpora`): I limiti della conoscenza
Il sistema opera su una Knowledge Base chiusa, limitata ai documenti presenti in `dataset/corpora`. Non c'è accesso a Internet. La libreria digitale copre quattro domini: **CLAPNQ** (cultura generale), **FiQA** (finanza), **Govt** (legale) e **Cloud** (tecnico).

### 3. La Valutazione (Dietro le quinte): Come ragionano i giudici
La valutazione per i Task B e C (Generazione) si affida a un LLM Giudice che valuta la semantica della risposta. Nello specifico, gli script di valutazione locale utilizzano il modello **`ibm-granite/granite-3.3-8b-instruct`** per simulare il giudizio umano su metriche come Faithfulness e Correctness.

---

<<<<<<< HEAD
## Architettura LangGraph Pipeline: Flusso Decisionale e Gestione dei Casi Limite 🧠

Un sistema RAG avanzato non è una sequenza lineare di istruzioni. Va immaginato come un agente intelligente capace di prendere decisioni, valutare il proprio operato e correggere la rotta se necessario. Questa logica complessa è orchestrata da **LangGraph**, che gestisce il flusso attraverso nodi funzionali e ramificazioni decisionali. Di seguito descriviamo narrativamente il percorso di una richiesta utente, specificando i modelli utilizzati.

Il viaggio inizia nel nodo **`rewrite_node`**, che agisce come un traduttore intelligente. Spesso gli utenti formulano domande ambigue o dipendenti dal contesto (es. "Chi è lui?"). Qui utilizziamo il modello **`meta-llama/Llama-3.1-8B-Instruct`** per riscrivere la query rendendola autonoma ed esplicita (es. "Chi è Elon Musk?"). Se la domanda è già chiara, il modello la lascia invariata, gestendo così l'edge case dell'identità.

L'output passa al **`retrieve_node`**, il bibliotecario del sistema. Utilizzando il modello di embedding **`BAAI/bge-m3`** e il reranker **`BAAI/bge-reranker-v2-m3`**, questo nodo interroga il database Qdrant per recuperare i documenti più pertinenti. Un caso limite importante qui è il fallimento del retrieval: se il database è vuoto o offline, il nodo restituisce una lista vuota, preparando il sistema a un possibile fallback.

I documenti recuperati vengono poi esaminati dal **`grade_documents_node`**, un filtro di qualità critico. Anche qui ci affidiamo a **`meta-llama/Llama-3.1-8B-Instruct`**, che legge ogni documento e decide se contiene informazioni utili. Questo passaggio è fondamentale per filtrare il "rumore". Se tutti i documenti vengono scartati come irrilevanti, il grafo attiva un bivio decisionale intelligente: invece di procedere, devia verso il fallback per rispondere "Non lo so", evitando di inventare informazioni.

Se invece i documenti superano il filtro, il **`generate_node`** entra in azione. Sempre guidato da **`meta-llama/Llama-3.1-8B-Instruct`**, questo nodo compone la risposta finale utilizzando esclusivamente le informazioni verificate.

L'ultimo baluardo prima dell'utente è il **`hallucination_check_node`** (Self-RAG). Qui, un'altra istanza di **`meta-llama/Llama-3.1-8B-Instruct`** agisce come un fact-checker paranoico, confrontando ogni frase generata con i documenti di supporto. Se rileva un'allucinazione (un'informazione non supportata dal testo), blocca la risposta e attiva il fallback. Solo le risposte verificate e fedeli raggiungono l'utente finale.
=======
## Architettura LangGraph Pipeline: Flusso Decisionale con Retry Loop 🧠

Un sistema RAG avanzato non è una sequenza lineare di istruzioni. Va immaginato come un agente intelligente capace di prendere decisioni, valutare il proprio operato e correggere la rotta se necessario. Questa logica complessa è orchestrata da **LangGraph**, che gestisce il flusso attraverso nodi funzionali e ramificazioni decisionali.

La caratteristica distintiva del nostro sistema è il **Retry Loop**: quando viene rilevata un'allucinazione, invece di arrendersi immediatamente, il sistema ritenta la generazione fino a un massimo di `MAX_RETRIES` volte (default: 2, quindi fino a 3 tentativi totali). Questo massimizza la qualità delle risposte anche con modelli più piccoli come Llama 3.1 8B.

Il viaggio inizia nel nodo **`rewrite_node`**, che agisce come un traduttore intelligente. Spesso gli utenti formulano domande ambigue o dipendenti dal contesto (es. "Chi è lui?"). Qui utilizziamo il modello **`meta-llama/Llama-3.1-8B-Instruct`** per riscrivere la query rendendola autonoma ed esplicita (es. "Chi è Elon Musk?"). Se la domanda è già chiara, il modello la lascia invariata.

L'output passa al **`retrieve_node`**, il bibliotecario del sistema. Utilizzando il modello di embedding **`BAAI/bge-m3`** e il reranker **`BAAI/bge-reranker-v2-m3`**, questo nodo interroga il database Qdrant per recuperare i documenti più pertinenti. In questa fase, il contatore `retry_count` viene azzerato a 0.

I documenti recuperati vengono poi esaminati dal **`grade_documents_node`**, un filtro di qualità critico. Anche qui ci affidiamo a **`meta-llama/Llama-3.1-8B-Instruct`**, che legge ogni documento e decide se contiene informazioni utili. Se tutti i documenti vengono scartati come irrilevanti, il grafo devia direttamente verso il fallback.

Se invece i documenti superano il filtro, il **`generate_node`** entra in azione. Sempre guidato da **`meta-llama/Llama-3.1-8B-Instruct`**, questo nodo compone la risposta finale utilizzando esclusivamente le informazioni verificate.

L'ultimo baluardo prima dell'utente è il **`hallucination_check_node`** (Self-RAG). Qui, un'altra istanza di **`meta-llama/Llama-3.1-8B-Instruct`** agisce come un fact-checker paranoico. Se rileva un'allucinazione, il sistema non si arrende subito: controlla prima il contatore `retry_count`. Se `retry_count < MAX_RETRIES`, il flusso passa a **`increment_retry_node`** che incrementa il contatore e ritorna a **`generate_node`** per un nuovo tentativo. Se invece `retry_count >= MAX_RETRIES`, il sistema ha esaurito i tentativi e attiva il **`fallback_node`** restituendo "I_DONT_KNOW". Solo le risposte verificate e fedeli raggiungono l'utente finale.
>>>>>>> main

### Riassunto Flusso Visivo
```mermaid
graph TD
    Start --> Rewrite
    Rewrite --> Retrieve
    Retrieve --> GradeDocs
<<<<<<< HEAD
    GradeDocs -- Documenti OK? --> Generate
    GradeDocs -- Nessun Doc? --> Fallback
    Generate --> HallucinationCheck
    HallucinationCheck -- Allucinazione? --> Fallback
    HallucinationCheck -- Fedele? --> END
```
=======
    GradeDocs -- Documenti OK --> Generate
    GradeDocs -- Nessun Doc --> Fallback
    Generate --> HallucinationCheck
    HallucinationCheck -- Grounded --> END
    HallucinationCheck -- Hallucinated + Retries Left --> IncrementRetry
    IncrementRetry --> Generate
    HallucinationCheck -- Hallucinated + Max Retries --> Fallback
    Fallback --> END
```

### Configurazione Retry Loop
Il parametro `MAX_RETRIES` in `src/graph.py` controlla quanti tentativi di rigenerazione sono consentiti dopo il rilevamento di un'allucinazione. Il valore di default è `2`, il che significa che il generatore può essere invocato fino a 3 volte totali prima del fallback.
>>>>>>> main
