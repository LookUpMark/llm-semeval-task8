# Manuale Definitivo per Valutazione e Submission (Zero-to-Hero)

Questa guida è stata scritta per accompagnarti **passo dopo passo** attraverso l'intero processo di valutazione e sottomissione per il SemEval 2026. Segui le istruzioni nell'ordine esatto in cui sono presentate.

---
---

## Fase 0: Verifica Dati (Knowledge Base)

Prima di iniziare, assicurati che la cartella dei dati sia popolata.
Il Task A (Retrieval) e il Task C (RAG) richiedono i documenti originali ("corpora") per funzionare.

### 0.1 Controllo Cartelle
Verifica che le seguenti cartelle esistano e contengano file:
*   `dataset/human`: Contiene le domande di test (`reference.jsonl`).
*   `dataset/corpora`: Contiene i documenti da recuperare (es. `mt-rag-clapnq...`).

*Nota*: Se `dataset/corpora` è vuota, il retriever non troverà nulla! (L'abbiamo scaricata per te, ma controlla sempre).

---
## Fase 1: Preparazione dell'Ambiente

Prima di generare qualsiasi risultato, dobbiamo assicurarci che il tuo computer abbia tutti gli strumenti necessari installati.

### 1.1 Attivazione Environment
Apri il terminale nella cartella principale del progetto e attiva il tuo ambiente virtuale.
*Perché?* Gli script Python hanno bisogno delle librerie corrette per funzionare.

```bash
# Se usi venv
source .venv/bin/activate
# Se usi Conda
conda activate llm-semeval-task8
```
*Cosa succede:* Il nome del tuo environment apparirà all'inizio della riga di comando (es. `(.venv) user@pc:~$`).

### 1.2 Installazione Dipendenze di Valutazione
Abbiamo importato gli script ufficiali di IBM nella cartella `scripts/evaluation`. Questi script richiedono pacchetti specifici.

```bash
pip install -c scripts/evaluation/constraints.txt -r scripts/evaluation/requirements.txt
```
*Cosa succede:* Il terminale scaricherà e installerà librerie come `ragas` o `scikit-learn` necessarie per calcolare i punteggi. Se vedi scritte bianche che scorrono e finisce con "Successfully installed...", è andato tutto bene.

---

## Fase 2: Generazione delle Risposte (Submission)

Ora dobbiamo usare il tuo modello per rispondere alle domande del test set. Abbiamo creato un notebook automatico per questo.

### 2.1 Apri il Notebook di Submission
1. Avvia Jupyter Lab o VS Code.
2. Naviga nella cartella `eval/`.
3. Apri il file `submission_pipeline.ipynb`.

### 2.2 Configurazione del Notebook
Vai alla **Cella 2 ("Configurazione")** del notebook.
Modifica le seguenti variabili:
- `TEAM_NAME`: Scrivi il nome del tuo team (es. `"SuperRAGTeam"`).
- `TASK_NAME`: Lascia `"TaskB"` se stai facendo la generazione.
- `INPUT_FILE`: Assicurati che punti al file corretto (di default: `../dataset/human/generation_tasks/reference.jsonl`).

### 2.3 Esecuzione
Clicca su "Run All" o esegui le celle una per una.

*Cosa succede passo per passo nel notebook:*
1.  **Caricamento**: Legge il file JSONL originale. Ti dirà "Caricate X istanze".
2.  **Inferenza**: Eseguirà la tua RAG Pipeline per OGNI domanda. Vedrai una barra di progresso.
    *   *Nota*: Potrebbe volerci del tempo a seconda della velocità del tuo modello.
3.  **Salvataggio**: Creerà un NUOVO file in `data/submissions/` chiamato `submission_TaskB_SuperRAGTeam.jsonl`.
4.  **Auto-Validazione**: L'ultima cella farà un controllo preliminare. Se vedi una scritta verde "Validation Passed", procedi alla fase successiva.

---

## Fase 3: Validazione Ufficiale (Format Checker)

Anche se il notebook dice che è tutto ok, **DEVI** usare lo script ufficiale per essere sicuro al 100% che il file non venga rifiutato.

### 3.1 Esegui il comando di validazione
Torna al terminale (dove hai attivato l'ambiente). Esegui questo comando esatto, sostituendo il nome del file con quello appena creato:

```bash
python scripts/evaluation/format_checker.py \
    --input_file dataset/human/generation_tasks/reference.jsonl \
    --prediction_file data/submissions/submission_TaskB_MioTeam.jsonl \
    --mode generation_taskb
```

### 3.2 Analisi dell'Output
*   **Se va tutto bene**:
    Vedrai un output simile a:
    ```
    Loading file...
    Checking structure...
    Validation Successful! File is ready for submission.
    ```
*   **Se c'è un errore**:
    Lo script ti dirà esattamente cosa non va (es. "Missing prediction field at line 5").
    *Cosa fare:* Se succede, controlla il notebook `submission_pipeline.ipynb` e rieseguilo.

---

## Fase 4: Valutazione Locale (Opzionale ma Raccomandata)

Se vuoi sapere che punteggio fa il tuo modello PRIMA di inviarlo, puoi usare gli script di valutazione locale.

### 4.1 Valutazione Retrieval (Task A)
Se hai generato un file con i contesti recuperati:
```bash
python scripts/evaluation/run_retrieval_eval.py \
    --input_file data/submissions/submission_TaskA_MioTeam.jsonl \
    --output_file eval/results_retrieval.json
```
*Cosa succede:* Calcola Recall e nDCG. Troverai i risultati in `eval/results_retrieval.json`.

### 4.2 Valutazione Generation (Task B)
Questa fase richiede molta memoria (usa Llama come giudice).
```bash
python scripts/evaluation/run_generation_eval.py \
    --input_file data/submissions/submission_TaskB_MioTeam.jsonl \
    --output_file eval/results_generation.json \
    --config scripts/evaluation/config.yaml \
    --provider hf \
    --judge_model ibm-granite/granite-3.3-8b-instruct
```
*Cosa succede:* Il "modello giudice" leggerà le tue risposte e darà un voto.

---

## Fase 5: Invio (Submission)

1.  Vai al link del **Google Form** fornito sulla pagina della competizione (sarà attivo solo dal 12 Gennaio).
2.  Carica il file `.jsonl` che hai validato nella Fase 3.
3.  Clicca invio.

**Ricorda**: Vale solo l'ultima sottomissione. Invia solo quando sei sicuro.
