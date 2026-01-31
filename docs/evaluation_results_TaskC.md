# Task C Evaluation Results - Post-Fix Analysis

**Date:** 31 gennaio 2026  
**Submission File:** `submission_TaskC_Gbgers.jsonl`  
**Ground Truth:** `dataset/human/evaluations/reference.json`  
**Team:** Gbgers

---

## Methodology

### Metrics Used
- **Semantic Similarity:** Cosine similarity using `all-MiniLM-L6-v2` embeddings between generated answer and ground truth
- **Faithfulness (1-10):** Semantic alignment between answer and retrieved context (grounding)
- **Correctness (1-10):** Semantic alignment with reference answer × 10

### Evaluation Logic
- **Refusal Detection:** Response contains "I_DONT_KNOW" or phrases like "i cannot", "unable to", etc.
- **Ground Truth Answerability:** From `reference.json` → `Answerability: ["UNANSWERABLE"]`
- **Correct Refusal:** We refused AND GT says UNANSWERABLE
- **False Refusal:** We refused BUT GT says ANSWERABLE (bad!)
- **Answered Unanswerable:** We answered BUT GT says UNANSWERABLE (hallucination risk)

---

## Aggregate Results

| Metric | Value |
|--------|-------|
| **Total Samples** | 110 |
| **Total Refusals** | 77 (70.0%) |
| **Total Answered** | 33 (30.0%) |

### Refusal Breakdown

| Type | Count | Interpretation |
|------|-------|----------------|
| ✅ Correct Refusals | 6 | GT = UNANSWERABLE, we correctly refused |
| ❌ False Refusals | 71 | GT = ANSWERABLE, we wrongly refused |

### Answer Breakdown

| Type | Count | Interpretation |
|------|-------|----------------|
| ❌ Answered Unanswerable | 2 | GT = UNANSWERABLE but we answered (hallucination) |
| ✅ Good Answers | 1 | Faith ≥ 5 AND Corr ≥ 5 |
| ⚠️ Needs Check | 30 | Faith < 5 OR Corr < 5 |

### Quality Metrics (Answerable Questions Only, n=31)

| Metric | Value |
|--------|-------|
| Avg Semantic Similarity | 0.1536 |
| Avg Faithfulness (1-10) | 4.49 |
| Avg Correctness (1-10) | 1.54 |

---

## Fallback Reason Distribution

| Reason | Count | % of Refusals |
|--------|-------|---------------|
| `llm_refusal` | 42 | 54.5% |
| `hallucination_loop_exhausted` | 19 | 24.7% |
| `irrelevant_docs` | 16 | 20.8% |

---

## Per-Domain Summary

| Domain | Refusals | Correct | False | Answered | Good |
|--------|----------|---------|-------|----------|------|
| **GOVT** | 22 | 1 | 21 | 6 | 0 |
| **CLAPNQ** | 25 | 4 | 21 | 4 | 0 |
| **FIQA** | 14 | 1 | 13 | 13 | 0 |
| **CLOUD** | 16 | 0 | 16 | 10 | 1 |

---

## Critical Observations

### 🔴 Major Problems

1. **False Refusal Rate: 64.5%** (71/110)
   - Il sistema rifiuta di rispondere a domande che HANNO una risposta nel ground truth
   - La causa principale è `llm_refusal` (42 casi) → il modello si auto-censura troppo

2. **Extremely Low Correctness: 1.54/10**
   - Le risposte date non sono allineate semanticamente alle reference answer
   - Media semantic similarity = 0.15 (praticamente ortogonali)

3. **Solo 1 "Good Answer" su 110**
   - Solo cloud_3 supera entrambe le soglie (faith≥5, corr≥5)

### 🟡 Secondary Issues

4. **Hallucination Loop Exhausted: 19 casi**
   - Il Self-CRAG rileva hallucination e va in loop fino a esaurimento retry
   - Dopo 3 tentativi → fallback a I_DONT_KNOW

5. **Irrelevant Docs: 16 casi**
   - Il relevance grader giudica i documenti non pertinenti
   - Possibile problema di retrieval o soglia troppo alta

### 🟢 Positive Aspects

6. **Solo 2 Answered Unanswerable**
   - Il sistema è conservativo: quando risponde, tende ad avere contesto
   - Ma forse troppo conservativo

---

## Root Cause Analysis

### Problema 1: LLM Refusal (42 casi)
Il prompt di generazione contiene:
```
If the documents lack the information needed to answer the question, respond with "I_DONT_KNOW"
```
Con il prompt "conservative" applicato durante i fix, il modello è diventato **troppo prudente** e preferisce rifiutare.

### Problema 2: Hallucination Grader Troppo Stringente
Il grader modificato richiede che:
```
Every key fact in the answer MUST be explicitly stated in the documents
```
Questo è troppo restrittivo per risposte che richiedono ragionamento o sintesi.

### Problema 3: Domain Filter Mismatch
Durante i fix è stato corretto il filtro da `metadata.domain` a `metadata.source`, ma questo potrebbe aver peggiorato il recall se il campo `source` ha valori diversi da quelli attesi.

---

## Recommendations for Next Submission

1. **Rilassare il Prompt di Generazione**
   - Rimuovere "I_DONT_KNOW" come opzione esplicita
   - Usare "Answer based on the provided context"

2. **Abbassare Soglia Hallucination Grader**
   - Da "strict: every key fact MUST be explicitly stated"
   - A "the answer should be generally consistent with the documents"

3. **Verificare Domain Filter**
   - Controllare i valori effettivi di `metadata.source` nel Qdrant
   - Considerare di disabilitare il filtro domain (`use_filter=False`)

4. **Aumentare Max Retries**
   - Da 3 a 5 per dare più chance al Self-CRAG

5. **Debug sul Retrieval**
   - I 16 casi `irrelevant_docs` suggeriscono problemi di retrieval
   - Verificare che i documenti siano effettivamente indicizzati per i domini corretti

---

## Comparison: Pre-Fix vs Post-Fix

### Riepilogo Fix Applicati

| Fix | Descrizione | Impatto Atteso |
|-----|-------------|----------------|
| Domain Filter | `metadata.domain` → `metadata.source` | ✅ Retrieval corretto per dominio |
| Conservative Prompt | "MUST be directly supported" | ❌ Troppo restrittivo |
| Strict Hallucination Grader | "Every key fact MUST be explicit" | ❌ Troppo stringente |
| Dual Query Retrieval | Original + Standalone query | ⚠️ Non sufficiente |
| Fallback Cascade | Filter OFF se 0 risultati | ⚠️ Mitigazione parziale |

### Verdetto Finale

**I risultati post-fix sono PEGGIORI di quanto ci si aspettasse.**

La submission precedente (presumibilmente) aveva un tasso di refusal più basso, anche se potenzialmente con più hallucination. I fix applicati hanno:

1. **Corretto** il bug del domain filter → retrieval più accurato
2. **Peggiorato** la recall → il sistema rifiuta troppo spesso (70%)
3. **Peggiorato** la qualità delle risposte → solo 1/110 è "good"

### Causa Principale del Fallimento

Il problema NON era l'hallucination rate alto come suggerito dall'altra AI, ma piuttosto:

1. **Il modello Llama 3.1 8B quantizzato a 4-bit** non ha sufficiente capacità per generare risposte fedeli
2. **Il Self-CRAG crea un loop** dove il grader rileva "hallucination" anche su risposte corrette
3. **I prompt "conservative"** hanno reso il modello troppo prudente

### Affidabilità dei Risultati

| Aspetto | Pre-Fix (stimato) | Post-Fix | Differenza |
|---------|-------------------|----------|------------|
| Refusal Rate | ~30-40% | 70% | ⬆️ Peggio |
| False Refusals | ~20-30% | 64.5% | ⬆️ Molto peggio |
| Avg Correctness | ~2-3/10 | 1.54/10 | ⬇️ Peggio |
| Hallucination | Alto (?) | Basso (ma irrilevante perché rifiuta tutto) | - |

---

## Comparison: Strictness vs Model/Docs

### Due configurazioni confrontate

| Aspetto | Config precedente (meno strict) | Config attuale (più strict) | Effetto osservato |
|---------|--------------------------------|-----------------------------|-------------------|
| Prompt generazione | Meno restrittivo | "MUST be directly supported" | ⬆️ Refusal rate |
| Hallucination grader | Lax/moderato | "Every key fact MUST be explicit" | ⬆️ Loop + fallback |
| Risposte | Più frequenti | Molto rare | ⬇️ Coverage |
| Hallucination | Potenzialmente più alta | Più bassa | ✅ ma irrilevante |

### Evidenza che il problema è modello + documenti

1. **Modello sottodimensionato**
   - Llama 3.1 8B 4-bit mostra bassa accuratezza semantica anche quando risponde
   - Avg Correctness = **1.54/10** e Avg Semantic Similarity = **0.1536** → problema di capacità

2. **Documenti mancanti o non recuperati**
   - Refusals con `irrelevant_docs` + `hallucination_loop_exhausted` = **35/77**
   - Questo indica che spesso il retriever NON porta contesto utile

3. **Strictness non è la causa primaria**
   - Stringere i prompt riduce le hallucination ma **non migliora la correttezza**
   - L’accuratezza rimane bassa → limite del modello e/o dei documenti

### Conclusione della comparison

**La differenza di strictness cambia il tasso di refusal, non la qualità reale delle risposte.**
Il collo di bottiglia è la combinazione di:

- **modello troppo piccolo** per risposte multi-turn e grounding
- **documenti non sufficienti o retrieval insufficiente**

---

## Conclusione

**I risultati NON sono migliori né più affidabili.**

L'approccio conservativo ha fallito. Per migliorare serve:
1. Rimuovere o rilassare significativamente il Self-CRAG
2. Usare un modello più grande (13B+) o non quantizzato
3. Verificare che l'indice Qdrant contenga effettivamente i documenti corretti
4. Considerare un approccio più semplice: retrieval + generazione diretta senza CRAG

