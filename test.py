import sys
from src.graph import build_graph

def run_test():
    # 1. Inizializzazione del Grafo
    print("--- INIZIALIZZAZIONE SISTEMA MTRAGEval ---")
    app = build_graph()
    
    # 2. Definizione dell'Input Iniziale
    # Simuliamo una domanda che richiede contesto (Multi-turn)
    inputs = {
        "question": "E per quanto riguarda i limiti di VRAM?",
        "messages": [], # Inizialmente vuoto, add_messages lo gestirà
        "retry_count": 0
    }

    # 3. Esecuzione in Streaming
    # .stream ci permette di vedere l'attivazione di ogni singolo nodo [cite: 271]
    print("\n--- ESECUZIONE FLUSSO GRAFO ---")
    try:
        for output in app.stream(inputs):
            # Ogni 'output' è un dizionario con il nome del nodo eseguito
            for node_name, state_update in output.items():
                print(f"\n>>> Nodo Completato: {node_name}")
                # Visualizziamo i cambiamenti chiave nello stato
                if "standalone_question" in state_update:
                    print(f"    Standalone Query: {state_update['standalone_question']}")
                if "generation" in state_update:
                    print(f"    Risposta: {state_update['generation']}")
                if "retry_count" in state_update:
                    print(f"    Retry Count: {state_update['retry_count']}")
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")

if __name__ == "__main__":
    run_test()