import sys
import os
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation import create_generation_components

# CONFIGURATION
# MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Standard
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct" # Light for local testing

if __name__ == "__main__":
    progress_bar = tqdm(
        total=100,
        desc="Generation Test",
        leave=True
    )

    print("--- GENERATION TESTS STARTED ---")
    components = create_generation_components(model_id=MODEL_ID)
    progress_bar.update(10)

    print("--- TEST GENERATOR (POSITIVE) ---")
    res = components.generator.invoke(
        {
            "context": "Tim Cook is the CEO of Apple.",
            "question": "Who is the CEO of Apple?"
        }
    )
    print(f"Output: {res}\nExpected: Somethink like 'Tim Cook'\n\n")
    progress_bar.update(50)

    print("--- TEST GENERATOR (NEGATIVE) ---")
    res = components.generator.invoke(
        {
            "context": "The sky is blue.",
            "question": "Who is the CEO of Apple?"
        }
    )
    print(f"Output: {res}\nExpected: 'I_DONT_KNOW'\n\n")
    progress_bar.update(90)

    print("--- TEST REWRITER---")
    res = components.query_rewriter.invoke(
        {
            "messages": [
                ("human", "Who is the creator of Python?"),
                ('assistant', 'Guido van Rossum')
            ],
            "question": "When was he born?"
        }
    )
    print(f"Output: {res}\nExpected: 'When was Guido van Rossum born?'\n\n")
    progress_bar.update(100)

    print("--- GENERATION TESTS COMPLETED ---")

        