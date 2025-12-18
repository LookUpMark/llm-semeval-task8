from src.generation import create_generation_components
from tqdm import tqdm

if __name__ == "__main__":
    progress_bar = tqdm(
        total=100,
        desc="Generation Test",
        leave=True
    )

    print("--- GENERATION TESTS STARTED ---")
    components = create_generation_components()
    progress_bar.update(10)

    print("--- TEST GENERATOR (POSITIVE) ---")
    res = components.generator.invoke(
        {
            "context": "Tim Cook is the CEO of Apple.",
            "question": "Who runs Apple?"
        }
    )
    print(f"Output: {res}\nExpected: Somethink like 'Tim Cook'")
    progress_bar.update(50)

    print("--- TEST GENERATOR (NEGATIVE) ---")
    res = components.generator.invoke(
        {
            "context": "The sky is blue.",
            "question": "Who is the CEO of Apple?"
        }
    )
    print(f"Output: {res}\nExpected: 'I_DONT_KNOW'")
    progress_bar.update(90)

    print("--- TEST REWRITER---")
    res = components.query_rewriter.invoke(
        {
            "messages": [
                ("human", "Who is the creator of Python?")
            ],
            "question": "When was he born?"
        }
    )
    print(f"Output: {res}\nExpected: 'When was Guido van Rossum born?'")
    progress_bar.update(100)

    print("--- GENERATION TESTS COMPLETED ---")

        