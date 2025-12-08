"""
Main entry point for MTRAGEval: SemEval 2026 Task 8.

This module provides a CLI interface to interact with the Self-CRAG
Multi-Turn RAG system using Llama 3.1 locally.

Usage:
    python main.py

The system will load the model (1-2 minutes on T4), then accept
user questions in a loop until 'exit' or 'quit' is entered.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage


def initialize_system():
    """
    Initialize the MTRAGEval system.
    
    Loads environment variables, imports the graph, and waits
    for model loading to complete.
    
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("initialize_system: Implement system initialization")


def process_user_input(user_input: str, chat_history: list) -> str:
    """
    Process a single user input through the RAG pipeline.
    
    Args:
        user_input: The user's question.
        chat_history: List of previous messages (HumanMessage/AIMessage).
        
    Returns:
        The AI-generated response.
        
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("process_user_input: Implement input processing through the graph")


def run_interactive_session():
    """
    Run an interactive CLI session.
    
    Maintains chat history and processes user inputs until
    'exit' or 'quit' is entered.
    
    Raises:
        NotImplementedError: Function not yet implemented.
    """
    raise NotImplementedError("run_interactive_session: Implement the interactive CLI loop")


def main():
    """
    Main entry point for the MTRAGEval system.
    
    Initializes the system and starts the interactive session.
    """
    print("=== SEMEVAL 2026 TASK 8: MTRAGEval SYSTEM (LLAMA 3.1 LOCAL) ===")
    print("Waiting for model and weights to load on GPU...\n")
    
    try:
        initialize_system()
        run_interactive_session()
    except NotImplementedError as e:
        print(f"[NOT IMPLEMENTED] {e}")
    except KeyboardInterrupt:
        print("\nSession terminated by user.")
    except Exception as e:
        print(f"Critical error: {e}")


if __name__ == "__main__":
    main()
