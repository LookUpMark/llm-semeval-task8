#!/usr/bin/env python3
"""
SemEval 2026 Task 8 - Task C: RAG (End-to-End)
Python script version of TaskC_RAG.ipynb
"""

import os
import sys
import json
from tqdm import tqdm

# --- SETUP PATHS ---
if os.path.exists("src"):
    PROJECT_ROOT = os.getcwd()
else:
    PROJECT_ROOT = os.path.abspath("..")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph import initialize_graph

# --- CONFIGURATION ---
TEAM_NAME = "Gbgers"
DOMAINS = ["govt", "clapnq", "fiqa", "cloud"]

# TEST MODE: Set to True for quick verification
TEST_MODE = True
TEST_QUERY_LIMIT = 5  # Queries per domain in test mode

CONVERSATIONS_FILE = os.path.join(PROJECT_ROOT, "dataset/human/conversations/conversations.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/submissions")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"submission_TaskC_{TEAM_NAME}.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- HELPER FUNCTIONS ---

def extract_last_user_question(messages):
    """Extract the last user message from conversation."""
    for msg in reversed(messages):
        if msg.get("speaker") == "user":
            return msg.get("text", "")
    return ""


def format_input_for_output(messages):
    """Format messages for submission output."""
    return [{"speaker": m["speaker"], "text": m["text"]} for m in messages]


def format_contexts(docs, domain):
    """Format retrieved documents for output."""
    contexts = []
    for i, doc in enumerate(docs):
        if hasattr(doc, 'metadata'):
            meta = doc.metadata
            contexts.append({
                "document_id": str(meta.get("doc_id") or meta.get("parent_id") or f"{domain}_{i}"),
                "score": float(meta.get("relevance_score", 0.0)),
                "text": meta.get("parent_text") or getattr(doc, 'page_content', str(doc))
            })
        else:
            contexts.append({
                "document_id": f"{domain}_{i}",
                "score": 0.0,
                "text": str(doc)
            })
    return contexts


# --- MAIN ---

def main():
    print(f"Processing domains: {DOMAINS}")
    if TEST_MODE:
        print(f"⚠️ TEST MODE: Processing only {TEST_QUERY_LIMIT} queries per domain.")

    # Initialize graph
    print("🔧 Initializing RAG graph...")
    app = initialize_graph()
    print("✅ Graph ready.")

    # Load conversations
    print("📂 Loading conversations...")
    with open(CONVERSATIONS_FILE, 'r') as f:
        all_conversations = json.load(f)
    print(f"Total conversations: {len(all_conversations)}")

    all_results = []

    for domain in DOMAINS:
        print(f"\n{'='*40}\n🌍 DOMAIN: {domain.upper()}\n{'='*40}")
        
        # Filter by domain
        domain_convs = [c for c in all_conversations if domain.lower() in c.get("domain", "").lower()]
        print(f"Found {len(domain_convs)} conversations")
        
        if not domain_convs:
            continue
        
        if TEST_MODE:
            print(f"✂️ TEST MODE: Processing {TEST_QUERY_LIMIT} queries")
            domain_convs = domain_convs[:TEST_QUERY_LIMIT]
        
        print(f"🚀 Running RAG pipeline...")
        for conv in tqdm(domain_convs):
            messages = conv.get("messages", [])
            question = extract_last_user_question(messages)
            
            if not question:
                continue
            
            try:
                response = app.invoke({"question": question})
                raw_docs = response.get("documents", [])
                gen_text = response.get("generation", "No Answer")
            except Exception as e:
                print(f"Error: {e}")
                raw_docs = []
                gen_text = "Error"
            
            all_results.append({
                "conversation_id": conv.get("author"),
                "Collection": f"mt-rag-{domain}",
                "input": format_input_for_output(messages),
                "contexts": format_contexts(raw_docs, domain),
                "predictions": [{"text": gen_text}]
            })

    # Save results
    print(f"\n💾 Saving {len(all_results)} results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("✅ Done!")


if __name__ == "__main__":
    main()
