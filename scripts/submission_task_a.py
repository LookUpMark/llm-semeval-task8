#!/usr/bin/env python3
"""
SemEval 2026 Task 8 - Task A: Retrieval
Uses a UNIFIED collection for all domains.
"""

import os
import sys
import json
import zipfile
from tqdm import tqdm
from pathlib import Path

# --- SETUP PATHS ---
if os.path.exists("src"):
    PROJECT_ROOT = os.getcwd()
else:
    PROJECT_ROOT = os.path.abspath("..")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ingestion import load_and_chunk_data, build_vector_store
from src.retrieval import get_retriever, get_qdrant_client

# --- CONFIGURATION ---
TEAM_NAME = "Gbgers"
DOMAINS = ["govt", "clapnq", "fiqa", "cloud"]
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

# UNIFIED COLLECTION NAME
COLLECTION_NAME = "mtrag_unified"

# TEST MODE: Set to True for quick verification
TEST_MODE = True
TEST_SUBSET_SIZE = 1000   # Number of chunks to index per domain (if building new)
TEST_QUERY_LIMIT = 10     # Number of queries to process per domain

CORPUS_BASE_DIR = os.path.join(PROJECT_ROOT, "dataset/corpora/passage_level")
CONVERSATIONS_FILE = os.path.join(PROJECT_ROOT, "dataset/human/conversations/conversations.json")
QDRANT_PATH = os.path.join(PROJECT_ROOT, "qdrant_db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/submissions")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"submission_TaskA_{TEAM_NAME}.jsonl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- HELPER FUNCTIONS ---

def extract_last_query(messages):
    """Extract last user question from messages."""
    for msg in reversed(messages):
        if msg.get("speaker") == "user":
            return msg.get("text", "")
    return ""

def get_corpus_file(domain):
    """Get or extract corpus file path."""
    jsonl_path = os.path.join(CORPUS_BASE_DIR, f"{domain}.jsonl")
    zip_path = os.path.join(CORPUS_BASE_DIR, f"{domain}.jsonl.zip")
    
    if not os.path.exists(jsonl_path):
        if os.path.exists(zip_path):
            print(f"📦 Extracting {domain}.jsonl...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(CORPUS_BASE_DIR)
        else:
            return None
    return jsonl_path

def build_unified_collection():
    """Build a single unified Qdrant collection with all domains."""
    
    # Check if collection already exists
    if os.path.exists(QDRANT_PATH):
        try:
            client = get_qdrant_client(QDRANT_PATH)
            if client.collection_exists(COLLECTION_NAME):
                info = client.get_collection(COLLECTION_NAME)
                print(f"✅ Unified collection '{COLLECTION_NAME}' exists ({info.points_count} vectors)")
                return True
        except Exception as e:
            print(f"⚠️ Warning checking collection: {e}")
    
    # Build unified collection
    print(f"🔄 Building unified collection '{COLLECTION_NAME}' with all domains...")
    all_docs = []
    
    for domain in DOMAINS:
        corpus_path = get_corpus_file(domain)
        if not corpus_path:
            print(f"⚠️ Corpus not found for {domain}, skipping...")
            continue
        
        print(f"📂 Loading {domain}...")
        docs = load_and_chunk_data(corpus_path)
        
        # Add domain metadata to each document
        for doc in docs:
            doc.metadata["domain"] = domain
        
        if TEST_MODE and len(docs) > TEST_SUBSET_SIZE:
            print(f"✂️ TEST MODE: Slicing {domain} to first {TEST_SUBSET_SIZE} chunks (from {len(docs)})")
            docs = docs[:TEST_SUBSET_SIZE]
        
        all_docs.extend(docs)
        print(f"   Added {len(docs)} chunks from {domain}")
    
    print(f"📊 Total documents to index: {len(all_docs)}")
    build_vector_store(all_docs, persist_dir=QDRANT_PATH, collection_name=COLLECTION_NAME)
    print("✅ Unified collection built and saved")
    return True

# --- MAIN EXECUTION ---

def main():
    print(f"Processing domains: {DOMAINS}")
    if TEST_MODE:
        print(f"⚠️ TEST MODE ACTIVE: Indexing only {TEST_SUBSET_SIZE} chunks per domain, processing {TEST_QUERY_LIMIT} queries per domain.")

    # 1. Build unified collection (once for all domains)
    if not build_unified_collection():
        print("❌ Failed to build unified collection. Exiting.")
        return
    
    # 2. Initialize single retriever
    print("🔍 Initializing unified retriever...")
    retriever = get_retriever(
        qdrant_path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        top_k_retrieve=TOP_K_RETRIEVE,
        top_k_rerank=TOP_K_RERANK
    )
    print("✅ Retriever ready")

    # 3. Load ALL conversations
    print("📂 Loading conversations...")
    with open(CONVERSATIONS_FILE, 'r') as f:
        all_conversations = json.load(f)
    print(f"Total conversations: {len(all_conversations)}")

    all_results = []

    for domain in DOMAINS:
        print(f"\n{'='*40}\n🌍 PROCESSING DOMAIN: {domain.upper()}\n{'='*40}")
        
        # Filter Conversations (Substring matching)
        domain_convs = [
            c for c in all_conversations 
            if domain.lower() in c.get("domain", "").lower()
        ]
        print(f"Found {len(domain_convs)} conversations for {domain}")
        
        if not domain_convs:
            continue
            
        if TEST_MODE:
            print(f"✂️ TEST MODE: Processing only first {TEST_QUERY_LIMIT} conversations")
            domain_convs = domain_convs[:TEST_QUERY_LIMIT]
        
        # Run Retrieval (using same unified retriever)
        print(f"🚀 Running retrieval for {domain}...")
        for conv in tqdm(domain_convs):
            messages = conv.get("messages", [])
            query = extract_last_query(messages)
            if not query: 
                continue
                
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"Error: {e}\")
                docs = []
                
            # Format output
            contexts = []
            for i, doc in enumerate(docs):
                meta = doc.metadata
                contexts.append({
                    "document_id": str(meta.get("doc_id") or meta.get("parent_id") or f"{domain}_{i}"),
                    "score": float(meta.get("relevance_score") or 0.0),
                    "text": meta.get("parent_text") or doc.page_content
                })
                
            all_results.append({
                "conversation_id": conv.get("author"),
                "task_id": f"{conv.get('author')}::1",
                "Collection": f"mt-rag-{domain}",
                "input": [{"speaker": m["speaker"], "text": m["text"]} for m in messages],
                "contexts": contexts
            })

    # --- SAVE RESULTS ---
    print(f"\n💾 Saving {len(all_results)} total results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("✅ Done!")

if __name__ == "__main__":
    main()
