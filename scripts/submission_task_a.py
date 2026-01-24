
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
from src.retrieval import get_retriever
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
TEAM_NAME = "Gbgers"
DOMAINS = ["govt", "clapnq", "fiqa", "cloud"]  # All domains
TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5

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

def extract_last_query(turns):
    """Extract last user question from turns."""
    for turn in reversed(turns):
        if turn.get("role") == "user":
            return turn.get("content", "")
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

def ensure_vector_store(domain):
    """Ensure Qdrant collection exists for domain."""
    collection_name = f"mtrag_{domain}"
    corpus_path = get_corpus_file(domain)
    
    if not corpus_path:
        print(f"⚠️ Corpus not found for {domain}")
        return None
    
    need_build = True
    if os.path.exists(QDRANT_PATH):
        client = None
        try:
            client = QdrantClient(path=QDRANT_PATH)
            if client.collection_exists(collection_name):
                info = client.get_collection(collection_name)
                print(f"✅ Collection '{collection_name}' exists ({info.points_count} vectors)")
                need_build = False
        except Exception as e:
            print(f"⚠️ Warning checking collection: {e}")
        finally:
            if client:
                client.close()  # Ensure lock is released
    
    if need_build:
        print(f"🔄 Building collection '{collection_name}' for {domain}...")
        docs = load_and_chunk_data(corpus_path)
        
        if TEST_MODE and len(docs) > TEST_SUBSET_SIZE:
            print(f"✂️ TEST MODE: Slicing to first {TEST_SUBSET_SIZE} chunks (from {len(docs)}) ")
            docs = docs[:TEST_SUBSET_SIZE]
            
        build_vector_store(docs, persist_dir=QDRANT_PATH, collection_name=collection_name)
        print("✅ Built and saved")
    
    return collection_name

# --- MAIN EXECUTION ---

def main():
    print(f"Processing domains: {DOMAINS}")
    if TEST_MODE:
        print(f"⚠️ TEST MODE ACTIVE: Indexing only {TEST_SUBSET_SIZE} chunks, processing {TEST_QUERY_LIMIT} queries.")

    # Load ALL conversations once
    print("📂 Loading conversations...")
    with open(CONVERSATIONS_FILE, 'r') as f:
        all_conversations = json.load(f)
    print(f"Total conversations: {len(all_conversations)}")

    all_results = []

    for domain in DOMAINS:
        print(f"\n{'='*40}\n🌍 PROCESSING DOMAIN: {domain.upper()}\n{'='*40}")
        
        # 1. Setup Vector Store
        try:
            collection_name = ensure_vector_store(domain)
            if not collection_name:
                continue
        except Exception as e:
            print(f"❌ Critical error setting up vector store for {domain}: {e}")
            continue
            
        # 2. Filter Conversations (Substring matching)
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
        
        # 3. Init Retriever
        retriever = get_retriever(
            qdrant_path=QDRANT_PATH,
            collection_name=collection_name,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_rerank=TOP_K_RERANK
        )
        
        # 4. Run Retrieval
        print(f"🚀 Running retrieval for {domain}...")
        for conv in tqdm(domain_convs):
            query = extract_last_query(conv.get("turns", []))
            if not query: 
                continue
                
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                print(f"Error: {e}")
                docs = []
                
            # Format
            contexts = []
            for i, doc in enumerate(docs):
                meta = doc.metadata
                contexts.append({
                    "document_id": str(meta.get("doc_id") or meta.get("parent_id") or f"{domain}_{i}"),
                    "score": float(meta.get("relevance_score") or 0.0),
                    "text": meta.get("parent_text") or doc.page_content
                })
                
            all_results.append({
                "conversation_id": conv.get("conversation_id"),
                "task_id": f"{conv.get('conversation_id')}::1",
                "Collection": f"mt-rag-{domain}",
                "input": [{"speaker": t["role"], "text": t["content"]} for t in conv["turns"]],
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
