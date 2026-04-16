import json
import os

# Fix OLLAMA_HOST before importing ollama — the client captures the env var at import time.
# If OLLAMA_HOST is set to 0.0.0.0 (a server bind address), replace with 127.0.0.1.
_raw_host = os.getenv("OLLAMA_HOST", "")
if _raw_host.startswith("0.0.0.0"):
    os.environ["OLLAMA_HOST"] = _raw_host.replace("0.0.0.0", "127.0.0.1")

import chromadb
import ollama

DATA_FILE = "Data/hdss_synthetic_50.jsonl"
EMBED_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "gemma3:4b"
N_SEMANTIC = 5   # top-k from semantic search
N_KEYWORD  = 5   # max docs added by keyword matching


class OllamaEmbeddingFunction(chromadb.utils.embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str):
        self.model_name = model_name
        _host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if not _host.startswith("http"):
            _host = f"http://{_host}"
        self._client = ollama.Client(host=_host)

    def __call__(self, input: chromadb.api.types.Documents) -> chromadb.api.types.Embeddings:
        embeddings = []
        for text in input:
            res = self._client.embeddings(model=self.model_name, prompt=text)
            embeddings.append(res["embedding"])
        return embeddings


def load_data(file_path):
    documents = []
    metadata = []
    ids = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            other_name = record.get("other_name1") or record.get("other_name2") or "N/A"
            doc_str = (
                f"Record ID: {record.get('id')}\n"
                f"First Name: {record.get('name')}\n"
                f"Other Name: {other_name}\n"
                f"Surname: {record.get('surname')}\n"
                f"Gender: {record.get('gender')}\n"
                f"Village: {record.get('village_name')} (Code: {record.get('village_code')})\n"
                f"Date of Birth: {record.get('dob')}\n"
                f"Household Relation: {record.get('hh_relation')}\n"
                f"Status/Exit Type: {record.get('exit_type') or 'Active'}"
            )

            documents.append(doc_str)
            ids.append(record.get("id"))
            metadata.append({
                "id": record.get("id"),
                "name": f"{record.get('name')} {record.get('surname')}"
            })

    return documents, metadata, ids


def keyword_search(query: str, all_docs: list[str], all_ids: list[str], max_results: int) -> list[tuple[str, str]]:
    """Return (id, doc) pairs whose text contains any word from the query (case-insensitive, min 4 chars)."""
    words = [w.lower() for w in query.split() if len(w) >= 4]
    hits = []
    for doc_id, doc in zip(all_ids, all_docs):
        doc_lower = doc.lower()
        if any(w in doc_lower for w in words):
            hits.append((doc_id, doc))
        if len(hits) >= max_results:
            break
    return hits


def get_ollama_client():
    _host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not _host.startswith("http"):
        _host = f"http://{_host}"
    return ollama.Client(host=_host)


def main():
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found! Run this script from the workspace root.")
        return

    documents, metadata, ids = load_data(DATA_FILE)
    print(f"Loaded {len(documents)} records.")

    print("Initializing ChromaDB and encoding documents...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    embed_fn = OllamaEmbeddingFunction(model_name=EMBED_MODEL)

    collection = chroma_client.get_or_create_collection(
        name="hdss_population",
        embedding_function=embed_fn
    )

    if collection.count() == 0 and documents:
        print("Embedding data (this may take a moment)...")
        batch_size = 50
        for i in range(0, len(documents), batch_size):
            collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadata[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print("Vector database populated.")
    else:
        print(f"Database already contains {collection.count()} records. Skipping re-embedding.")

    print("\n--- Command Line RAG Ready ---")
    print(f"Using Embedder: {EMBED_MODEL}")
    print(f"Using LLM:      {LLM_MODEL}")
    print("Type 'exit' or 'quit' to stop.\n")

    ollama_client = get_ollama_client()

    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if query.strip().lower() in ["exit", "quit", "q"]:
            break
        if not query.strip():
            continue

        # --- Hybrid retrieval ---
        print("[Retrieving...]", end="", flush=True)

        # 1. Semantic search
        sem_results = collection.query(query_texts=[query], n_results=N_SEMANTIC)
        sem_docs = sem_results["documents"][0] if sem_results["documents"] else []
        sem_ids  = sem_results["ids"][0]       if sem_results["ids"]       else []

        # 2. Keyword search over all loaded documents
        kw_hits = keyword_search(query, documents, ids, max_results=N_KEYWORD)

        # 3. Merge: keyword hits first (most likely exact matches), then semantic, deduped
        seen = set(sem_ids)
        merged_docs = list(sem_docs)
        for doc_id, doc in kw_hits:
            if doc_id not in seen:
                merged_docs.insert(0, doc)   # prepend so they appear early in context
                seen.add(doc_id)

        print(f" Done. ({len(merged_docs)} docs: {len(sem_docs)} semantic + {len(merged_docs)-len(sem_docs)} keyword)\n")

        context = "\n\n".join(merged_docs)

        # --- Build prompt ---
        system_prompt = (
            "You are a helpful assistant analyzing demographic health survey data. "
            "Use ONLY the records provided in the context below to answer the question. "
            "Quote the Record ID when referring to a specific person. "
            "If multiple records match, list all of them. "
            "If the answer cannot be found in the context, say so.\n\n"
            f"Context:\n{context}"
        )

        # --- Generate ---
        print("[Generating answer...]")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": query}
        ]

        try:
            response = ollama_client.chat(model=LLM_MODEL, messages=messages, stream=True)
            for chunk in response:
                print(chunk["message"]["content"], end="", flush=True)
            print()
        except Exception as e:
            print(f"\nError communicating with Ollama: {e}")


if __name__ == "__main__":
    main()
