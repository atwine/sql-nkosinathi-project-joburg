# Changelog

All notable changes to this project are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] — 2026-04-16

### Added
- **Cross-encoder reranking stage** between retrieval and generation
  - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (via `sentence-transformers`)
  - Hybrid retrieval now casts a wider net (10 semantic + 10 keyword candidates)
  - Reranker rescores all candidates against the exact query and keeps the top 5
  - Reduces noise passed to the LLM, improving answer accuracy
- `N_FINAL` constant to control how many docs survive reranking
- Startup banner now shows Embedder, Reranker, and LLM in use

### Changed
- `N_SEMANTIC` increased from 5 → 10 (more candidates before reranking)
- `N_KEYWORD` increased from 5 → 10
- README updated: new 3-stage pipeline diagram, updated configuration table, reranking item marked complete in future updates

---

## [0.3.0] — 2026-04-16

### Added
- **Hybrid retrieval**: keyword search layer added alongside semantic search
  - Any query word (4+ characters) is matched directly against all loaded documents
  - Keyword hits are prepended to context so they appear first
  - Guarantees exact name matches are always retrieved regardless of embedding similarity
- `N_SEMANTIC` and `N_KEYWORD` constants for tunable retrieval

### Fixed
- Name-specific queries (e.g. "date of birth for Miriam Kirunda") were failing because `nomic-embed-text` does not rank exact name matches highly in semantic space — resolved by keyword search layer
- Small LLM (gemma3:4b) confused by 50-document context ("lost in the middle") — resolved by capping context to ~10 focused docs

### Changed
- Document format: `other_name1` is now an explicit `Other Name:` field instead of being merged into the Name line — enables accurate "other name" queries
- ChromaDB collection rebuilt to reflect updated document format

---

## [0.2.0] — 2026-04-16

### Added
- **Narrative dataset**: `Data/Narative Data/hdss_synthetic_50_narratives.jsonl`
  - Full prose descriptions for all 50 records covering migration history, household context, and observation events
  - Listed as a future upgrade path in README (narrative embeddings improve semantic retrieval quality)
- `README.md` with full project documentation:
  - Pipeline architecture diagram
  - Setup and usage instructions
  - Data field reference table
  - Configuration constants table
  - Potential future updates section

### Fixed
- `OLLAMA_HOST=0.0.0.0` environment variable caused Python ollama client connection failure
  - Root cause: `0.0.0.0` is a server bind address; the Python client needs a connectable address
  - Fix: env var is remapped to `127.0.0.1` before `import ollama` so the default client initialises correctly
- Streaming response chunks are now accessed via `chunk["message"]["content"]` (compatible with ollama SDK 0.5.x)
- Switched from `ollama.<function>()` global calls to explicit `ollama.Client(host=...)` instances to guarantee correct host resolution

---

## [0.1.0] — 2026-04-16

### Added
- Initial RAG pipeline (`main.py`)
  - Loads 50 synthetic HDSS records from `Data/hdss_synthetic_50.jsonl`
  - Embeds documents using `nomic-embed-text:latest` via Ollama
  - Stores embeddings in a persistent ChromaDB vector store (`chroma_db/`)
  - Retrieves top-5 semantically similar docs per query
  - Generates streamed answers via `gemma3:4b` (Ollama)
  - Interactive CLI loop with `exit`/`quit` support
- `OllamaEmbeddingFunction` class wrapping Ollama embeddings API for ChromaDB
- Batch ingestion (50 docs/batch) with skip-if-already-embedded logic
- `.gitignore` excluding `chroma_db/`, `__pycache__`, `.env`
- Synthetic HDSS dataset: 50 records with demographics, migration, and household fields
