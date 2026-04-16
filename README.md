# RAG Pipeline — HDSS Demographic Data

A local, privacy-preserving Retrieval-Augmented Generation (RAG) system for querying Health and Demographic Surveillance System (HDSS) data using natural language.

All inference runs fully offline via [Ollama](https://ollama.com) — no cloud API calls, no data leaves your machine.

---

## How It Works

```
User Query
  ↓
Hybrid Retrieval
  ├── Semantic search  (ChromaDB + nomic-embed-text)
  └── Keyword search   (exact word matching on all records)
  ↓
Top ~10 most relevant records sent as context
  ↓
Local LLM (gemma3:4b via Ollama) generates answer
  ↓
Streamed response
```

The hybrid retrieval strategy ensures both conceptual queries ("who migrated for health reasons?") and name-specific lookups ("what is the date of birth for Miriam Kirunda?") return accurate results.

---

## Requirements

### Python packages
```bash
pip install chromadb ollama
```

### Ollama models
Install [Ollama](https://ollama.com/download), then pull the required models:
```bash
ollama pull nomic-embed-text
ollama pull gemma3:4b
```

---

## Usage

Run from the project root:
```bash
python main.py
```

On first run, all 50 records are embedded and stored in `chroma_db/` (takes ~30 seconds). Subsequent runs skip this step.

**Example queries:**
```
> what is the date of birth for miriam kirunda?
> what is the other name for patrick nansubuga?
> list all female records in the dataset
> who migrated from Kampala?
> which individuals have an exit type of DEATH?
```

Type `exit` or `quit` to stop.

---

## Project Structure

```
├── main.py                      # RAG pipeline
├── Data/
│   └── hdss_synthetic_50.jsonl  # 50 synthetic HDSS records
├── chroma_db/                   # Vector store (auto-generated, git-ignored)
└── .gitignore
```

---

## Data

The dataset contains 50 synthetic HDSS records with the following fields:

| Field | Description |
|-------|-------------|
| `id` | Unique individual ID (e.g. IND-0001) |
| `name` / `surname` | First and last name |
| `other_name1` | Middle / other name |
| `gender` | M / F |
| `dob` | Date of birth |
| `village_name` / `village_code` | Village of residence |
| `hh_relation` | Household relationship (HEAD, PARENT, SIBLING, etc.) |
| `exit_type` | LOSS_TO_FOLLOWUP, DEATH, or null (Active) |
| `entry_type` / `exit_type` | Migration event types |
| `province` / `move_place` | Migration origin/destination |

---

## Configuration

Key constants in `main.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL` | `nomic-embed-text:latest` | Ollama embedding model |
| `LLM_MODEL` | `gemma3:4b` | Ollama chat model |
| `N_SEMANTIC` | `5` | Documents retrieved via semantic search |
| `N_KEYWORD` | `5` | Documents added via keyword matching |
| `DATA_FILE` | `Data/hdss_synthetic_50.jsonl` | Input data path |

---

## Notes

- If `OLLAMA_HOST` is set to `0.0.0.0` (server bind address), the pipeline automatically remaps it to `127.0.0.1` for the Python client.
- The `chroma_db/` directory is git-ignored. Delete it to force a full re-embedding (e.g. after changing the document format).

---

## Potential Future Updates

- **Narrative-based embeddings** — `Data/Narative Data/hdss_synthetic_50_narratives.jsonl` contains full prose descriptions for each record (migration history, household context, observation events). Switching the pipeline to embed these narratives instead of the structured field format would improve semantic retrieval quality, especially for open-ended conceptual queries.
- **Larger dataset support** — extend ingestion to handle thousands of records with chunking and metadata filtering.
- **Web UI** — add a Gradio or Streamlit front-end for non-technical users.
- **Reranking** — add a cross-encoder reranker step between retrieval and generation for higher answer accuracy.
