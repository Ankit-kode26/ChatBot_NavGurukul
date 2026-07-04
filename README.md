# ChatBot_NavGurukul
HACKATHON Challenge 1

A production-grade Retrieval-Augmented Generation (RAG) chatbot over large PDF corpora.  
**Fully open-source stack** · **2–5s latency** · **Source citations**

## Stack

| Layer | Tool |
|---|---|
| PDF Extraction | PyMuPDF (native) + pytesseract (OCR fallback) |
| Chunking | Sliding window — 512 tokens, 100 token overlap |
| Embeddings | `all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB — HNSW cosine index (local, free) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, free) |
| LLM | Groq `llama-3.1-8b-instant` (free tier) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/CSS/JS |

---

## Setup

### 1. Prerequisites

```bash
# Python 3.10+
python --version

# (Optional) Tesseract for OCR of scanned PDFs
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS:   brew install tesseract
# Linux:   sudo apt install tesseract-ocr
```

### 2. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` in the project root and add your Groq API key:

```
GROQ_API_KEY=gsk_...
```

### 4. Add PDFs

Drop your PDF files into `data/pdfs/`.

### 5. Run the server

```bash
cd backend
python main.py
```

Open **http://localhost:8000** in your browser.

---

## Usage

1. **Upload PDFs** — drag & drop into the sidebar or browse.
2. **Ingest** — click "Start Ingestion" to run the full pipeline (extract → chunk → embed → index).
3. **Ask questions** — type in the chat box. Each answer includes:
   - 📄 Source citations (PDF filename + page number)
   - ⚡ Latency breakdown (embed / retrieval / rerank / LLM)
   - 📚 Retrieved chunks visualization (collapsible panel)

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/status` | Vector DB stats + ingestion state |
| `POST` | `/api/ingest` | Trigger ingestion pipeline (background) |
| `POST` | `/api/upload` | Upload a PDF file |
| `POST` | `/api/query` | RAG query → answer + sources + chunks |

### Example query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main findings?"}'
```

---

## Architecture

```
User Query
    │
    ▼
Embed query (all-MiniLM-L6-v2, ~20ms)
    │
    ▼
ChromaDB HNSW ANN Search → top-20 chunks (~10ms)
    │
    ▼
Cross-encoder Reranker → top-5 chunks (~80ms)
    │
    ▼
Groq LLM (llama-3.1-8b-instant) → Answer + Citations (~500ms)
    │
    ▼
Response with sources [PDF filename + page number]
```

**Total latency: ~1.5–3s** (hardware dependent)

---

## Tuning for Latency

| Parameter | File | Default | Notes |
|---|---|---|---|
| `TOP_K_RETRIEVE` | `config.py` | 20 | Reduce to 10 for faster reranking |
| `TOP_K_RERANK` | `config.py` | 5 | Final context window size |
| `CHUNK_SIZE` | `config.py` | 512 tokens | Larger = fewer chunks |
| `GROQ_MAX_TOKENS` | `config.py` | 1024 | Reduce for faster LLM response |
| `EMBEDDING_BATCH_SIZE` | `config.py` | 64 | Increase if RAM allows |
