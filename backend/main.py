"""
FastAPI backend — RAG Chatbot
Endpoints:
  POST /api/ingest      — run ingestion pipeline
  POST /api/query       — RAG query
  GET  /api/status      — vector DB stats
  GET  /api/metrics     — evaluation metrics (p95, R@k, MRR, citation accuracy)
  GET  /api/health      — health check
  GET  /pdfs/<filename> — serve PDF file for in-browser viewing
"""
import time
import logging
import numpy as np
from pathlib import Path
from threading import Thread
from collections import deque

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent))

from ingest import ingest_pdfs, DATA_DIR
from retriever import retrieve, get_collection_stats
from generator import generate_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Chatbot", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global ingestion state
ingestion_state = {
    "running": False,
    "progress": [],
    "result": None,
    "error": None,
}

# Rolling window of last 100 query results for metrics
query_log = deque(maxlen=100)


# ─── Request/Response Models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list
    chunks: list      # display chunks (truncated text)
    full_chunks: list # full text for viewer
    timings: dict
    total_ms: float


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    stats = get_collection_stats()
    return {
        "vector_db": stats,
        "ingestion": {
            "running": ingestion_state["running"],
            "progress": ingestion_state["progress"][-5:],
            "last_result": ingestion_state["result"],
        },
        "data_dir": str(DATA_DIR),
        "pdf_files_on_disk": [f.name for f in DATA_DIR.glob("*.pdf")],
    }


@app.get("/api/metrics")
def metrics():
    """
    Overall evaluation metrics across ALL queries in the session (rolling last 100).
    NOT per-query — these are aggregate statistics.
    """
    if not query_log:
        return {"status": "no_data", "message": "No queries recorded yet."}

    logs = list(query_log)

    # ── Latency ────────────────────────────────────────────
    latencies = [q["total_ms"] for q in logs]
    p95 = float(np.percentile(latencies, 95))
    p50 = float(np.percentile(latencies, 50))

    # ── R@5 ──────────────────────────────────────────────
    # Fraction of queries where top chunk cosine similarity >= 0.5
    COSINE_THRESHOLD = 0.5
    recall_hits = sum(
        1 for q in logs
        if q["chunks"] and q["chunks"][0]["cosine_similarity"] >= COSINE_THRESHOLD
    )
    recall_at_k = round(recall_hits / len(logs), 3)

    # ── MRR ──────────────────────────────────────────────
    # Cross-encoder scores can all be positive; use session median as dynamic threshold
    all_rerank_scores = [c["rerank_score"] for q in logs for c in q["chunks"]]
    median_rerank = float(np.median(all_rerank_scores)) if all_rerank_scores else 0.0

    rr_scores = []
    for q in logs:
        rr = 0.0
        for rank, chunk in enumerate(q["chunks"], 1):
            if chunk["rerank_score"] >= median_rerank:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    mrr = round(float(np.mean(rr_scores)) if rr_scores else 0.0, 3)

    # ── Average retrieval quality ──────────────────────────────
    avg_similarity = round(float(np.mean([
        c["cosine_similarity"] for q in logs for c in q["chunks"]
    ])), 3) if all_rerank_scores else 0.0
    avg_rerank = round(float(np.mean(all_rerank_scores)), 3) if all_rerank_scores else 0.0

    # ── Citation accuracy ────────────────────────────────────
    citation_hits = sum(1 for q in logs if "[Source:" in q["answer"])
    citation_accuracy = round(citation_hits / len(logs), 3)

    # ── Hallucination proxy ─────────────────────────────────
    hallucination_count = sum(
        1 for q in logs
        if "does not contain information" in q["answer"].lower()
        or "knowledge base does not" in q["answer"].lower()
    )
    hallucination_rate = round(hallucination_count / len(logs), 3)

    # ── Per-query history (last 5 for UI) ───────────────────────
    recent = []
    for q in list(reversed(logs))[:5]:
        top_chunk = q["chunks"][0] if q["chunks"] else {}
        recent.append({
            "query": q["query"][:80] + ("…" if len(q["query"]) > 80 else ""),
            "total_ms": q["total_ms"],
            "top_similarity": round(top_chunk.get("cosine_similarity", 0), 3),
            "top_rerank": round(top_chunk.get("rerank_score", 0), 3),
            "cited": "[Source:" in q["answer"],
            "no_answer": "does not contain information" in q["answer"].lower(),
        })

    return {
        "total_queries": len(logs),
        "scope": "overall_session",
        "latency": {
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "avg_ms": round(float(np.mean(latencies)), 1),
            "min_ms": round(float(np.min(latencies)), 1),
            "max_ms": round(float(np.max(latencies)), 1),
        },
        "retrieval": {
            "R_at_5": recall_at_k,
            "MRR": mrr,
            "avg_cosine_similarity": avg_similarity,
            "avg_rerank_score": avg_rerank,
            "k_retrieve": 20,
            "k_rerank": 5,
        },
        "generation": {
            "citation_accuracy": citation_accuracy,
            "hallucination_rate_proxy": hallucination_rate,
        },
        "recent_queries": recent,
    }


@app.post("/api/ingest")
def trigger_ingest():
    """Trigger background ingestion of all PDFs in data/pdfs/."""
    if ingestion_state["running"]:
        return {"status": "already_running", "message": "Ingestion is already in progress."}

    def run():
        ingestion_state["running"] = True
        ingestion_state["progress"] = []
        ingestion_state["error"] = None

        def callback(current, total, filename, state):
            ingestion_state["progress"].append({
                "current": current, "total": total,
                "filename": filename, "state": state
            })

        try:
            result = ingest_pdfs(progress_callback=callback)
            ingestion_state["result"] = result
        except Exception as e:
            ingestion_state["error"] = str(e)
            logger.error(f"Ingestion error: {e}")
        finally:
            ingestion_state["running"] = False

    Thread(target=run, daemon=True).start()
    return {"status": "started", "message": "Ingestion started in background. Poll /api/status."}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file to the data directory."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    dest = DATA_DIR / file.filename
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"status": "uploaded", "filename": file.filename, "size_bytes": len(content)}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Full RAG pipeline: embed → retrieve → rerank → generate."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    total_start = time.time()

    # Step 1 & 2: Embed + Retrieve (HNSW ANN, top-20) + Rerank (top-5)
    retrieval_result = retrieve(req.query)
    chunks = retrieval_result["chunks"]
    timings = retrieval_result["timings"]

    if not chunks:
        return QueryResponse(
            answer="The knowledge base does not contain information about this. Please make sure PDFs have been ingested.",
            sources=[], chunks=[], full_chunks=[],
            timings=timings,
            total_ms=round((time.time() - total_start) * 1000, 1),
        )

    # Step 3: Generate answer with Groq LLM
    gen_result = generate_answer(req.query, chunks)
    timings["llm_ms"] = gen_result["llm_ms"]
    timings["total_ms"] = round((time.time() - total_start) * 1000, 1)

    # Build display chunks (truncated) and full chunks separately
    display_chunks = []
    full_chunks = []
    for c in chunks:
        display_chunks.append({
            "text": c["text"][:400] + ("…" if len(c["text"]) > 400 else ""),
            "metadata": c["metadata"],
            "cosine_similarity": c["cosine_similarity"],
            "rerank_score": c["rerank_score"],
        })
        full_chunks.append({
            "text": c["text"],
            "metadata": c["metadata"],
            "cosine_similarity": c["cosine_similarity"],
            "rerank_score": c["rerank_score"],
        })

    # Log this query for metrics
    query_log.append({
        "query": req.query,
        "answer": gen_result["answer"],
        "chunks": full_chunks,
        "total_ms": timings["total_ms"],
    })

    return QueryResponse(
        answer=gen_result["answer"],
        sources=gen_result["sources"],
        chunks=display_chunks,
        full_chunks=full_chunks,
        timings=timings,
        total_ms=timings["total_ms"],
    )


# ─── Serve PDFs ────────────────────────────────────────────────────────────────

@app.get("/pdfs/{filename}")
def serve_pdf(filename: str):
    """Serve a PDF file for in-browser viewing."""
    pdf_path = DATA_DIR / filename
    if not pdf_path.exists() or not pdf_path.suffix == ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(str(pdf_path), media_type="application/pdf")


# ─── Serve Frontend ───────────────────────────────────────────────────────────

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    def serve_ui():
        return FileResponse(str(frontend_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
