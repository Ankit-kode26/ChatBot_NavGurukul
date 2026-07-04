"""
Retrieval pipeline:
  query → embed → ChromaDB ANN search → cross-encoder reranking → top-K chunks
"""
import time
import logging

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    VECTORSTORE_DIR, EMBEDDING_MODEL, RERANKER_MODEL,
    TOP_K_RETRIEVE, TOP_K_RERANK, CHROMA_COLLECTION
)
from ingest import get_chroma_client, get_or_create_collection

logger = logging.getLogger(__name__)

# Singletons — loaded once at startup
_embedder: SentenceTransformer = None
_reranker: CrossEncoder = None
_collection = None


def _load_models():
    global _embedder, _reranker, _collection
    if _embedder is None:
        logger.info("Loading embedding model...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    if _reranker is None:
        logger.info("Loading reranker model...")
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    if _collection is None:
        client = get_chroma_client()
        _collection = get_or_create_collection(client)


def retrieve(query: str) -> dict:
    """
    Full retrieval pipeline:
      1. Embed query
      2. ANN search in ChromaDB (top TOP_K_RETRIEVE)
      3. Cross-encoder rerank → top TOP_K_RERANK
    Returns dict with chunks, scores, latency breakdown.
    """
    _load_models()
    timings = {}

    # Step 1: Embed query
    t0 = time.time()
    query_emb = _embedder.encode([query], normalize_embeddings=True).tolist()
    timings["embed_ms"] = round((time.time() - t0) * 1000, 1)

    # Step 2: ANN retrieval from ChromaDB
    t0 = time.time()
    results = _collection.query(
        query_embeddings=query_emb,
        n_results=min(TOP_K_RETRIEVE, _collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

    if not results["documents"] or not results["documents"][0]:
        return {"chunks": [], "timings": timings}

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]  # cosine distances (lower = more similar)

    # Step 3: Cross-encoder reranking
    t0 = time.time()
    pairs = [[query, doc] for doc in docs]
    rerank_scores = _reranker.predict(pairs).tolist()
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    # Sort by reranker score (descending)
    ranked = sorted(
        zip(docs, metas, distances, rerank_scores),
        key=lambda x: x[3],
        reverse=True
    )[:TOP_K_RERANK]

    chunks = []
    for doc, meta, dist, rscore in ranked:
        chunks.append({
            "text": doc,
            "metadata": meta,
            "cosine_similarity": round(1 - dist, 4),
            "rerank_score": round(float(rscore), 4),
        })

    return {"chunks": chunks, "timings": timings}


def get_collection_stats() -> dict:
    """Return ChromaDB collection statistics."""
    _load_models()
    count = _collection.count()
    # Get unique PDFs from metadata
    if count == 0:
        return {"total_chunks": 0, "total_pdfs": 0, "pdf_list": []}
    sample = _collection.get(limit=min(count, 10000), include=["metadatas"])
    seen = {}
    for meta in sample["metadatas"]:
        fn = meta.get("filename", "unknown")
        if fn not in seen:
            seen[fn] = meta.get("pdf_id", "")
    return {
        "total_chunks": count,
        "total_pdfs": len(seen),
        "pdf_list": list(seen.keys()),
    }
