"""
RAG Generation:
  retrieved chunks → structured prompt → Groq LLM → answer with citations
"""
import time
import logging
from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE

logger = logging.getLogger(__name__)

_client: Groq = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a precise research assistant answering questions from a private PDF knowledge base.

STRICT OUTPUT RULES:
1. Write in plain, clear English. Do NOT use asterisks (*), bold (**), or any markdown formatting.
2. Be concise and direct — answer exactly what was asked, nothing more.
3. Cite every factual claim inline like this: [Source: filename.pdf, p.12]
4. If the context does not contain the answer, say exactly: "The knowledge base does not contain information about this."
5. When listing items, use simple numbered lists (1. 2. 3.) — never bullet symbols.
6. Do not pad the answer with phrases like "Great question!" or "In conclusion".
7. Write as if explaining to an intelligent colleague — professional, factual, and to the point.
"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parts.append(
            f"[CHUNK {i}] Source: {meta['filename']}, Page {meta['page_number']}\n"
            f"{chunk['text']}\n"
        )
    return "\n---\n".join(parts)


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """
    Generate RAG answer conditioned on retrieved chunks.
    Returns answer text, sources list, and latency.
    """
    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the knowledge base.",
            "sources": [],
            "llm_ms": 0,
        }

    context = _build_context(chunks)
    user_message = f"""Context from knowledge base:
{context}

Question: {query}

Answer (with inline citations [Source: filename, p.N]):"""

    t0 = time.time()
    client = _get_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=GROQ_MAX_TOKENS,
        temperature=GROQ_TEMPERATURE,
    )
    llm_ms = round((time.time() - t0) * 1000, 1)

    answer = response.choices[0].message.content.strip()

    # Build deduplicated sources list
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["filename"], meta["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": meta["filename"],
                "page": meta["page_number"],
                "rerank_score": chunk.get("rerank_score", 0),
                "similarity": chunk.get("cosine_similarity", 0),
            })

    return {
        "answer": answer,
        "sources": sources,
        "llm_ms": llm_ms,
        "model": GROQ_MODEL,
        "tokens_used": response.usage.total_tokens if response.usage else None,
    }
