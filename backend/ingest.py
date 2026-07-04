"""
PDF Ingestion Pipeline:
  PDF → extract text (PyMuPDF) + OCR fallback (pytesseract) → clean → chunk → embed → ChromaDB
"""
import re
import time
import logging
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
import tiktoken
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    DATA_DIR, VECTORSTORE_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION, OCR_CHAR_THRESHOLD
)

# OCR is optional — requires Tesseract binary installed on the OS.
# It activates ONLY for pages with very little native text (scanned pages).
try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
    logger_ocr = logging.getLogger(__name__)
except ImportError:
    OCR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def _clean_text(text: str) -> str:
    """Remove excessive whitespace, normalize unicode."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u0900-\u097F]+', ' ', text)
    return text.strip()


def _strip_headers_footers(page: fitz.Page) -> str:
    """Extract text excluding top/bottom 7% of page (header/footer zone)."""
    rect = page.rect
    h = rect.height
    clip = fitz.Rect(rect.x0, rect.y0 + h * 0.07, rect.x1, rect.y1 - h * 0.07)
    return page.get_text("text", clip=clip)


def _ocr_page(page: fitz.Page) -> str:
    """
    OCR fallback: renders the page as a high-res image and runs Tesseract.
    WHERE: Called only when PyMuPDF extracts < OCR_CHAR_THRESHOLD characters
           (i.e., the page is scanned or image-based).
    WHY: Scanned PDFs have no embedded text layer. Without OCR these pages
         would be silently skipped, losing critical content.
    HOW: Page is rendered at 2x zoom (144 DPI) for better OCR accuracy,
         converted to PIL Image, then Tesseract extracts the text.
    """
    if not OCR_AVAILABLE:
        return ""
    mat = fitz.Matrix(2.0, 2.0)   # 2x zoom = 144 DPI for better OCR accuracy
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6")


def extract_pages(pdf_path: Path) -> list[dict]:
    """
    Extract text from all pages.
    For native-text pages: uses PyMuPDF (fast, ~milliseconds/page).
    For scanned/image pages: falls back to Tesseract OCR (slower, ~1-3s/page).
    Only pages below OCR_CHAR_THRESHOLD trigger the OCR fallback.
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = _strip_headers_footers(page)

        # Smart OCR fallback: only trigger for pages with almost no native text
        if len(text.strip()) < OCR_CHAR_THRESHOLD:
            ocr_text = _ocr_page(page)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text

        text = _clean_text(text)
        if text:
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "ocr_used": len(text.strip()) < OCR_CHAR_THRESHOLD,
            })
    doc.close()
    return pages


def chunk_pages(pages: list[dict], pdf_id: str, filename: str) -> list[dict]:
    """Sliding-window token chunking with metadata."""
    chunks = []
    chunk_idx = 0
    for page in pages:
        tokens = tokenizer.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens)
            if len(chunk_text.strip()) > 50:  # skip tiny chunks
                chunks.append({
                    "id": f"{pdf_id}_p{page['page_number']}_c{chunk_idx}",
                    "text": chunk_text.strip(),
                    "metadata": {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "page_number": page["page_number"],
                        "chunk_index": chunk_idx,
                        "token_count": len(chunk_tokens),
                    }
                })
                chunk_idx += 1
            if end == len(tokens):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_chroma_client():
    return chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(client):
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_pdfs(progress_callback=None) -> dict:
    """
    Main ingestion pipeline:
      scan DATA_DIR → extract → chunk → embed → upsert ChromaDB
    Returns summary stats.
    """
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        return {"status": "error", "message": f"No PDFs found in {DATA_DIR}"}

    logger.info(f"Found {len(pdf_files)} PDFs to ingest")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    # Track already-ingested PDFs by checking existing metadata
    existing_ids = set()
    try:
        existing = collection.get(include=["metadatas"])
        for meta in existing["metadatas"]:
            existing_ids.add(meta.get("pdf_id", ""))
    except Exception:
        pass

    total_chunks = 0
    ingested_pdfs = []

    for i, pdf_path in enumerate(pdf_files):
        pdf_id = pdf_path.stem.replace(" ", "_").lower()
        filename = pdf_path.name

        if pdf_id in existing_ids:
            logger.info(f"Skipping already-ingested: {filename}")
            if progress_callback:
                progress_callback(i + 1, len(pdf_files), filename, "skipped")
            continue

        logger.info(f"Processing [{i+1}/{len(pdf_files)}]: {filename}")
        if progress_callback:
            progress_callback(i + 1, len(pdf_files), filename, "processing")

        t0 = time.time()
        pages = extract_pages(pdf_path)
        chunks = chunk_pages(pages, pdf_id, filename)

        if not chunks:
            logger.warning(f"No extractable text in {filename}")
            continue

        # Batch embed
        texts = [c["text"] for c in chunks]
        embeddings = []
        for b_start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[b_start: b_start + EMBEDDING_BATCH_SIZE]
            embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            embeddings.extend(embs.tolist())

        # Upsert into ChromaDB
        collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        elapsed = time.time() - t0
        logger.info(f"  → {len(pages)} pages, {len(chunks)} chunks in {elapsed:.1f}s")
        total_chunks += len(chunks)
        ingested_pdfs.append({
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "time_s": round(elapsed, 2)
        })

        if progress_callback:
            progress_callback(i + 1, len(pdf_files), filename, "done")

    return {
        "status": "success",
        "ingested": ingested_pdfs,
        "total_chunks": total_chunks,
        "total_pdfs": len(ingested_pdfs),
        "vector_db_path": str(VECTORSTORE_DIR),
    }


if __name__ == "__main__":
    result = ingest_pdfs()
    print(result)
