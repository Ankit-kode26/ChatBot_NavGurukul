import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model (free, local, fast)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# Chunking
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 100    # ~20% overlap

# Retrieval
TOP_K_RETRIEVE = 20    # candidates from vector DB
TOP_K_RERANK = 5       # final chunks after reranking

# Reranker model (free, local cross-encoder)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ChromaDB
CHROMA_COLLECTION = "rag_corpus"

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.1

# OCR: min chars on a page before triggering OCR fallback
OCR_CHAR_THRESHOLD = 100
