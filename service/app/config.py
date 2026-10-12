"""Central configuration: model, chunking, storage paths and safety limits.

Everything tunable lives here so the rest of the code never hard-codes a value.
"""

import os

# --- Storage ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "/data/chroma")  # Docker volume
COLLECTION_NAME = "tenders"

# --- Embeddings ---
# TED notices for Germany are in German, so an English-only model (all-MiniLM-L6-v2)
# would embed them poorly. The multilingual L12 model covers 50+ languages.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking (chars) ---
# This model reads at most 128 tokens per input and silently truncates the rest.
# German averages ~3-4 chars per token, so 400 chars keeps a whole chunk inside
# the window (800, as in the RAG project, would lose half of every chunk).
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- Guardrails: document limits ---
# Tender PDFs are untrusted input from the internet: cap what we download and parse
# so a huge or broken file cannot hang the service or blow up memory.
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_PDF_PAGES = 50
DOWNLOAD_TIMEOUT_SECONDS = 30

# --- LLM ---
# Key comes from the environment only (.env locally, never committed).
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# Groq retired its Llama chat models; gpt-oss-20b is fast and supports JSON mode.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TIMEOUT_SECONDS = 30
LLM_MAX_RETRIES = 2  # the SDK retries with exponential backoff on 429 / 5xx

# --- Card ---
# The card is built from the most relevant chunks, not the whole PDF: it keeps
# each call small for the free tier's tokens-per-minute limit.
CARD_CHUNKS_PER_QUERY = 4
CARD_MAX_CHUNKS = 15
COMPANY_PROFILE = (
    "General building contractor based in Germany, about 50 employees. "
    "Does building construction, renovation and interior finishing "
    "(doors, windows, drywall, flooring). Typical project size 100k-5M EUR."
)
