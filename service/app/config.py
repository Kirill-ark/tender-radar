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
# The SDK retries with exponential backoff on 429 / 5xx. The free tier allows
# 8000 tokens per minute and one card takes ~3000, so bursts do hit the limit.
LLM_MAX_RETRIES = 5

# --- Card ---
# The card is built from the most relevant chunks, not the whole PDF: it keeps
# each call small for the free tier's tokens-per-minute limit.
CARD_CHUNKS_PER_QUERY = 4
CARD_MAX_CHUNKS = 15

# --- Fit score ---
# The LLM only classifies the tender against this profile; the score itself is
# computed in code from those classes, so the same tender always gets the same score
# and every point can be explained.
# The two lists must not overlap, or a borderline tender flips between them.
CORE_TRADES = (
    "interior renovation and finishing: interior doors, drywall, flooring, "
    "screed, painting, plastering, tiling"
)
ADJACENT_TRADES = (
    "building envelope and structure: windows, facades, roofing, carpentry and "
    "timber construction, metal works, shell construction"
)
PROJECT_SIZE = "100k-5M EUR"
TRADE_POINTS = {"core": 7, "adjacent": 4, "none": 0}
SIZE_POINTS = {"fits": 3, "unknown": 2, "too_large": 0, "too_small": 0}

# --- Q&A ---
# German notices, questions often in English: cross-language matches score lower, so
# the right chunk can rank below 5th (eval: "works start" ranked 8th). 8 covers it.
ASK_TOP_K = 8
# Relevance floor (cosine similarity of the best chunk). Below it the answer is
# "not found" without an LLM call. It only filters off-topic questions (pizza: 0.09);
# on-topic questions with no answer score higher than real answers (0.42 vs 0.30
# in eval), so those are refused by the grounded prompt, the second layer.
ASK_MIN_SIMILARITY = 0.2
MAX_QUESTION_CHARS = 500

# --- Metrics and token budget ---
# Every request and its LLM token usage is logged here; the README numbers are
# computed from this file, and the daily budget is checked against it.
METRICS_DB = os.getenv("METRICS_DB", "/data/metrics/metrics.sqlite")
# Guardrail: once this many tokens were spent today (UTC), LLM calls are refused,
# so a bug or a flood of tenders cannot burn the free tier. Kept below Groq's own
# free-tier limit (200k tokens/day for gpt-oss-20b): at 200k Groq refused first.
DAILY_TOKEN_BUDGET = int(os.getenv("DAILY_TOKEN_BUDGET", "150000"))
