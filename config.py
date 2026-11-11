"""Central configuration for LayaRank Lite."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("LAYARANK_DATA_DIR", ROOT_DIR / "data"))
DOCUMENTS_DIR = DATA_DIR / "documents"
DB_PATH = DATA_DIR / "metadata.db"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
CACHE_DIR = DATA_DIR / "cache"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LAYA_MODEL = os.getenv("LAYA_MODEL", "convaiinnovations/laya")
DEVICE = os.getenv("DEVICE", "cpu")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "100"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "20"))
RRF_TOP_K = int(os.getenv("RRF_TOP_K", "16"))
LAYA_MAX_CANDIDATES = int(os.getenv("LAYA_MAX_CANDIDATES", "16"))
RRF_K = int(os.getenv("RRF_K", "60"))

for _path in (DATA_DIR, DOCUMENTS_DIR, CACHE_DIR):
    _path.mkdir(parents=True, exist_ok=True)
