"""BM25 and dense retrieval."""
from __future__ import annotations
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from config import EMBEDDING_MODEL, BM25_TOP_K, DENSE_TOP_K
from jevrank.store import MetadataStore
from jevrank.ingestion import is_useful_chunk

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

class SearchEngine:
    def __init__(self, store: MetadataStore, model_name: str = EMBEDDING_MODEL):
        self.store = store
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.bm25 = None
        self._indexed = False

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device="cpu")
        return self.model

    def build_index(self):
        self.chunks = [
            c for c in self.store.get_all_chunks()
            if is_useful_chunk(c.get("text", ""), min_chars=40)
        ]
        self._indexed = True
        if not self.chunks:
            self.index = self.bm25 = None
            return
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        vectors = self._load_model().encode(
            [c["text"] for c in self.chunks],
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _ensure_index(self):
        if not self._indexed:
            self.build_index()

    def bm25_search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        self._ensure_index()
        if self.bm25 is None or not tokenize(query):
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:max(0, top_k)]
        results = []
        for idx in order:
            if float(scores[idx]) <= 0:
                continue
            item = dict(self.chunks[int(idx)])
            item.update(score=float(scores[idx]), method="bm25")
            results.append(item)
        return results

    def dense_search(self, query: str, top_k: int = DENSE_TOP_K) -> list[dict]:
        self._ensure_index()
        if self.index is None or not query.strip() or top_k <= 0:
            return []
        vector = self._load_model().encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(
            np.asarray(vector, dtype=np.float32),
            min(int(top_k), len(self.chunks))
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.chunks[int(idx)])
            item.update(score=float(score), method="dense")
            results.append(item)
        return results
