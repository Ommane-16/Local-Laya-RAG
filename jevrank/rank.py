"""Reciprocal Rank Fusion."""
from __future__ import annotations
from config import RRF_K, RRF_TOP_K

class HybridRanker:
    def __init__(self, search_engine, rrf_k: int = RRF_K):
        self.search_engine = search_engine
        self.rrf_k = max(1, int(rrf_k))

    @staticmethod
    def reciprocal_rank_fusion(*ranked_lists, k: int = 60):
        scores, records = {}, {}
        for ranked in ranked_lists:
            seen = set()
            for rank, item in enumerate(ranked, start=1):
                key = item.get("chunk_id") or item.get("id")
                if key is None:
                    key = (item.get("source"), item.get("page"), item.get("text"))
                if key in seen:
                    continue
                seen.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                records.setdefault(key, dict(item))
        ordered = sorted(scores, key=scores.get, reverse=True)
        output = []
        for key in ordered:
            item = records[key]
            item["score"] = float(scores[key])
            item["rrf_score"] = float(scores[key])
            item["method"] = "hybrid"
            output.append(item)
        return output

    def search(self, query: str, top_k: int = RRF_TOP_K,
               bm25_top_k: int | None = None,
               dense_top_k: int | None = None) -> list[dict]:
        if not query or not query.strip() or top_k <= 0:
            return []
        bm25 = self.search_engine.bm25_search(
            query, top_k=bm25_top_k or max(top_k, 20)
        )
        dense = self.search_engine.dense_search(
            query, top_k=dense_top_k or max(top_k, 20)
        )
        fused = self.reciprocal_rank_fusion(bm25, dense, k=self.rrf_k)
        return fused[:top_k]
