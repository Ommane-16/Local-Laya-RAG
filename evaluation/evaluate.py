"""Evaluate BM25, dense, hybrid RRF, and hybrid + Laya.
Fill relevant_chunk_ids in queries.json before interpreting quality metrics.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from config import DB_PATH, RRF_TOP_K, LAYA_MAX_CANDIDATES
from jevrank.store import MetadataStore

def precision_at_k(results, relevant, k):
    if k <= 0: return 0.0
    ids = [_id(x) for x in results[:k]]
    return sum(x in relevant for x in ids) / k

def recall_at_k(results, relevant, k):
    if not relevant or k <= 0: return 0.0
    ids = [_id(x) for x in results[:k]]
    return len(set(ids) & relevant) / len(relevant)

def reciprocal_rank(results, relevant):
    for rank, item in enumerate(results, 1):
        if _id(item) in relevant: return 1.0 / rank
    return 0.0

def reciprocal_rank_at_k(results, relevant, k):
    return reciprocal_rank(results[:k], relevant)

def ndcg_at_k(results, relevant, k):
    import math
    if k <= 0: return 0.0
    rels = [1 if _id(x) in relevant else 0 for x in results[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0

def _id(item):
    return item.get("chunk_id") or item.get("id")

def evaluate():
    from jevrank.search import SearchEngine
    from jevrank.rank import HybridRanker
    from jevrank.laya import LayaReranker

    path = Path(__file__).with_name("queries.json")
    dataset = json.loads(path.read_text(encoding="utf-8"))["queries"]
    store = MetadataStore(DB_PATH)
    search = SearchEngine(store)
    search.build_index()
    hybrid = HybridRanker(search)
    laya = LayaReranker()
    methods = ("bm25", "dense", "hybrid", "hybrid_laya")
    totals = {m: {metric: [] for metric in ("P@5","R@5","MRR","nDCG@5")} for m in methods}
    latencies = {m: [] for m in methods}
    judged = 0

    print("\nLAYARANK LITE — RETRIEVAL EVALUATION")
    print("=" * 64)
    for row in dataset:
        query = row["query"]
        relevant = set(row.get("relevant_chunk_ids") or [])
        start = time.perf_counter(); bm25 = search.bm25_search(query, RRF_TOP_K); latencies["bm25"].append(time.perf_counter()-start)
        start = time.perf_counter(); dense = search.dense_search(query, RRF_TOP_K); latencies["dense"].append(time.perf_counter()-start)
        start = time.perf_counter(); fused = hybrid.search(query, RRF_TOP_K); latencies["hybrid"].append(time.perf_counter()-start)
        start = time.perf_counter()
        try:
            reranked = laya.rerank(query, fused, top_k=RRF_TOP_K)
        except Exception as exc:
            print(f"Laya failed for {row['query_id']}: {exc}")
            reranked = fused
        latencies["hybrid_laya"].append(time.perf_counter()-start)

        print(f"\n{row['query_id']}: {query}")
        for name, results in (("BM25",bm25),("DENSE",dense),("HYBRID",fused),("HYBRID + LAYA",reranked)):
            print(f"\n{name}:")
            for rank, item in enumerate(results[:5], 1):
                print(f" {rank}. {item.get('source')} p.{item.get('page')} "
                      f"id={_id(item)} score={item.get('score',0):.5f}")
                print("    " + item.get("text","")[:220].replace("\n"," "))
        if not relevant:
            print("\nNo relevance labels for this query; quality metrics skipped.")
            continue
        judged += 1
        for name, results in (("bm25",bm25),("dense",dense),("hybrid",fused),("hybrid_laya",reranked)):
            totals[name]["P@5"].append(precision_at_k(results,relevant,5))
            totals[name]["R@5"].append(recall_at_k(results,relevant,5))
            totals[name]["MRR"].append(reciprocal_rank(results,relevant))
            totals[name]["nDCG@5"].append(ndcg_at_k(results,relevant,5))

    print("\n\nSUMMARY")
    print(f"Queries: {len(dataset)} | Judged: {judged}")
    for name in methods:
        print(f"\n{name.upper()}")
        if judged:
            for metric, vals in totals[name].items():
                print(f" {metric:8s}: {sum(vals)/len(vals):.4f}")
        else:
            print(" Metrics unavailable: add relevant_chunk_ids to queries.json.")
        print(f" Avg latency: {1000*sum(latencies[name])/max(1,len(latencies[name])):.2f} ms")
    print("\nLaya choice probabilities are not calibrated relevance scores.")

if __name__ == "__main__":
    evaluate()
