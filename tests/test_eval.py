from evaluation.evaluate import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

def test_metrics():
    results = [{"chunk_id":"a"},{"chunk_id":"b"},{"chunk_id":"c"}]
    relevant = {"b","c"}
    assert precision_at_k(results,relevant,2) == 0.5
    assert recall_at_k(results,relevant,2) == 0.5
    assert reciprocal_rank(results,relevant) == 0.5
    assert 0 < ndcg_at_k(results,relevant,3) <= 1
