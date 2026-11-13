from jevrank.ingestion import normalize_text, split_text, is_useful_chunk
from jevrank.rank import HybridRanker

def test_normalize_and_split():
    assert normalize_text(" a   b \n\n\n c ") == "a b\n\n c"
    parts = split_text("This is a meaningful sentence. " * 50, chunk_size=300, overlap=30)
    assert parts

def test_filter_empty_and_short():
    assert not is_useful_chunk("---")
    assert not is_useful_chunk("123 456 789")
    assert is_useful_chunk("This is a sufficiently long sentence with meaningful alphabetic words. " * 3)

def test_rrf_combines_and_deduplicates():
    a = [{"chunk_id":"a","text":"A"},{"chunk_id":"b","text":"B"}]
    b = [{"chunk_id":"b","text":"B"},{"chunk_id":"c","text":"C"}]
    result = HybridRanker.reciprocal_rank_fusion(a,b)
    assert [x["chunk_id"] for x in result] == ["b","a","c"]
    assert result[0]["rrf_score"] > result[1]["rrf_score"]
