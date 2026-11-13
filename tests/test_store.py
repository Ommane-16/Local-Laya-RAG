from jevrank.store import MetadataStore

def test_store_roundtrip_and_duplicate(tmp_path):
    store = MetadataStore(tmp_path / "test.db")
    item = {"chunk_id":"x","source":"a.pdf","page":1,"chunk_index":1,"text":"A useful passage."}
    assert store.add_chunk(item)
    assert not store.add_chunk(item)
    assert store.count_chunks() == 1
    assert store.get_all_chunks()[0]["chunk_id"] == "x"
