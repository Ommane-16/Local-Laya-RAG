"""Streamlit interface for local document retrieval and Laya reranking."""
from pathlib import Path
import streamlit as st

from config import DOCUMENTS_DIR, RRF_TOP_K
from jevrank.store import MetadataStore
from jevrank.ingestion import ingest_file
from jevrank.search import SearchEngine
from jevrank.rank import HybridRanker
from jevrank.laya import LayaReranker

st.set_page_config(page_title="LayaRank Lite", page_icon="🔎", layout="wide")
st.title("LayaRank Lite")
st.caption("Local document retrieval with BM25 + dense search + RRF + Laya reranking.")

@st.cache_resource
def resources():
    store = MetadataStore()
    search = SearchEngine(store)
    return store, search, HybridRanker(search), LayaReranker()

store, search, hybrid, laya = resources()

with st.sidebar:
    st.subheader("Add documents")
    uploads = st.file_uploader(
        "Upload PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if st.button("Ingest uploaded documents", disabled=not uploads):
        count = 0
        for uploaded in uploads:
            target = DOCUMENTS_DIR / Path(uploaded.name).name
            target.write_bytes(uploaded.getbuffer())
            count += ingest_file(target, store)
        search._indexed = False
        st.success(f"Added {count} new chunks. Existing data was preserved.")
    st.metric("Stored chunks", store.count_chunks())
    st.caption("Laya inference runs locally on CPU by default.")

query = st.text_input("Search your documents", placeholder="e.g. What is natural language processing?")
top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
if st.button("Search", type="primary", disabled=not query.strip()):
    with st.spinner("Retrieving candidates..."):
        candidates = hybrid.search(query, top_k=RRF_TOP_K)
    if not candidates:
        st.warning("No useful passages found. Try a more descriptive query or ingest a document.")
    else:
        with st.spinner("Laya is reranking the candidates..."):
            try:
                reranked = laya.rerank_with_details(query, candidates, top_k=top_k)
            except Exception as exc:
                st.error(f"Laya reranking failed: {exc}")
                reranked = {"results": candidates[:top_k], "status": "fallback"}
        if reranked["status"] == "fallback":
            st.info("Showing hybrid retrieval results because Laya was unavailable.")
        elif reranked["status"] == "single_candidate":
            st.info("Only one candidate was available; no model comparison was needed.")
        elif reranked["status"] == "no_candidates":
            st.warning("No non-empty candidates were available.")
        for rank, item in enumerate(reranked["results"], start=1):
            with st.container(border=True):
                st.markdown(f"**#{rank} — {item.get('source', 'Unknown source')}**")
                if item.get("page") is not None:
                    st.caption(f"Page {item['page']}")
                if item.get("laya_selected"):
                    st.caption("Laya's selected passage")
                if item.get("laya_probability") is not None:
                    st.caption(f"Laya choice probability: {item['laya_probability']:.4f} (not calibrated relevance)")
                st.write(item.get("text", ""))
                st.caption(f"RRF score: {item.get('rrf_score', item.get('score', 0)):.5f}")
