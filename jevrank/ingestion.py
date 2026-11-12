"""PDF/text ingestion with conservative chunk cleanup."""
from __future__ import annotations
import hashlib
import re
from pathlib import Path
from typing import Iterable

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS

def normalize_text(text: str) -> str:
    import re

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def is_useful_chunk(text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    clean = normalize_text(text)
    if len(clean) < min_chars:
        return False
    words = re.findall(r"\b[\w'-]+\b", clean)
    if len(words) < 15:
        return False
    # Filter separator-only and mostly numeric/page-number fragments.
    alpha = sum(ch.isalpha() for ch in clean)
    if alpha / max(len(clean), 1) < 0.45:
        return False
    if re.fullmatch(r"[\W\d_]+", clean):
        return False
    return True

def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        piece = normalize_text(text[start:end])
        if is_useful_chunk(piece, min_chars=min(40, MIN_CHUNK_CHARS)):
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    # Remove exact duplicates while preserving order.
    return list(dict.fromkeys(chunks))

def _pdf_pages(path: Path):
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    reader = PdfReader(str(path))
    for page_no, page in enumerate(reader.pages, start=1):
        yield page_no, normalize_text(page.extract_text() or "")

def load_document(path: str | Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    records = []
    if suffix == ".pdf":
        for page_no, text in _pdf_pages(path):
            for chunk_no, chunk in enumerate(split_text(text), start=1):
                records.append({
                    "text": chunk, "page": page_no, "chunk_index": chunk_no,
                    "source": path.name,
                })
    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk_no, chunk in enumerate(split_text(text), start=1):
            records.append({
                "text": chunk, "page": None, "chunk_index": chunk_no,
                "source": path.name,
            })
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, TXT, or MD.")
    return records

def stable_chunk_id(source: str, page: int | None, text: str) -> str:
    raw = f"{source}|{page}|{normalize_text(text)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def ingest_file(path: str | Path, store) -> int:
    records = load_document(path)
    inserted = 0
    for record in records:
        record["chunk_id"] = stable_chunk_id(
            record["source"], record.get("page"), record["text"]
        )
        inserted += bool(store.add_chunk(record))
    return inserted
