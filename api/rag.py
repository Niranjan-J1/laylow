import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Embedding model ready.")
    return _embedder

DOCS_DIR = Path(__file__).parent.parent / "docs_store"
DOCS_DIR.mkdir(exist_ok=True)

_store: dict = {}

def _persist_doc(doc_id: str, filename: str, chunks: list[str]):
    """Save chunks to disk so they survive restarts."""
    meta = {"filename": filename, "chunks": chunks}
    (DOCS_DIR / f"{doc_id}.json").write_text(json.dumps(meta))

def _load_persisted():
    """Load all saved documents on startup."""
    for f in DOCS_DIR.glob("*.json"):
        doc_id = f.stem
        if doc_id in _store:
            continue
        try:
            meta = json.loads(f.read_text())
            chunks = meta["chunks"]
            embedder = get_embedder()
            embeddings = embedder.encode(chunks, show_progress_bar=False)
            _store[doc_id] = {
                "filename":   meta["filename"],
                "chunks":     chunks,
                "embeddings": embeddings,
            }
            print(f"Loaded persisted doc: {meta['filename']}")
        except Exception as e:
            print(f"Failed to load {f}: {e}")

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 30]

def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        import fitz
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    elif ext in [".txt", ".md"]:
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def add_document(file_bytes: bytes, filename: str) -> dict:
    doc_id = hashlib.md5(file_bytes).hexdigest()[:12]

    if doc_id in _store:
        return {"doc_id": doc_id, "chunks": len(_store[doc_id]["chunks"]), "cached": True}

    text      = extract_text(file_bytes, filename)
    chunks    = chunk_text(text)
    embedder  = get_embedder()
    embeddings = embedder.encode(chunks, show_progress_bar=False)

    _store[doc_id] = {
        "filename":   filename,
        "chunks":     chunks,
        "embeddings": embeddings,
    }

    _persist_doc(doc_id, filename, chunks)

    return {"doc_id": doc_id, "chunks": len(chunks), "cached": False}

def retrieve(query: str, doc_id: Optional[str] = None, top_k: int = 3) -> list[str]:
    # Load any persisted docs we don't have in memory
    _load_persisted()

    if not _store:
        return []

    embedder    = get_embedder()
    query_embed = embedder.encode([query])[0]

    all_chunks = []
    all_embeds = []

    docs = [_store[doc_id]] if doc_id and doc_id in _store else list(_store.values())
    for doc in docs:
        all_chunks.extend(doc["chunks"])
        all_embeds.append(doc["embeddings"])

    if not all_chunks:
        return []

    all_embeds  = np.vstack(all_embeds)
    query_norm  = query_embed / (np.linalg.norm(query_embed) + 1e-9)
    embeds_norm = all_embeds / (np.linalg.norm(all_embeds, axis=1, keepdims=True) + 1e-9)
    scores      = embeds_norm @ query_norm

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [all_chunks[i] for i in top_indices if scores[i] > 0.2]

def list_documents() -> list[dict]:
    _load_persisted()
    return [
        {"doc_id": did, "filename": d["filename"], "chunks": len(d["chunks"])}
        for did, d in _store.items()
    ]