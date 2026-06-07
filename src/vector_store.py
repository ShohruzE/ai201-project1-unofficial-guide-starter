"""Embedding + vector store for The Unofficial Guide (Milestone 4).

Reusable core shared by:
    - embed_store.py : builds the ChromaDB collection from data/chunks.json
    - retrieve.py    : queries the collection for the top-k relevant chunks

Embedding model and store follow planning.md -> Retrieval Approach:
    - sentence-transformers/all-MiniLM-L6-v2 (local, no API key, 384-dim)
    - ChromaDB persistent store, cosine distance, metadata for source attribution
    - default top_k = 4

We embed both the chunks and the query with the *same* model, and we use
cosine distance so scores live in [0, 2] (0 = identical). With this setup,
strong matches land well under 0.5, which is the Milestone 4 checkpoint target.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

# --- Config ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = ROOT / "data" / "chunks.json"
CHROMA_DIR = ROOT / "chroma_db"           # gitignored local persistence
COLLECTION_NAME = "unofficial_guide"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4                          # planning.md -> Retrieval Approach

# Metadata keys we attach to every chunk (used later for source attribution).
METADATA_KEYS = ("source", "filename", "title", "section_heading", "chunk_index")


class RetrievedChunk(TypedDict):
    rank: int
    text: str
    distance: float
    source: str
    filename: str
    title: str
    section_heading: str
    chunk_index: int


# --- Model & client (loaded once, then cached) -----------------------------
@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load all-MiniLM-L6-v2 once per process."""
    return SentenceTransformer(MODEL_NAME)


@functools.lru_cache(maxsize=1)
def get_client() -> chromadb.api.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with L2-normalized vectors so cosine distance is well scaled."""
    vectors = get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# --- Build the store -------------------------------------------------------
def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH.relative_to(ROOT)} not found -- run `python src/ingest.py` first."
        )
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))


def build_store() -> chromadb.api.models.Collection.Collection:
    """Embed every chunk and (re)load it into a fresh ChromaDB collection.

    The collection is dropped and recreated each run so re-ingesting never
    leaves stale or duplicate chunks behind.
    """
    chunks = load_chunks()
    client = get_client()

    # Start clean: drop any existing collection of the same name.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance, not the L2 default
    )

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{k: c[k] for k in METADATA_KEYS} for c in chunks]
    embeddings = embed(documents)

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    return collection


def get_collection() -> chromadb.api.models.Collection.Collection:
    """Return the existing collection, building it on demand if missing/empty."""
    client = get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            return collection
    except Exception:
        pass
    return build_store()


# --- Retrieval -------------------------------------------------------------
def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    """Return the top-k chunks most relevant to `query`, with source metadata
    and cosine distance (lower = more similar)."""
    collection = get_collection()
    q_emb = embed([query])
    res = collection.query(
        query_embeddings=q_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    results: list[RetrievedChunk] = []
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        results.append(
            RetrievedChunk(
                rank=rank,
                text=doc,
                distance=float(dist),
                source=meta["source"],
                filename=meta["filename"],
                title=meta["title"],
                section_heading=meta["section_heading"],
                chunk_index=int(meta["chunk_index"]),
            )
        )
    return results
