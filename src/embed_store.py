"""Embed all chunks and load them into ChromaDB (Milestone 4, embedding step).

Run:  python src/embed_store.py
Prereq: data/chunks.json from `python src/ingest.py`.
"""

from __future__ import annotations

from collections import Counter

from vector_store import (
    CHROMA_DIR,
    COLLECTION_NAME,
    MODEL_NAME,
    ROOT,
    build_store,
    load_chunks,
)


def main() -> None:
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with '{MODEL_NAME}' ...")

    collection = build_store()

    print(f"stored {collection.count()} chunks in collection '{COLLECTION_NAME}'")
    print(f"persisted to: {CHROMA_DIR.relative_to(ROOT)}/ (cosine distance, 384-dim)")

    # Sanity check: every chunk carries its source so attribution works later.
    by_source = Counter(c["filename"] for c in chunks)
    print("\nchunks per source document:")
    for filename, count in sorted(by_source.items()):
        print(f"  {filename:<42} {count}")

    peek = collection.get(limit=1, include=["metadatas"])
    print(f"\nexample stored metadata: {peek['metadatas'][0]}")


if __name__ == "__main__":
    main()
