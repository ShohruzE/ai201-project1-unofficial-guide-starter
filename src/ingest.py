"""Ingestion and chunking pipeline for The Unofficial Guide (Milestone 3).

Pipeline (see planning.md -> Architecture):
    1. Load every .txt guide from documents/ (recursively).
    2. Save the raw text to a consistent format (data/raw_loaded.json) before cleaning.
    3. Clean each document (normalize whitespace, flatten links, strip stray HTML).
    4. Heading-aware recursive chunking: keep short sections whole; split long
       sections by paragraph (then sentence) into 300-450 word chunks with overlap.
    5. Save chunks with metadata (source, title, chunk_index) to data/chunks.json.

Run:  python src/ingest.py
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import TypedDict

# --- Paths -----------------------------------------------------------------
# Project root is the parent of the src/ folder this file lives in.
ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = ROOT / "documents"
DATA_DIR = ROOT / "data"
RAW_LOADED_PATH = DATA_DIR / "raw_loaded.json"
CHUNKS_PATH = DATA_DIR / "chunks.json"


class Document(TypedDict):
    source: str          # path relative to project root, e.g. documents/cs-club/career_paths.txt
    filename: str        # career_paths.txt
    folder: str          # cs-club
    title: str           # first non-empty line of the file
    raw_text: str        # exact text as read from disk


# --- Stage 1: load ---------------------------------------------------------
def load_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Document]:
    """Read every non-empty .txt file under documents_dir (recursively).

    The first non-empty line of each file is treated as the document title
    (these guides put their title on line 1, e.g. "Career Paths").
    """
    docs: list[Document] = []
    for path in sorted(documents_dir.rglob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        if not raw_text.strip():
            print(f"  ! skipping empty file: {path}")
            continue

        title = next((line.strip() for line in raw_text.splitlines() if line.strip()), path.stem)
        docs.append(
            Document(
                source=path.relative_to(ROOT).as_posix(),
                filename=path.name,
                folder=path.parent.name,
                title=title,
                raw_text=raw_text,
            )
        )
    return docs


def save_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --- Stage 2: clean --------------------------------------------------------
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_INLINE_WS = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize a raw guide into clean text while preserving paragraph and
    section structure (the chunker relies on blank lines and heading lines).

    - Unescape HTML entities and strip any stray HTML tags (defensive; these
      hand-written guides have none, but a future scraped source might).
    - Flatten Markdown links ``[label](url)`` to ``label (url)`` so both the
      resource name and its link survive for retrieval and citation.
    - Normalize newlines and whitespace; collapse runs of blank lines to one.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = _HTML_TAG.sub("", text)
    text = _MD_LINK.sub(r"\1 (\2)", text)
    # Normalize non-breaking / zero-width spaces.
    text = text.replace("\xa0", " ").replace("​", "")
    # Collapse runs of spaces/tabs within each line and trim line ends.
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


# --- Stage 3: heading-aware recursive chunking -----------------------------
MAX_WORDS = 450          # hard cap; sections longer than this get split
TARGET_WORDS = 375       # aim point when packing a split section (~middle of 300-450)
OVERLAP_WORDS = 60       # carried between sub-chunks of the same long section
MERGE_BELOW_WORDS = 35   # sections smaller than this are merged into a neighbor

_LIST_ITEM = re.compile(r"^(\d+[.)]|[-*•])\s")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _is_heading(stripped: str, prev_blank: bool, next_line: str) -> bool:
    """Plain-text heading heuristic (see planning.md -> Chunking Strategy)."""
    if not stripped or not prev_blank:
        return False
    if not next_line.strip():           # a heading is followed immediately by body
        return False
    if len(stripped.split()) > 8:
        return False
    if stripped[-1] in ".,;:":
        return False
    if _LIST_ITEM.match(stripped):
        return False
    return True


def split_into_sections(text: str, title: str) -> list[dict[str, str]]:
    """Split a cleaned document into {heading, body} sections.

    The first non-empty line is the document title and heads the intro section;
    later heading lines start new sections.
    """
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if ln.strip()), 0)

    sections: list[dict[str, str]] = []
    cur_heading, cur_body = title, []
    prev_blank = True
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if _is_heading(stripped, prev_blank, next_line):
            sections.append({"heading": cur_heading, "body": "\n".join(cur_body).strip()})
            cur_heading, cur_body = stripped, []
        else:
            cur_body.append(lines[i])
        prev_blank = stripped == ""
    sections.append({"heading": cur_heading, "body": "\n".join(cur_body).strip()})

    return [s for s in sections if s["heading"].strip() or s["body"].strip()]


def _section_text(section: dict[str, str]) -> str:
    """Render a section as 'heading' over 'body' so the heading stays with its text."""
    heading, body = section["heading"].strip(), section["body"].strip()
    return f"{heading}\n{body}".strip() if body else heading


def _pack_units(units: list[str], heading: str) -> list[str]:
    """Greedily pack text units (paragraphs or sentences) into chunks of about
    TARGET_WORDS (never exceeding MAX_WORDS), repeating `heading` on each chunk
    and carrying ~OVERLAP_WORDS of trailing context between consecutive chunks.
    """
    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush() -> list[str]:
        """Emit the buffer as a chunk and return the overlap tail (as words)."""
        body = "\n\n".join(buf).strip()
        chunks.append(f"{heading}\n{body}".strip())
        tail = body.split()[-OVERLAP_WORDS:]
        return tail

    for unit in units:
        uw = _word_count(unit)
        if buf and buf_words + uw > MAX_WORDS:
            tail = flush()
            buf, buf_words = ([" ".join(tail)] if tail else []), len(tail)
        buf.append(unit)
        buf_words += uw
        if buf_words >= TARGET_WORDS:
            tail = flush()
            buf, buf_words = ([" ".join(tail)] if tail else []), len(tail)

    # Flush whatever remains, but if it is only the carried-over overlap, drop it.
    leftover = "\n\n".join(buf).strip()
    if leftover and _word_count(leftover) > OVERLAP_WORDS // 2:
        chunks.append(f"{heading}\n{leftover}".strip())
    return chunks


def _split_long_section(section: dict[str, str]) -> list[str]:
    """Split an over-long section: by paragraph first, falling back to sentences
    for any single paragraph that is still longer than MAX_WORDS."""
    heading = section["heading"].strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section["body"]) if p.strip()]

    units: list[str] = []
    for para in paragraphs:
        if _word_count(para) > MAX_WORDS:
            units.extend(s.strip() for s in _SENTENCE_SPLIT.split(para) if s.strip())
        else:
            units.append(para)
    return _pack_units(units, heading)


def chunk_document(doc: Document) -> list[dict]:
    """Produce heading-aware chunks for one document, with metadata."""
    cleaned = doc.get("cleaned_text") or clean_text(doc["raw_text"])
    sections = split_into_sections(cleaned, doc["title"])

    # Build raw section-level chunk texts (splitting long sections).
    raw: list[tuple[str, str]] = []  # (section_heading, chunk_text)
    for section in sections:
        text = _section_text(section)
        if _word_count(text) <= MAX_WORDS:
            raw.append((section["heading"].strip(), text))
        else:
            for part in _split_long_section(section):
                raw.append((section["heading"].strip(), part))

    # Merge genuine fragments (< MERGE_BELOW_WORDS) into the previous chunk of
    # the same document when the result still fits under MAX_WORDS.
    merged: list[tuple[str, str]] = []
    for heading, text in raw:
        if (
            merged
            and _word_count(text) < MERGE_BELOW_WORDS
            and _word_count(merged[-1][1]) + _word_count(text) <= MAX_WORDS
        ):
            merged[-1] = (merged[-1][0], f"{merged[-1][1]}\n\n{text}")
        else:
            merged.append((heading, text))

    stem = Path(doc["filename"]).stem
    return [
        {
            "id": f"{stem}::{idx}",
            "text": text,
            "source": doc["source"],
            "filename": doc["filename"],
            "title": doc["title"],
            "section_heading": heading,
            "chunk_index": idx,
            "word_count": _word_count(text),
        }
        for idx, (heading, text) in enumerate(merged)
    ]


def _hr(label: str = "") -> None:
    print("\n" + "=" * 78)
    if label:
        print(label)
        print("=" * 78)


def main() -> None:
    # --- Stage 1: load -----------------------------------------------------
    _hr("Stage 1/5: load documents")
    docs = load_documents()
    print(f"loaded {len(docs)} documents from {DOCUMENTS_DIR.relative_to(ROOT)}/")
    for d in docs:
        print(f"  - {d['source']:<58} words={_word_count(d['raw_text'])}")
    save_json(docs, RAW_LOADED_PATH)
    print(f"saved raw text -> {RAW_LOADED_PATH.relative_to(ROOT)}")

    # --- Stage 2: clean ----------------------------------------------------
    _hr("Stage 2/5: clean documents")
    for d in docs:
        d["cleaned_text"] = clean_text(d["raw_text"])
    sample = docs[2]  # create_resume.txt -- has Markdown links to verify flattening
    print(f"Inspecting one cleaned document: {sample['source']}\n")
    print("-" * 78)
    print(sample["cleaned_text"])
    print("-" * 78)

    # --- Stages 3-4: chunk -------------------------------------------------
    _hr("Stage 3-4/5: heading-aware chunking")
    print(f"params: TARGET_WORDS={TARGET_WORDS} MAX_WORDS={MAX_WORDS} "
          f"OVERLAP_WORDS={OVERLAP_WORDS} MERGE_BELOW_WORDS={MERGE_BELOW_WORDS}\n")
    all_chunks: list[dict] = []
    for d in docs:
        chunks = chunk_document(d)
        all_chunks.extend(chunks)
        print(f"  {d['filename']:<42} sections->chunks: {len(chunks):>2}")
    save_json(all_chunks, CHUNKS_PATH)
    print(f"\nsaved {len(all_chunks)} chunks -> {CHUNKS_PATH.relative_to(ROOT)}")

    # --- Stage 5: inspect --------------------------------------------------
    _hr("Stage 5/5: inspect 5 representative chunks")
    counts = [c["word_count"] for c in all_chunks]
    # Spread the 5 samples across the corpus (start, quartiles, end).
    n = len(all_chunks)
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    for i in idxs:
        c = all_chunks[i]
        print(f"\n--- chunk #{i}  [{c['id']}]  ({c['word_count']} words) ---")
        print(f"source : {c['source']}")
        print(f"title  : {c['title']}   |   section: {c['section_heading']}")
        print("text   :")
        print(c["text"])

    # --- Stats + checkpoint guardrails ------------------------------------
    _hr("Chunk statistics & checkpoint")
    empties = sum(1 for c in all_chunks if not c["text"].strip())
    html_left = sum(1 for c in all_chunks if _HTML_TAG.search(c["text"]) or "&" in c["text"] and re.search(r"&[a-z]+;|&#", c["text"]))
    print(f"total chunks      : {n}")
    print(f"word count min/avg/max : {min(counts)} / {sum(counts)//n} / {max(counts)}")
    print(f"empty chunks      : {empties}")
    print(f"chunks w/ HTML/entities : {html_left}")
    print(f"chunks under {MERGE_BELOW_WORDS}w : {sum(1 for w in counts if w < MERGE_BELOW_WORDS)}")
    print(f"chunks over {MAX_WORDS}w : {sum(1 for w in counts if w > MAX_WORDS)}")

    print("\nCheckpoint:")
    print(f"  [{'!' if n < 50 else 'ok'}] count >= 50 (got {n}) -- corpus is only "
          f"~{sum(_word_count(d['raw_text']) for d in docs)} words, so a smaller count is expected here")
    print(f"  [{'!' if n > 2000 else 'ok'}] count <= 2000 (got {n})")
    print(f"  [{'ok' if empties == 0 else '!'}] no empty chunks")
    print(f"  [{'ok' if html_left == 0 else '!'}] no HTML/entity artifacts")


if __name__ == "__main__":
    main()
