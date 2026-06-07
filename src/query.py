"""Grounded generation for The Unofficial Guide (Milestone 5).

End-to-end RAG: retrieve the top-k chunks, build a grounded prompt, call Groq's
llama-3.3-70b-versatile, and return an answer together with the source documents
the answer was allowed to draw from.

`ask(question)` is the single entry point used by the interface (app.py) and by
the CLI / eval harness at the bottom of this file.

Grounding is enforced in three layers, so it does not depend on the model
"behaving":
    1. System prompt -- the model is told to answer ONLY from the supplied
       context and to refuse with a fixed sentence when the context falls short.
       Temperature is near zero so it sticks to the text.
    2. Relevance gate -- if retrieval finds nothing close enough (cosine
       distance above RELEVANCE_THRESHOLD), we return the refusal *without ever
       calling the LLM*. An out-of-scope question can't be answered from
       training knowledge because the model never sees it.
    3. Programmatic attribution -- `sources` is derived from the retrieved
       chunks' metadata, never from whatever the model writes. Citation is
       guaranteed by code, not requested from the LLM.

Run `python src/query.py` to test grounded generation on the planning.md
evaluation queries plus one deliberately out-of-scope question.
"""

from __future__ import annotations

import functools
import os
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq

from vector_store import DEFAULT_TOP_K, RetrievedChunk, retrieve

# --- Config ----------------------------------------------------------------
MODEL = "llama-3.3-70b-versatile"   # planning.md -> Architecture (Groq, free tier)
TEMPERATURE = 0.1                   # low: stay anchored to the retrieved text
MAX_TOKENS = 700

# Cosine distance above which a chunk is treated as "not really relevant".
# Strong matches land well under 0.5 (see retrieve.py checkpoint); 1.0 leaves
# head-room for looser-but-valid matches while still gating off-topic queries.
RELEVANCE_THRESHOLD = 1.0

# Exact sentence the model must use (and that the gate returns) when the
# documents don't cover the question. Kept as a constant so both the prompt and
# our own refusals stay identical.
REFUSAL = "I don't have enough information on that."

SYSTEM_PROMPT = f"""You are The Unofficial Guide, a question-answering assistant \
for Hunter College computer science students. You answer strictly from a set of \
student-written advice documents that are provided to you with each question.

Follow these rules without exception:
- Use ONLY the information in the CONTEXT section below. Do not use any outside \
or prior knowledge, and do not infer facts that are not stated there.
- If the context does not contain enough information to answer the question, \
reply with exactly this sentence and nothing else: "{REFUSAL}"
- Never guess, speculate, or fill gaps with general knowledge, even if you think \
you know the answer.
- Because this is student advice rather than official policy, attribute claims \
to the guides using phrasing like "the guide recommends" or "according to \
<filename>".
- Keep the answer focused and concise, and base every sentence on the context."""


class Answer(TypedDict):
    answer: str          # the model's grounded answer (or the refusal sentence)
    sources: list[str]   # filenames the answer was allowed to draw from
    chunks: list[RetrievedChunk]  # the raw retrieved chunks (for inspection)


# --- Groq client (built once) ----------------------------------------------
@functools.lru_cache(maxsize=1)
def get_client() -> Groq:
    """Build the Groq client once, reading GROQ_API_KEY from .env / the env."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your free "
            "key from https://console.groq.com"
        )
    return Groq(api_key=api_key)


# --- Prompt building --------------------------------------------------------
def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as labelled, attributable context blocks."""
    blocks = []
    for c in chunks:
        header = f"[Source {c['rank']}: {c['filename']} -- {c['section_heading']}]"
        blocks.append(f"{header}\n{c['text'].strip()}")
    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "CONTEXT:\n"
        f"{format_context(chunks)}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using only the context above."
    )


def unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """Distinct source filenames, in retrieval-rank order (best match first)."""
    seen: list[str] = []
    for c in chunks:
        if c["filename"] not in seen:
            seen.append(c["filename"])
    return seen


# --- End-to-end -------------------------------------------------------------
def ask(question: str, k: int = DEFAULT_TOP_K) -> Answer:
    """Retrieve, generate a grounded answer, and attach source attribution.

    Returns a dict with `answer`, `sources` (list of filenames), and the raw
    retrieved `chunks`. Out-of-scope questions short-circuit to the refusal with
    no sources and no LLM call.
    """
    question = (question or "").strip()
    if not question:
        return Answer(answer="Please enter a question.", sources=[], chunks=[])

    chunks = retrieve(question, k=k)

    # Relevance gate: nothing close enough -> refuse before calling the model.
    if not chunks or chunks[0]["distance"] > RELEVANCE_THRESHOLD:
        return Answer(answer=REFUSAL, sources=[], chunks=chunks)

    response = get_client().chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, chunks)},
        ],
    )
    answer = response.choices[0].message.content.strip()

    # If the model refused, there is no grounded claim to attribute.
    refused = REFUSAL.lower() in answer.lower()
    sources = [] if refused else unique_sources(chunks)
    return Answer(answer=answer, sources=sources, chunks=chunks)


# --- Manual end-to-end test -------------------------------------------------
# Three in-scope evaluation queries from planning.md plus one deliberately
# out-of-scope question to confirm the system refuses instead of guessing.
_TEST_QUERIES = [
    "When should a Hunter CS student start seriously preparing for technical interviews?",
    "What resource does the guide recommend for exploring different computer science career paths?",
    "What should a Hunter CS student do if they have little or no programming experience for their resume?",
    "What are the official graduation requirements and exact credit count for the Hunter CS major?",  # out of scope
]


def main() -> None:
    for q in _TEST_QUERIES:
        result = ask(q)
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        print("-" * 80)
        print(result["answer"])
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  - {s}")
        else:
            print("\n(no sources cited)")


if __name__ == "__main__":
    main()
