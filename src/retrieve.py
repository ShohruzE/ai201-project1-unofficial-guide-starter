"""Test retrieval against the planning.md evaluation queries (Milestone 4).

Run:  python src/retrieve.py
Prints the top-k chunks + cosine distances for each evaluation question so you
can judge relevance before wiring in the LLM (Milestone 5).

Checkpoint target (planning.md): the best result for each query should be a
visibly relevant chunk with cosine distance < 0.5.
"""

from __future__ import annotations

import textwrap

from vector_store import DEFAULT_TOP_K, retrieve

# The five evaluation queries from planning.md -> Evaluation Plan.
EVAL_QUERIES = [
    "When should a Hunter CS student start seriously preparing for technical interviews?",
    "What resource does the guide recommend for exploring different computer science career paths?",
    "What should a Hunter CS student do if they have little or no programming experience for their resume?",
    "What advice does the guide give to women or nonbinary CS students who feel isolated in computer science classes?",
    "What should students do outside of classes to build experience as CS students?",
]

PREVIEW_CHARS = 320


def show_query(query: str, k: int = DEFAULT_TOP_K) -> float:
    """Print the top-k results for one query and return the best distance."""
    results = retrieve(query, k=k)
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    for r in results:
        preview = textwrap.shorten(" ".join(r["text"].split()), width=PREVIEW_CHARS, placeholder=" ...")
        print(f"\n  #{r['rank']}  distance={r['distance']:.3f}  "
              f"[{r['filename']} :: chunk {r['chunk_index']} :: {r['section_heading']}]")
        print(textwrap.fill(preview, width=78, initial_indent="      ", subsequent_indent="      "))
    return results[0]["distance"] if results else float("inf")


def main() -> None:
    best_distances: list[tuple[str, float]] = []
    for query in EVAL_QUERIES:
        best = show_query(query)
        best_distances.append((query, best))

    print("\n" + "=" * 80)
    print("CHECKPOINT: best (top-1) cosine distance per query  (target < 0.5)")
    print("=" * 80)
    all_pass = True
    for query, best in best_distances:
        ok = best < 0.5
        all_pass &= ok
        print(f"  [{'ok' if ok else '!!'}] {best:.3f}  {textwrap.shorten(query, width=64)}")
    print(f"\n{'ALL queries pass the < 0.5 checkpoint.' if all_pass else 'Some queries exceed 0.5 -- inspect those above.'}")


if __name__ == "__main__":
    main()
