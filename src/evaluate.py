"""Run the 5 planning.md evaluation queries end-to-end and print, for each:
the grounded answer, the programmatic source list, and the retrieved chunks with
cosine distances (so retrieval quality can be judged separately from accuracy).

Run:  python src/evaluate.py
"""

from __future__ import annotations

import textwrap

from query import ask

EVAL_QUERIES = [
    "When should a Hunter CS student start seriously preparing for technical interviews?",
    "What resource does the guide recommend for exploring different computer science career paths?",
    "What should a Hunter CS student do if they have little or no programming experience for their resume?",
    "What advice does the guide give to women or nonbinary CS students who feel isolated in computer science classes?",
    "What should students do outside of classes to build experience as CS students?",
]


def main() -> None:
    for i, q in enumerate(EVAL_QUERIES, start=1):
        result = ask(q)
        print("\n" + "=" * 90)
        print(f"Q{i}: {q}")
        print("=" * 90)
        print("RETRIEVED CHUNKS (rank | distance | filename :: section):")
        for c in result["chunks"]:
            preview = textwrap.shorten(" ".join(c["text"].split()), width=110, placeholder=" ...")
            print(f"  #{c['rank']}  d={c['distance']:.3f}  {c['filename']} :: {c['section_heading']}")
            print(f"        {preview}")
        print("\nANSWER:")
        print(textwrap.fill(result["answer"], width=88, initial_indent="  ", subsequent_indent="  "))
        print("\nSOURCES (programmatic):", ", ".join(result["sources"]) or "(none)")


if __name__ == "__main__":
    main()
