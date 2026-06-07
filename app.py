"""Gradio web UI for The Unofficial Guide (Milestone 5).

A single-page interface over the grounded RAG pipeline: type a question, the
system retrieves the most relevant chunks, asks Groq's llama-3.3-70b-versatile
to answer *only* from those chunks, and shows the answer alongside the source
documents it was allowed to draw from.

Run:
    python app.py
then open http://localhost:7860

All grounding and source attribution live in src/query.ask(); this file is just
the interface, so the same behaviour is shared by the CLI test in query.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

# Make `src/` importable when running `python app.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from query import ask  # noqa: E402  (import after sys.path tweak)

EXAMPLE_QUESTIONS = [
    "When should I start preparing for technical interviews?",
    "What resource is recommended for exploring CS career paths?",
    "How do I build a resume with little programming experience?",
    "What advice is there for women or nonbinary students in CS?",
    "What can I do outside of class to build experience?",
]


def handle_query(question: str) -> tuple[str, str]:
    """Run one query and return (answer, formatted source list) for the UI."""
    result = ask(question)
    if result["sources"]:
        sources = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources = "(no sources — the guide doesn't cover this)"
    return result["answer"], sources


with gr.Blocks(title="The Unofficial Guide") as demo:
    gr.Markdown(
        "# The Unofficial Guide — Hunter CS\n"
        "Ask about courses, internships, interview prep, resumes, projects, and "
        "student life. Answers come **only** from the student-written guides, and "
        "every answer lists the documents it drew from. If the guides don't cover "
        "your question, the system will say so instead of guessing."
    )

    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. When should I start preparing for technical interviews?",
        autofocus=True,
    )
    ask_btn = gr.Button("Ask", variant="primary")

    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question)

    # Both the button and pressing Enter in the textbox run the query.
    ask_btn.click(handle_query, inputs=question, outputs=[answer, sources])
    question.submit(handle_query, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
