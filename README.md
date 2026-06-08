# The Unofficial Guide — Project 1

A RAG question-answering system over student-written advice guides for Hunter
College CS students. Ask a question → it retrieves the most relevant chunks →
Groq's `llama-3.3-70b-versatile` answers **only** from those chunks and cites
the source documents, declining when the guides don't cover the question.

**Run:** `pip install -r requirements.txt` → copy `.env.example` to `.env` and
add your `GROQ_API_KEY` → `python src/ingest.py` → `python src/embed_store.py` →
`python app.py` (opens http://localhost:7860). CLI tests: `python src/query.py`,
`python src/evaluate.py`.

---

## Domain and Document Sources

**Domain:** practical, peer-to-peer advice for Hunter College CS students —
course planning, interview prep, resumes, internships, projects, community, and
gaining experience outside class. This is valuable because it's the informal
knowledge students normally get from clubs and upperclassmen, not official
catalog pages.

**Sources:** 10 student-written `.txt` guides in the repo:

| # | Guide | Path |
|---|-------|------|
| 1 | Career Paths (recommends roadmap.sh) | `documents/cs-club/career_paths.txt` |
| 2 | Create LinkedIn | `documents/cs-club/create_linkedin.txt` |
| 3 | Create Resume | `documents/cs-club/create_resume.txt` |
| 4 | Internships | `documents/cs-club/internships.txt` |
| 5 | Interview Prep (LeetCode, Grind 75, CSCI 235/335) | `documents/cs-club/interview_prep.txt` |
| 6 | Women in CS at Hunter | `documents/hunter_cs_extra_sources/women_in_cs_hunter.txt` |
| 7 | Hunter CS Course Planning | `documents/hunter_cs_extra_sources/hunter_cs_course_planning.txt` |
| 8 | Projects for Hunter CS Students | `documents/hunter_cs_extra_sources/projects_for_hunter_cs_students.txt` |
| 9 | Updated Interview Prep with NeetCode | `documents/hunter_cs_extra_sources/neetcode_interview_prep.txt` |
| 10 | Hackathons, Research, and CUNY Resources | `documents/hunter_cs_extra_sources/hackathons_research_cuny_resources.txt` |

---

## Chunking Strategy

Heading-aware recursive chunking ([src/ingest.py](src/ingest.py)): **300–450
words** per chunk, splitting on section headings then paragraphs, with
**~50–75 words overlap only when a long section is split** (no overlap for short,
self-contained sections to avoid duplicate chunks). Each chunk keeps its section
heading and `filename`/`section_heading`/`chunk_index` metadata. This fits the
documents because they are short advice guides with clear headings — one section
is the natural unit that answers a question, and fixed-size splitting would cut
a heading away from its advice. **Final count: 51 chunks across 10 documents.**

---

## Sample Chunks

| # | Source document | Section | Chunk text (preview) |
|---|-----------------|---------|----------------------|
| 1 | `career_paths.txt` | Career Paths | "There are many fields in Computer Science that you can pursue… roadmap.sh is an open-source community effort to make clear the process necessary to pursue a certain career path…" |
| 2 | `create_resume.txt` | LaTeX Edited Resume | "Resumes can be created any way you want, but for ease of ATS and organization, it is recommended to use templates such as Overleaf…" |
| 3 | `women_in_cs_hunter.txt` | Women in CS at Hunter | "Computer Science can feel intimidating for anyone, but it can feel even more isolating if you are a woman or nonbinary student in a class where you do not see many people like you…" |
| 4 | `neetcode_interview_prep.txt` | Why NeetCode? | "NeetCode (https://neetcode.io/) is one of the most popular interview prep platforms right now because it organizes problems by patterns instead of throwing random questions at you…" |
| 5 | `hackathons_research_cuny_resources.txt` | Final Takeaways | "If you want to grow as a CS student, you need to collect experiences outside class. That can be hackathons, research, CUNY Tech Prep, club projects, internships, or volunteering…" |

---

## Embedding Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim) in **ChromaDB**
with cosine distance, top-k = 4. Chosen because it runs locally with no API key
or cost, which fits a small (51-chunk) corpus.

**Production tradeoff reflection:** for a real deployment I'd weigh accuracy on
domain-specific text (Hunter terms like "CSCI 235", CUNY program names), context
length (MiniLM truncates at 256 tokens), latency/cost of a hosted embedding API
vs. local inference, and privacy of student-authored content. I'd benchmark a
stronger hosted model against MiniLM on my own eval queries before switching, and
keep ChromaDB for its metadata support (which enables source attribution).

---

## Retrieval Test Results

Top chunks (cosine distance, lower = closer) from `python src/evaluate.py`:

**Query A — "What resource does the guide recommend for exploring CS career paths?"**
1. `career_paths.txt :: Career Paths` (d=0.348)
2. `create_resume.txt :: General Guide` (d=0.458)
3. `hackathons…txt :: Final Takeaways` (d=0.470)
> **Why relevant:** the #1 chunk is the exact passage that introduces roadmap.sh
> as a tool for exploring developer career paths — it directly answers the query,
> hence the low distance.

**Query B — "What advice is there for women/nonbinary CS students who feel isolated?"**
1. `women_in_cs_hunter.txt :: Women in CS at Hunter` (d=0.257)
2. `women_in_cs_hunter.txt :: Outside Communities` (d=0.402)
3. `women_in_cs_hunter.txt :: Final Takeaways` (d=0.423)
> **Why relevant:** all three top chunks come from the one document written for
> this exact audience; #1 (the lowest distance in the whole eval set) is the
> section about feeling isolated, and #2 names support communities — together
> they cover the question.

**Query C — "What should a student with little programming experience do for their resume?"**
1. `hackathons…txt :: Final Takeaways` (d=0.317)
2. `create_resume.txt :: No Programming Experience?` (d=0.353)
3. `hunter_cs_course_planning.txt :: Where Do I Start?` (d=0.435)
> **Why relevant:** the #2 chunk (`create_resume.txt :: No Programming
> Experience?`) is the section that directly answers this query, so it is
> correctly retrieved. Note the #1 chunk is a generic "collect experiences"
> passage whose vocabulary overlaps many career queries — it edges out the
> resume chunk on distance but the resume chunk is still in the top-k, so the
> grounded answer is correct (see Failure Case for more on this overlap).

---

## Grounded Generation

Enforced in three layers ([src/query.py](src/query.py)) so it doesn't rely on
the model behaving:
1. **System prompt:** the model is told to use **only** the provided context, to
   never use prior knowledge "even if you think you know the answer," and to
   reply with the exact sentence *"I don't have enough information on that."*
   when the context falls short. Temperature = 0.1.
2. **Relevance gate:** if the best retrieved chunk's distance exceeds 1.0,
   `ask()` returns the refusal **without calling the LLM** — out-of-scope
   questions never reach the model.
3. **Programmatic attribution:** the source list is built in code from chunk
   `filename` metadata, not parsed from the model's text, so citation is
   guaranteed.

---

## Example Responses

**1. "What resource is recommended for exploring CS career paths?"**
> According to `career_paths.txt`, the guide recommends **roadmap.sh** as a
> resource for exploring different CS career paths, as it provides simple and
> objective roadmaps to pursue a career in development.
> **Retrieved from:** career_paths.txt, create_resume.txt, hackathons_research_cuny_resources.txt, hunter_cs_course_planning.txt

**2. "What should a student with little programming experience do for their resume?"**
> According to `create_resume.txt`, they should list all experiences, condense
> to 1 page to showcase the most technical experiences, and add technical
> projects built from class knowledge.
> **Retrieved from:** hackathons_research_cuny_resources.txt, create_resume.txt, hunter_cs_course_planning.txt

**3. Out-of-scope — "What are the official graduation requirements and exact credit count for the Hunter CS major?"**
> I don't have enough information on that.
> **Retrieved from:** (none)

---

## Query Interface

A Gradio web UI ([app.py](app.py)). **Input:** a single "Your question" textbox
(submit by button or Enter), plus clickable example questions. **Output:** an
**Answer** box (grounded text, naming source files inline) and a **Retrieved
from** box (programmatic bullet list of source filenames).

**Sample interaction transcript:**
```
Your question:  When should I start preparing for technical interviews?

Answer:         According to neetcode_interview_prep.txt, a student should take
                interview prep more seriously around CSCI 235, since data
                structures connect directly to interviews, and practice
                consistently by CSCI 335 for internships or new-grad roles.

Retrieved from: • neetcode_interview_prep.txt
                • hunter_cs_course_planning.txt
                • hackathons_research_cuny_resources.txt
```

---

## Evaluation Report

All 5 questions from [planning.md](planning.md), via `python src/evaluate.py`:

| # | Question | Expected | System response (summary) | Accuracy |
|---|----------|----------|---------------------------|----------|
| 1 | When to start interview prep? | Light in CSCI 135; serious by CSCI 235; consistent by CSCI 335 | Serious at CSCI 235, consistent by CSCI 335; **omits the early-light-prep nuance**. Cites neetcode_interview_prep.txt | **Partially accurate** |
| 2 | Resource for exploring career paths? | roadmap.sh | Names **roadmap.sh** with explanation. Cites career_paths.txt | **Accurate** |
| 3 | Resume with little experience? | List all, condense to 1 page, add projects | **Exactly that**, plus building projects from class. Cites create_resume.txt | **Accurate** |
| 4 | Advice for isolated women/nonbinary students? | Not behind; build community; outside communities (Rewriting the Code, Break Through Tech); mentors | "Not behind… act like you belong," but **omits community-building and named communities**. Cites women_in_cs_hunter.txt | **Partially accurate** |
| 5 | What to do outside class? | Projects, hackathons, research, CUNY Tech Prep, clubs, internships; commit to one | Lists hackathons, research, CUNY Tech Prep, clubs, internships, volunteering; commit to one. Cites hackathons_research_cuny_resources.txt | **Accurate** |

**Summary:** 3 accurate, 2 partially accurate, 0 inaccurate; no hallucinations,
and the out-of-scope control question was correctly refused. The two partials are
*under-answers* (correct but incomplete), not wrong answers.

---

## Failure Case

**Q4** — *advice for isolated women/nonbinary students.* The answer captured the
"you're not behind / act like you belong" framing but **omitted the
community-building advice and the named outside communities (Rewriting the Code,
Break Through Tech)** the expected answer calls for.

**Root cause — generation, not retrieval.** The chunk that names those
communities (`women_in_cs_hunter.txt :: Outside Communities`) **was retrieved at
#2, d=0.402** — the information was in the context window. The model anchored on
the single highest-ranked chunk (#1, d=0.257) and synthesized only from it,
ignoring the still-relevant #2. **Fix:** instruct the prompt to synthesize across
*all* provided chunks when multiple are relevant, and raise `max_tokens` slightly
for completeness.

---

## Spec Reflection

**Helped:** `planning.md`'s Retrieval Approach and Architecture diagram fixed the
model, top-k, store, and the retrieve→prompt→cited-answer flow, so I could direct
the AI tool with a precise target instead of "build RAG." The plan's example
refusal sentence went straight into the system prompt.

**Diverged:** the plan framed grounding as a prompt instruction only; I added a
**structural relevance gate** and **programmatic source attribution** in code,
because a prompt alone can be coaxed into off-topic answers and LLM-written
citations can be fabricated. Moving both into the pipeline makes them guarantees,
not hopes.

---

## AI Usage

**Instance 1 — generating Milestone 5 code.** I gave the AI my Architecture
diagram, the grounding requirement, the answer+sources format, and the Gradio
skeleton, and asked it to wire retrieval into a Groq call plus a UI. It produced
`src/query.py` and `app.py`. **I overrode** it to make source attribution
**programmatic** (from chunk metadata, not LLM-written) and to add a **relevance
gate** that refuses before calling the LLM on out-of-scope questions.

**Instance 2 — tightening grounding and fixing a Gradio crash.** I gave it the
draft prompt and the grounded-vs-non-grounded examples and asked it to make
grounding *enforced, not suggested*. **I changed** the prompt so the refusal is
an exact required sentence and prior knowledge is explicitly forbidden, and when
`app.py` crashed I removed the `show_copy_button` arg (unsupported in my Gradio
6.16) rather than pinning an older version.

---

## Demo Video

_Link:_ _(add after recording)_ — covers 3+ queries with visible source
citations, one query that works well (Q2), the Q4 failure case (narrated), the
out-of-scope refusal, and a walkthrough of the evaluation report.
