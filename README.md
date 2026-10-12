# RAG Agent with Citation Grounding

**Project 2 / 12 — Agentic AI portfolio.**
*Shows: you can prevent hallucinations at scale.*

A retrieval-augmented agent where every claim in every answer is bound to
a source chunk, low-confidence answers are gated, and unanswerable
questions degrade gracefully to external search instead of hallucinating.

## Why this matters

Vanilla RAG still hallucinates: the model pads answers with outside
knowledge, cites nothing, and sounds equally confident when right and
wrong. This project implements the four defenses that stop that at scale:

1. **Cite every claim** — the LLM must answer only from retrieved chunks,
   tagging each fact with `[doc#chunk]` ids.
2. **Post-hoc enforcement** — sentences without a *valid* citation are
   stripped from the answer, even if the model wrote them. Invented
   citation ids are dropped too.
3. **Confidence gating** — answers carry a confidence derived from
   retrieval scores and citation support; below-threshold retrieval means
   the agent won't answer from the corpus at all.
4. **Graceful fallback** — when the corpus can't support an answer, the
   agent calls external search and labels the result as ungrounded, or
   honestly says "I don't have enough grounded information."

## Architecture

```
question ──> HybridRetriever ──┬─ BM25 keyword index
              (RRF fusion)     └─ vector cosine (embedder)
                                    │
                        top-k chunks + fused scores
                                    │
                    score < min_score? ──yes──> fallback search / honest unknown
                                    │no
                        LLM (context = cited chunks only)
                                    │
                        strip_uncited_sentences()  ← grounding enforcement
                                    │
                        Answer(text, citations, confidence, source)
```

- `rag_agent/corpus.py` — chunked documents + BM25 index
- `rag_agent/embedder.py` — `HashEmbedder` (offline) and `ApiEmbedder` (live)
- `rag_agent/retriever.py` — hybrid keyword+vector retrieval with RRF fusion
- `rag_agent/agent.py` — grounding loop, confidence gating, search fallback
- `rag_agent/demo_corpus.py` — synthetic "Aurora Robotics" wiki (100% fictional)

## Quickstart

```bash
uv pip install -e ".[demo]"
python demo.py            # deterministic demo, no network
python demo.py --live     # live LLM (any OpenAI-compatible API)
```

Set `LLM_API_KEY` (plus optional `LLM_BASE_URL`, `LLM_MODEL`) for the live
demo. The offline demo uses a deterministic hash embedder, so retrieval
behavior is fully reproducible in tests.

## Tests

```bash
pytest -q
```

15 tests cover: overlapping chunking, stable citation ids, BM25 relevance,
embedder determinism/normalization, hybrid retrieval accuracy, citation
stripping (uncited and fake-id sentences), grounded answers with
confidence, refusal → fallback search → honest unknown, retrieval-score
gating, and audit events.

## Design decisions

- **Post-hoc stripping over prompt-only trust** — prompts alone can't
  guarantee grounding; a regex pass that deletes uncited sentences makes
  grounding a hard invariant, not a request.
- **Hybrid retrieval (BM25 + vectors)** — keyword search catches exact
  terms (part numbers, policy names) that embeddings blur; reciprocal rank
  fusion combines them without needing score calibration.
- **Offline-first embedder** — the hash embedder keeps the whole system
  testable without network; `ApiEmbedder` swaps in for production.
- **Confidence is computed, not asked** — models can't self-report
  calibration, so confidence derives from retrieval scores and citation
  support, which are measurable.

## Portfolio context

Project 2 of a 12-project agentic-AI portfolio (structured output, RAG
grounding, ReAct planning, tool orchestration, memory, HITL approval,
cost routing, event automation, debate, self-eval, observability, OSS
contribution). See the profile README for the full map.
