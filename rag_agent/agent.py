"""The RAG agent: generate answers bound to citations, gate on confidence,
fall back to external search when the corpus can't support an answer.

Grounding rules (the heart of the project):
- The LLM must answer using ONLY the retrieved chunks, citing [doc#chunk]
  for every claim. Sentences without citations are stripped post-hoc.
- Answer confidence = f(retrieval scores, number of supporting chunks).
- If the best retrieval score is below `min_score`, the agent refuses to
  answer from the corpus and calls the fallback search instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from .corpus import Corpus
from .retriever import HybridRetriever

LLM = Callable[[list[dict[str, str]]], str]
SearchFn = Callable[[str], list[str]]

SYSTEM_PROMPT = """You are a grounded question-answering agent.

Answer ONLY from the context chunks below.
- Cite every factual sentence with one or more citation ids like [KB-101#0].
- If the context does not contain the answer, reply exactly: NO_ANSWER
- Never use outside knowledge. Never invent citation ids.

Context chunks:
{context}"""


class Citation(BaseModel):
    id: str
    title: str
    snippet: str


class Answer(BaseModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    grounded: bool = True
    source: str = "corpus"  # corpus | fallback_search | none


@dataclass
class RagAgent:
    corpus: Corpus
    retriever: HybridRetriever
    llm: LLM
    search: SearchFn | None = None
    min_score: float = 0.015  # below this, corpus cannot support an answer
    top_k: int = 4

    audit: list[dict] = field(default_factory=list)

    def ask(self, question: str) -> Answer:
        scored = self.retriever.retrieve(question, top_k=self.top_k)
        self.audit.append({"event": "retrieve", "question": question,
                           "scores": [(s.chunk.citation_id, round(s.score, 4)) for s in scored]})

        if not scored or scored[0].score < self.min_score:
            self.audit.append({"event": "low_confidence", "reason": "retrieval_below_threshold"})
            return self._fallback(question)

        context = "\n\n".join(
            f"[{s.chunk.citation_id}] ({s.chunk.title})\n{s.chunk.text}" for s in scored
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ]
        raw = self.llm(messages)

        if raw.strip().upper().startswith("NO_ANSWER"):
            self.audit.append({"event": "llm_refused", "reason": "no_answer_in_context"})
            return self._fallback(question)

        valid_ids = {s.chunk.citation_id for s in scored}
        text, used = strip_uncited_sentences(raw, valid_ids)
        if not text:
            self.audit.append({"event": "ungrounded_answer_stripped"})
            return self._fallback(question)

        by_id = {s.chunk.citation_id: s.chunk for s in scored}
        citations = [
            Citation(id=cid, title=by_id[cid].title,
                     snippet=by_id[cid].text[:160].rstrip() + "...")
            for cid in sorted(used)
        ]
        # Confidence: fused retrieval score of best supporting chunk, lifted
        # slightly when multiple chunks were actually cited.
        support = max((s.score for s in scored if s.chunk.citation_id in used), default=0.0)
        confidence = min(1.0, support * (1.0 + 0.25 * (len(used) - 1)) * 12.0)

        self.audit.append({"event": "answer", "citations": sorted(used),
                           "confidence": round(confidence, 3)})
        return Answer(text=text, citations=citations, confidence=round(confidence, 3),
                      grounded=True, source="corpus")

    def _fallback(self, question: str) -> Answer:
        """Graceful degradation: try external search, else honest 'unknown'."""
        if self.search is not None:
            snippets = self.search(question)
            if snippets:
                self.audit.append({"event": "fallback_search", "snippets": len(snippets)})
                return Answer(
                    text="(Low corpus confidence -- answer from external search, "
                         "verify before use)\n" + "\n---\n".join(snippets[:3]),
                    citations=[Citation(id=f"web-{i+1}", title="external search",
                                        snippet=s[:160]) for i, s in enumerate(snippets[:3])],
                    confidence=0.35, grounded=False, source="fallback_search",
                )
        self.audit.append({"event": "no_answer"})
        return Answer(text="I don't have enough grounded information to answer that.",
                      confidence=0.0, grounded=False, source="none")


_CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+#\d+)\]")


def strip_uncited_sentences(text: str, valid_ids: set[str]) -> tuple[str, set[str]]:
    """Post-hoc grounding enforcement: drop sentences with no valid citation.

    Returns (clean_text, citation_ids_actually_used).
    """
    kept: list[str] = []
    used: set[str] = set()
    for para in text.split("\n"):
        if not para.strip():
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            ids = {m for m in _CITATION_RE.findall(sent) if m in valid_ids}
            if ids:
                used |= ids
                kept.append(sent.strip())
    return " ".join(kept), used
