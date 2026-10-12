"""Tests for the citation-grounded RAG agent. All offline & deterministic."""

import math

import pytest

from rag_agent.agent import Answer, Citation, RagAgent, strip_uncited_sentences
from rag_agent.corpus import Corpus, tokenize
from rag_agent.demo_corpus import build_corpus
from rag_agent.embedder import HashEmbedder
from rag_agent.retriever import HybridRetriever, cosine


# ---------------------------------------------------------------- corpus
class TestCorpus:
    def test_chunking_with_overlap(self):
        c = Corpus()
        c.add_document("D-1", "t", " ".join(f"word{i}" for i in range(1500)), chunk_size=600)
        assert len(c.chunks) > 1
        # chunk 1 should overlap chunk 0 (share words)
        t0, t1 = set(tokenize(c.chunks[0].text)), set(tokenize(c.chunks[1].text))
        assert t0 & t1, "expected 100-word overlap between consecutive chunks"

    def test_citation_ids_stable(self):
        c = build_corpus()
        ids = [ch.citation_id for ch in c.chunks]
        assert len(ids) == len(set(ids)), "citation ids must be unique"
        assert all(i.startswith("KB-") and "#" in i for i in ids)

    def test_bm25_ranks_relevant_doc_first(self):
        c = build_corpus()
        scores = c.bm25_scores("how many vacation days do employees get")
        best = c.chunks[max(range(len(scores)), key=lambda i: scores[i])]
        assert best.doc_id == "KB-102"

    def test_bm25_no_overlap_scores_zero(self):
        c = build_corpus()
        assert max(c.bm25_scores("quantum entanglement poetry")) == 0.0


# ---------------------------------------------------------------- embedder
class TestHashEmbedder:
    def test_deterministic_and_normalized(self):
        e = HashEmbedder()
        v1, v2 = e.embed(["hello world", "hello world"])
        assert v1 == v2
        assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-6)

    def test_cosine_self_is_one(self):
        e = HashEmbedder()
        v = e.embed(["the vacation policy says twenty five days"])[0]
        assert cosine(v, v) > 0.999


# ---------------------------------------------------------------- retriever
class TestRetriever:
    def test_hybrid_finds_right_doc(self):
        r = HybridRetriever(build_corpus(), HashEmbedder())
        hits = r.retrieve("expense receipts reimbursement deadline", top_k=2)
        assert hits[0].chunk.doc_id == "KB-103"

    def test_top_k_respected(self):
        r = HybridRetriever(build_corpus(), HashEmbedder())
        assert len(r.retrieve("office parking", top_k=3)) == 3


# ---------------------------------------------------------------- grounding
class TestStripUncited:
    def test_strips_uncited_keeps_cited(self):
        text = ("Employees get 25 vacation days [KB-102#0]. "
                "This sentence has no citation. "
                "Alcohol is not reimbursable [KB-103#0].")
        clean, used = strip_uncited_sentences(text, {"KB-102#0", "KB-103#0"})
        assert "vacation days" in clean and "not reimbursable" in clean
        assert "no citation" not in clean
        assert used == {"KB-102#0", "KB-103#0"}

    def test_invalid_citation_ids_dropped(self):
        text = "Claim with fake id [KB-999#7]. Real claim [KB-101#0]."
        clean, used = strip_uncited_sentences(text, {"KB-101#0"})
        assert "fake" not in clean
        assert used == {"KB-101#0"}


# ---------------------------------------------------------------- agent
def make_agent(llm, search=None, **kw):
    corpus = build_corpus()
    retriever = HybridRetriever(corpus, HashEmbedder())
    return RagAgent(corpus, retriever, llm, search=search, **kw)


class ScriptedLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        return self.reply


class TestRagAgent:
    def test_grounded_answer_with_citations(self):
        llm = ScriptedLLM(
            "Employees receive 25 paid vacation days per year [KB-102#0]. "
            "Up to five unused days carry over [KB-102#0]."
        )
        agent = make_agent(llm)
        ans = agent.ask("How many vacation days do I get?")
        assert ans.grounded and ans.source == "corpus"
        assert ans.confidence > 0.3
        assert [c.id for c in ans.citations] == ["KB-102#0"]

    def test_ungrounded_claims_stripped(self):
        llm = ScriptedLLM(
            "The moon is made of cheese. Vacation is 25 days [KB-102#0]."
        )
        agent = make_agent(llm)
        ans = agent.ask("vacation days?")
        assert "cheese" not in ans.text
        assert "25 days" in ans.text

    def test_llm_refusal_triggers_fallback(self):
        llm = ScriptedLLM("NO_ANSWER")
        agent = make_agent(llm, search=lambda q: [f"web result about {q}"])
        ans = agent.ask("What is the company's carbon footprint?")
        assert ans.source == "fallback_search"
        assert "web result" in ans.text

    def test_no_search_honest_unknown(self):
        llm = ScriptedLLM("NO_ANSWER")
        agent = make_agent(llm, search=None)
        ans = agent.ask("What is the CEO's favorite color?")
        assert ans.source == "none" and ans.confidence == 0.0

    def test_low_retrieval_score_gates_answer(self):
        # Off-corpus question: retrieval score below threshold -> fallback,
        # even though the LLM would happily hallucinate an answer.
        llm = ScriptedLLM("Sure, anything [KB-101#0].")
        agent = make_agent(llm, search=lambda q: ["search hit"], min_score=0.99)
        ans = agent.ask("vacation")
        assert ans.source == "fallback_search"

    def test_audit_trail_records_events(self):
        llm = ScriptedLLM("NO_ANSWER")
        agent = make_agent(llm, search=None)
        agent.ask("anything")
        events = [e["event"] for e in agent.audit]
        assert "retrieve" in events and "no_answer" in events
