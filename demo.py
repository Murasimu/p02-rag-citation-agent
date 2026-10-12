"""Demo: ask the Aurora Robotics wiki grounded questions.

Three question types demonstrate the grounding ladder:
1. Answerable   -> cited, confidence-scored answer from the corpus
2. Off-corpus   -> refusal + fallback to (mock) external search
3. (with --live) same flow against a real LLM

Usage:
  python demo.py            # scripted LLM, no network
  python demo.py --live     # live LLM via any OpenAI-compatible API
"""

from __future__ import annotations

import sys

from rag_agent.agent import RagAgent
from rag_agent.demo_corpus import build_corpus
from rag_agent.embedder import HashEmbedder
from rag_agent.retriever import HybridRetriever


def banner(t):
    print(f"\n{'=' * 62}\n{t}\n{'=' * 62}")


def mock_search(query):
    return [
        f"[web] Top result for '{query}': Aurora Robotics filed a sustainability "
        "report mentioning Scope 2 emissions targets (fictional demo source).",
        f"[web] Second result for '{query}': industry blog coverage (demo source).",
    ]


def make_agent(llm):
    corpus = build_corpus()
    return RagAgent(corpus, HybridRetriever(corpus, HashEmbedder()), llm, search=mock_search)


def scripted(replies):
    def llm(messages):
        return replies.pop(0)
    return llm


def show(q, ans):
    print(f"Q: {q}")
    print(f"A: {ans.text}")
    print(f"   confidence={ans.confidence}  grounded={ans.grounded}  source={ans.source}")
    for c in ans.citations:
        print(f"   [{c.id}] {c.title}: {c.snippet[:90]}...")


def main():
    banner("Q1 (answerable): vacation policy")
    llm = scripted([
        "Employees receive 25 paid vacation days per year [KB-102#0]. "
        "Up to five unused days can carry into the next year [KB-102#0]. "
        "A medical certificate is only needed after five consecutive sick days [KB-102#0]."
    ])
    show("How many vacation days do I get, and what carries over?",
         make_agent(llm).ask("How many vacation days do I get, and what carries over?"))

    banner("Q2 (off-corpus): carbon footprint")
    llm2 = scripted(["NO_ANSWER"])
    show("What is the company's carbon footprint?",
         make_agent(llm2).ask("What is the company's carbon footprint?"))

    banner("Q3 (ungrounded claims get stripped)")
    llm3 = scripted([
        "The CEO owns a yacht. "  # uncited -> stripped
        "Flights must be booked in economy class [KB-103#0]. "
        "The per-diem meal cap is 60 dollars domestic [KB-103#0]."
    ])
    show("What is the flight booking rule?",
         make_agent(llm3).ask("What is the flight booking rule?"))

    if "--live" in sys.argv:
        banner("Live LLM: three real questions")
        from rag_agent.llm import make_llm
        try:
            live = make_llm()
        except RuntimeError as e:
            print(f"skipping live demo: {e}")
            return
        agent = make_agent(live)
        for q in [
            "How many remote days are allowed per week?",
            "What happens when I discover a security incident?",
            "What's the CEO's favorite pizza topping?",
        ]:
            show(q, agent.ask(q))
            print(f"   audit: {[e['event'] for e in agent.audit[-2:]]}")


if __name__ == "__main__":
    main()
