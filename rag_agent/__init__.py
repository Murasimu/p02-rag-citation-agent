"""RAG agent with citation grounding.

Retrieves context, generates answers where every claim is bound to a
source chunk, flags low-confidence answers, and falls back to external
search when the local corpus can't support an answer.

Sub-modules:
- corpus.py     chunked document store + keyword index (BM25-style)
- embedder.py   embeddings interface (hash-based local, or OpenAI-compatible API)
- retriever.py  hybrid retrieval (keyword + vector) with relevance scores
- agent.py      answer generation with citation spans + confidence gating
"""

from .agent import Answer, Citation, RagAgent
from .corpus import Chunk, Corpus
from .retriever import HybridRetriever

__all__ = [
    "Answer",
    "Citation",
    "RagAgent",
    "RetrievalDoc",
    "Chunk",
    "Corpus",
    "HybridRetriever",
]
__version__ = "1.0.0"
