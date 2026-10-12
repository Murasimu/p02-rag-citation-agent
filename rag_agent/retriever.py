"""Hybrid retrieval: BM25 keyword scores + vector cosine similarity,
reciprocal-rank-fused into one ranking with per-chunk relevance scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Chunk, Corpus
from .embedder import Embedder


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    keyword_score: float
    vector_score: float


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HybridRetriever:
    """Fuses keyword and vector rankings via reciprocal rank fusion (RRF)."""

    def __init__(self, corpus: Corpus, embedder: Embedder, k: int = 60.0) -> None:
        self.corpus = corpus
        self.embedder = embedder
        self.rrf_k = k
        self._chunk_vecs: list[list[float]] | None = None

    def _ensure_indexed(self) -> None:
        if self._chunk_vecs is None:
            self._chunk_vecs = self.embedder.embed([c.text for c in self.corpus.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        """Return the top-k chunks, best first, with fused relevance scores."""
        self._ensure_indexed()
        chunks = self.corpus.chunks
        if not chunks:
            return []

        kw = self.corpus.bm25_scores(query)
        qvec = self.embedder.embed([query])[0]
        vec = [cosine(qvec, cv) for cv in self._chunk_vecs]

        kw_rank = sorted(range(len(chunks)), key=lambda i: -kw[i])
        vec_rank = sorted(range(len(chunks)), key=lambda i: -vec[i])
        rrf_kw = {i: r for r, i in enumerate(kw_rank)}
        rrf_vec = {i: r for r, i in enumerate(vec_rank)}

        scored = []
        for i, chunk in enumerate(chunks):
            fused = 1.0 / (self.rrf_k + rrf_kw[i]) + 1.0 / (self.rrf_k + rrf_vec[i])
            scored.append(ScoredChunk(chunk, fused, kw[i], vec[i]))
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]
