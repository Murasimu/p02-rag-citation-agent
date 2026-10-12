"""Synthetic knowledge corpus: chunked documents + inverted keyword index.

The corpus is deliberately small, deterministic, and domain-neutral so
tests are reproducible and the demo ships with realistic-but-fake content.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    """One addressable unit of knowledge, with stable citation id."""

    doc_id: str
    title: str
    text: str
    index: int  # chunk position within the document

    @property
    def citation_id(self) -> str:
        return f"{self.doc_id}#{self.index}"


@dataclass
class Corpus:
    """In-memory corpus with a BM25-style keyword index."""

    chunks: list[Chunk] = field(default_factory=list)
    _df: dict[str, int] = field(default_factory=dict)  # document frequency

    def add_document(self, doc_id: str, title: str, text: str, chunk_size: int = 600) -> None:
        """Split `text` into overlapping word chunks and index them."""
        words = text.split()
        step = max(1, chunk_size - 100)  # 100-word overlap between chunks
        i = 0
        idx = 0
        while i < len(words):
            part = " ".join(words[i : i + chunk_size])
            self.chunks.append(Chunk(doc_id, title, part, idx))
            idx += 1
            if i + chunk_size >= len(words):
                break
            i += step
        self._recompute_df()

    def _recompute_df(self) -> None:
        self._df = {}
        for chunk in self.chunks:
            for term in set(tokenize(chunk.text)):
                self._df[term] = self._df.get(term, 0) + 1

    # ---------------------------------------------------------------- BM25
    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        """BM25 relevance of every chunk to `query` (0.0 when no term overlap)."""
        terms = tokenize(query)
        if not self.chunks:
            return []
        n = len(self.chunks)
        avg_len = sum(len(tokenize(c.text)) for c in self.chunks) / n
        scores = []
        for chunk in self.chunks:
            tf = {}
            for t in tokenize(chunk.text):
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for term in terms:
                if term not in tf:
                    continue
                df = self._df.get(term, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (tf[term] * (k1 + 1)) / (
                    tf[term] + k1 * (1 - b + b * len(tokenize(chunk.text)) / max(avg_len, 1e-9))
                )
            scores.append(score)
        return scores
