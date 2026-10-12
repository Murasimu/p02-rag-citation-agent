"""Embeddings interface.

Two backends, same contract (`embed(texts) -> list[vector]`):

- `HashEmbedder` -- deterministic feature-hashing embedder (pure stdlib).
  Zero network, good enough to demonstrate semantic-ish retrieval on a
  small corpus, and keeps tests fast and offline.
- `ApiEmbedder` -- any OpenAI-compatible /embeddings endpoint, for
  production-quality vectors in the live demo.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol

from .corpus import tokenize

DIM = 256


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Feature-hashing embedder with sublinear TF weighting + L2 norm."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            counts: dict[int, int] = {}
            for tok in tokenize(text):
                h = int.from_bytes(
                    hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big"
                )
                counts[h % self.dim] = counts.get(h % self.dim, 0) + 1
            for bucket, c in counts.items():
                vec[bucket] = 1.0 + math.log(c)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ApiEmbedder:
    """OpenAI-compatible /embeddings client (network use is opt-in)."""

    def __init__(self, model: str | None = None, dim: int = DIM) -> None:
        key = os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError("LLM_API_KEY is not set (see .env.example)")
        from openai import OpenAI  # lazy: demo-only dependency

        self.client = OpenAI(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=key,
        )
        self.model = model or os.environ.get("EMBED_MODEL", "text-embedding-3-small")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
