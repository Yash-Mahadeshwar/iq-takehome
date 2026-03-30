"""
Embedding service — wraps sentence-transformers for local, offline embedding.

Design rationale
────────────────
We use a local sentence-transformer model rather than a hosted embedding API
because:
1. No per-token cost for ingesting 10 K profiles (or re-ingesting later).
2. No network latency during batch ingestion.
3. The model is downloaded once (~90 MB) and cached by the transformers library.

Model choice: ``all-MiniLM-L6-v2``
- 384-dimensional vectors → compact ChromaDB storage.
- Trained on 1 B+ sentence pairs; excellent for semantic similarity.
- Inference: ~2 k sentences/sec on CPU — ingests 10 K profiles in < 1 min.
- Cosine similarity works well out of the box.

Alternative models (set EMBEDDING_MODEL env var):
- ``BAAI/bge-small-en-v1.5``   — slightly better BEIR benchmark, same size.
- ``BAAI/bge-large-en-v1.5``   — best quality, 1.3 GB, ~4× slower.
- ``multi-qa-MiniLM-L6-cos-v1``— fine-tuned for Q&A / query-passage matching.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the embedding model (downloaded on first call)."""
    model_name = get_settings().embedding_model
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded (dim=%d)", model.get_sentence_embedding_dimension())
    return model


class Embedder:
    """
    Thin wrapper around SentenceTransformer that provides consistent
    embedding of text strings (single or batch).
    """

    def __init__(self) -> None:
        self._model = _load_model()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed_one(self, text: str) -> list[float]:
        """Return a single embedding vector as a Python list of floats."""
        vec = self._model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()

    def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Return a list of embedding vectors for a batch of texts.

        Args:
            texts: Iterable of strings to embed.
            batch_size: Internal mini-batch size passed to the model.
            show_progress: Show tqdm progress bar (useful during ingestion).
        """
        vecs = self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return [v.tolist() for v in vecs]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Application-scoped singleton embedder."""
    return Embedder()
