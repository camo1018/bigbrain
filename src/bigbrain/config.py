"""Configuration for the bigbrain memory store.

All values can be overridden with environment variables so the CLI and the MCP
server behave identically without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


DEFAULT_HOME = Path.home() / ".bigbrain"


@dataclass(frozen=True)
class Config:
    """Runtime configuration.

    The vector is the embedding of a memory's *topic* (the fast index / key),
    while the content field holds the detailed knowledge we retrieve.
    """

    home: Path = DEFAULT_HOME
    collection: str = "memories"

    # HTTP transport for the shared long-lived server (see `bigbrain serve`).
    # Running one loopback-bound server that every Cursor workbench connects to
    # by URL avoids Cursor spawning a stdio subprocess per workbench (which can
    # saturate its client-creation IPC) and keeps a single owner of the DB.
    http_host: str = "127.0.0.1"
    http_port: int = 8765

    # Local embedding model (no API key, offline after first download).
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    # Embeddings are unit-normalized, so inner product == cosine similarity and
    # the metric value can be read directly as a similarity (higher = better).
    metric: str = "IP"

    # Ranking: recall reranks vector candidates by a weighted blend of semantic
    # similarity, recency, and importance so retrieval feels brain-like.
    w_similarity: float = 0.6
    w_recency: float = 0.25
    w_importance: float = 0.15
    recency_half_life_days: float = 30.0

    # Pull this many raw vector matches before reranking down to `limit`.
    candidate_multiplier: int = 6

    # On store, if the nearest existing topic is at least this cosine-similar,
    # treat it as the same memory and merge instead of inserting a duplicate.
    dedup_threshold: float = 0.92

    @property
    def db_path(self) -> Path:
        return self.home / "memory.db"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            home=_env_path("BIGBRAIN_HOME", DEFAULT_HOME),
            collection=os.environ.get("BIGBRAIN_COLLECTION", "memories"),
            embed_model=os.environ.get("BIGBRAIN_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
            embed_dim=_env_int("BIGBRAIN_EMBED_DIM", 384),
            w_similarity=_env_float("BIGBRAIN_W_SIMILARITY", 0.6),
            w_recency=_env_float("BIGBRAIN_W_RECENCY", 0.25),
            w_importance=_env_float("BIGBRAIN_W_IMPORTANCE", 0.15),
            recency_half_life_days=_env_float("BIGBRAIN_RECENCY_HALF_LIFE_DAYS", 30.0),
            candidate_multiplier=_env_int("BIGBRAIN_CANDIDATE_MULTIPLIER", 6),
            dedup_threshold=_env_float("BIGBRAIN_DEDUP_THRESHOLD", 0.92),
            http_host=os.environ.get("BIGBRAIN_HTTP_HOST", "127.0.0.1"),
            http_port=_env_int("BIGBRAIN_HTTP_PORT", 8765),
        )

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True)
        return self.home
