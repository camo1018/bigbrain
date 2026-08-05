"""MemoryStore: semantic memory backed by Milvus Lite + local embeddings.

Design (brain-like):
    - The embedding of a memory's `topic` is the vector key we search on.
    - `content` holds the detailed knowledge retrieved once a topic matches.
    - recall() pulls vector candidates, then reranks by a blend of semantic
      similarity, recency, and importance.
    - store() deduplicates: a near-identical topic merges into the existing
      memory instead of creating a duplicate.
"""

from __future__ import annotations

import math
import secrets
import time
from typing import Any, Literal

from .config import Config
from .embeddings import Embedder
from .models import Memory, now_ts

OnConflict = Literal["merge", "replace", "skip", "new"]

_MAX_ID = (1 << 63) - 1

# Milvus Lite holds an exclusive single-process lock on its data dir. Several
# editor windows can each run a bigbrain server against the same home, so
# opening the client may transiently collide; retry with capped backoff.
_LOCK_RETRY_ATTEMPTS = 50
_LOCK_RETRY_BASE_DELAY = 0.05
_LOCK_RETRY_MAX_DELAY = 0.5


def _new_id() -> int:
    return secrets.randbelow(_MAX_ID) + 1


class MemoryStore:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.from_env()
        self.config.ensure_home()
        self._embedder = Embedder(self.config.embed_model)
        self._client = None

    # -- infrastructure ---------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            self._connect_with_retry()
        return self._client

    def _connect_with_retry(self) -> None:
        """Open the Milvus Lite client, waiting out a sibling process that
        briefly holds the single-process data-dir lock."""
        from pymilvus import MilvusClient

        delay = _LOCK_RETRY_BASE_DELAY
        last_exc: Exception | None = None
        for _ in range(_LOCK_RETRY_ATTEMPTS):
            try:
                self._client = MilvusClient(uri=str(self.config.db_path))
                self._ensure_collection()
                return
            except Exception as exc:
                self._client = None
                if not _is_lock_error(exc):
                    raise
                last_exc = exc
                time.sleep(delay)
                delay = min(delay * 2, _LOCK_RETRY_MAX_DELAY)
        assert last_exc is not None
        raise last_exc

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        name = self.config.collection
        if self._client.has_collection(name):
            # An existing collection is not loaded automatically in a fresh
            # process; load it before any search/get/query.
            self._client.load_collection(name)
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("topic", DataType.VARCHAR, max_length=1024)
        schema.add_field("content", DataType.VARCHAR, max_length=65535)
        schema.add_field("tags", DataType.JSON)
        schema.add_field("source", DataType.VARCHAR, max_length=1024)
        schema.add_field("importance", DataType.FLOAT)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("updated_at", DataType.INT64)
        schema.add_field("access_count", DataType.INT64)
        schema.add_field("last_accessed", DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.config.embed_dim)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="FLAT",
            metric_type=self.config.metric,
        )
        self._client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
        )

    # -- write ------------------------------------------------------------

    def store(
        self,
        topic: str,
        content: str,
        *,
        tags: list[str] | None = None,
        source: str = "",
        importance: float = 0.5,
        on_conflict: OnConflict = "merge",
        dedup: bool = True,
    ) -> tuple[Memory, str]:
        """Store a memory. Returns (memory, action) where action is one of
        'created', 'merged', 'replaced', 'skipped'."""
        topic = topic.strip()
        if not topic:
            raise ValueError("topic must not be empty")
        tags = sorted({t.strip() for t in (tags or []) if t.strip()})
        importance = _clamp01(importance)

        vector = self._embedder.embed_one(topic)

        if dedup:
            existing = self._nearest(vector)
            if existing is not None and existing.similarity >= self.config.dedup_threshold:
                return self._resolve_conflict(
                    existing, topic, content, tags, source, importance, vector, on_conflict
                )

        mem = Memory(
            id=_new_id(),
            topic=topic,
            content=content,
            tags=tags,
            source=source,
            importance=importance,
        )
        self.client.insert(self.config.collection, data=[mem.to_row(vector)])
        return mem, "created"

    def _resolve_conflict(
        self,
        existing: Memory,
        topic: str,
        content: str,
        tags: list[str],
        source: str,
        importance: float,
        vector: list[float],
        on_conflict: OnConflict,
    ) -> tuple[Memory, str]:
        if on_conflict == "skip":
            return existing, "skipped"
        if on_conflict == "new":
            mem = Memory(
                id=_new_id(),
                topic=topic,
                content=content,
                tags=tags,
                source=source,
                importance=importance,
            )
            self.client.insert(self.config.collection, data=[mem.to_row(vector)])
            return mem, "created"

        if on_conflict == "replace":
            existing.content = content
            existing.topic = topic
        else:  # merge
            existing.content = _merge_content(existing.content, content)
            existing.importance = max(existing.importance, importance)

        existing.tags = sorted(set(existing.tags) | set(tags))
        if source:
            existing.source = source
        existing.updated_at = now_ts()

        # Topic may have shifted on replace; re-embed to keep the key accurate.
        new_vector = vector if on_conflict == "replace" else self._embedder.embed_one(existing.topic)
        self.client.upsert(self.config.collection, data=[existing.to_row(new_vector)])
        return existing, "merged" if on_conflict == "merge" else "replaced"

    def update(self, memory_id: int, **fields: Any) -> Memory | None:
        mem = self.get(memory_id)
        if mem is None:
            return None

        topic_changed = False
        for key, value in fields.items():
            if value is None:
                continue
            if key == "tags":
                mem.tags = sorted({t.strip() for t in value if t.strip()})
            elif key == "importance":
                mem.importance = _clamp01(float(value))
            elif key == "topic":
                new_topic = str(value).strip()
                topic_changed = new_topic != mem.topic
                mem.topic = new_topic
            elif key in ("content", "source"):
                setattr(mem, key, value)

        mem.updated_at = now_ts()
        vector = self._embedder.embed_one(mem.topic) if topic_changed else self._vector_of(memory_id)
        self.client.upsert(self.config.collection, data=[mem.to_row(vector)])
        return mem

    def delete(self, memory_ids: int | list[int]) -> int:
        ids = [memory_ids] if isinstance(memory_ids, int) else list(memory_ids)
        if not ids:
            return 0
        self.client.delete(self.config.collection, ids=ids)
        return len(ids)

    # -- read -------------------------------------------------------------

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        min_similarity: float = 0.0,
        tags: list[str] | None = None,
        source: str | None = None,
        touch: bool = True,
    ) -> list[Memory]:
        query = query.strip()
        if not query:
            return []

        vector = self._embedder.embed_query(query)
        candidates = max(limit * self.config.candidate_multiplier, limit)
        expr = _build_filter(tags, source)

        hits = self.client.search(
            self.config.collection,
            data=[vector],
            limit=candidates,
            output_fields=list(Memory._PERSISTED_FIELDS),
            search_params={"metric_type": self.config.metric},
            filter=expr,
        )
        rows = hits[0] if hits else []

        now = time.time()
        results: list[Memory] = []
        for hit in rows:
            similarity = _clamp01(float(hit["distance"]))
            if similarity < min_similarity:
                continue
            mem = Memory.from_entity(hit["entity"])
            mem.similarity = similarity
            mem.score = self._rerank_score(mem, similarity, now)
            results.append(mem)

        results.sort(key=lambda m: m.score, reverse=True)
        top = results[:limit]

        if touch and top:
            self._touch([m.id for m in top])
        return top

    def _rerank_score(self, mem: Memory, similarity: float, now: float) -> float:
        age_days = max(0.0, (now - mem.updated_at) / 86400.0)
        recency = math.pow(0.5, age_days / self.config.recency_half_life_days)
        c = self.config
        return (
            c.w_similarity * similarity
            + c.w_recency * recency
            + c.w_importance * mem.importance
        )

    def get(self, memory_id: int) -> Memory | None:
        rows = self.client.get(
            self.config.collection,
            ids=[memory_id],
            output_fields=list(Memory._PERSISTED_FIELDS),
        )
        if not rows:
            return None
        return Memory.from_entity(rows[0])

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        tags: list[str] | None = None,
        source: str | None = None,
    ) -> list[Memory]:
        expr = _build_filter(tags, source) or "id >= 0"
        rows = self.client.query(
            self.config.collection,
            filter=expr,
            output_fields=list(Memory._PERSISTED_FIELDS),
            limit=limit,
            offset=offset,
        )
        mems = [Memory.from_entity(r) for r in rows]
        mems.sort(key=lambda m: m.updated_at, reverse=True)
        return mems

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.config.collection)
        return int(stats.get("row_count", 0))

    # -- helpers ----------------------------------------------------------

    def _nearest(self, vector: list[float]) -> Memory | None:
        hits = self.client.search(
            self.config.collection,
            data=[vector],
            limit=1,
            output_fields=list(Memory._PERSISTED_FIELDS),
            search_params={"metric_type": self.config.metric},
        )
        rows = hits[0] if hits else []
        if not rows:
            return None
        mem = Memory.from_entity(rows[0]["entity"])
        mem.similarity = _clamp01(float(rows[0]["distance"]))
        return mem

    def _vector_of(self, memory_id: int) -> list[float]:
        rows = self.client.get(
            self.config.collection, ids=[memory_id], output_fields=["vector"]
        )
        return list(rows[0]["vector"])

    def _touch(self, ids: list[int]) -> None:
        rows = self.client.get(
            self.config.collection,
            ids=ids,
            output_fields=list(Memory._PERSISTED_FIELDS) + ["vector"],
        )
        now = now_ts()
        updates = []
        for row in rows:
            mem = Memory.from_entity(row)
            mem.access_count += 1
            mem.last_accessed = now
            updates.append(mem.to_row(list(row["vector"])))
        if updates:
            self.client.upsert(self.config.collection, data=updates)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        _release_milvus_server()


def _release_milvus_server() -> None:
    """Stop the in-process Milvus Lite server so the data-dir flock is dropped.

    ``MilvusClient.close()`` only tears down the gRPC client channel; the
    milvus_lite server it started keeps the data dir open (and its exclusive
    flock) until process exit. That would let a single long-lived window lock
    every sibling window out, so stop the server explicitly after each op.
    """
    try:
        from milvus_lite.server_manager import server_manager_instance
    except Exception:
        return
    try:
        server_manager_instance.release_all()
    except Exception:
        pass


def _is_lock_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "lock" in msg
        or "open local milvus" in msg
        or "resource temporarily unavailable" in msg
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _merge_content(existing: str, incoming: str) -> str:
    existing = existing.strip()
    incoming = incoming.strip()
    if not incoming or incoming in existing:
        return existing
    if not existing:
        return incoming
    return f"{existing}\n\n---\n\n{incoming}"


def _escape(value: str) -> str:
    # Escape for a single-quoted Milvus string literal.
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _build_filter(tags: list[str] | None, source: str | None) -> str:
    """Build a Milvus filter expression.

    Milvus Lite does not support json_contains/array_contains on JSON arrays, so
    tag matching is done via a LIKE against the JSON-serialized array, matching
    the quoted element (e.g. '"milvus"') to avoid substring false positives.
    """
    clauses: list[str] = []
    for tag in tags or []:
        tag = tag.strip()
        if tag:
            clauses.append(f"""tags like '%"{_escape(tag)}"%'""")
    if source:
        clauses.append(f"source == '{_escape(source)}'")
    return " and ".join(clauses)
