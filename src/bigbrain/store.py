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

import logging
import math
import secrets
import time
from typing import Any, Literal

from .config import Config
from .embeddings import Embedder
from .models import Memory, now_ts

OnConflict = Literal["auto", "merge", "replace", "skip", "new"]

# Milvus rejects a VARCHAR longer than its declared max_length (counted in UTF-8
# bytes), and 65535 is also Milvus's ceiling, so this cannot simply be raised.
MAX_CONTENT_BYTES = 65535
# Past this size an append no longer reads as one memory; on_conflict="auto"
# asks the caller to rewrite the entry instead of growing it further.
REWRITE_THRESHOLD_CHARS = 16000


class ContentTooLargeError(ValueError):
    """Raised when a write would exceed the content column's hard limit."""

    def __init__(self, length: int, limit: int = MAX_CONTENT_BYTES) -> None:
        self.length = length
        self.limit = limit
        super().__init__(
            f"content is {length} bytes, over the {limit}-byte limit; rewrite it "
            "as a compact current-truth entry (on_conflict='replace') or split it "
            "into a separate, more specific memory"
        )


def _check_size(content: str) -> None:
    length = len(content.encode("utf-8"))
    if length > MAX_CONTENT_BYTES:
        raise ContentTooLargeError(length)

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
        schema.add_field("content", DataType.VARCHAR, max_length=MAX_CONTENT_BYTES)
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
        on_conflict: OnConflict = "auto",
        dedup: bool = True,
    ) -> tuple[Memory, str]:
        """Store a memory. Returns (memory, action) where action is one of
        'created', 'merged', 'replaced', 'skipped', or 'needs_rewrite'.

        on_conflict="auto" merges into a near-duplicate unless the merged content
        would pass REWRITE_THRESHOLD_CHARS; then nothing is written and the
        existing memory comes back with action 'needs_rewrite' so the caller can
        re-store a compacted version with on_conflict="replace". Any write over
        MAX_CONTENT_BYTES raises ContentTooLargeError."""
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

        _check_size(content)
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
            _check_size(content)
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
            _check_size(content)
            existing.content = content
            existing.topic = topic
        else:  # merge / auto
            merged = _merge_content(existing.content, content)
            if (
                on_conflict == "auto"
                and merged != existing.content
                and len(merged) > REWRITE_THRESHOLD_CHARS
            ):
                return existing, "needs_rewrite"
            _check_size(merged)
            existing.content = merged
            existing.importance = max(existing.importance, importance)

        existing.tags = sorted(set(existing.tags) | set(tags))
        if source:
            existing.source = source
        existing.updated_at = now_ts()

        # Topic may have shifted on replace; re-embed to keep the key accurate.
        new_vector = vector if on_conflict == "replace" else self._embedder.embed_one(existing.topic)
        self.client.upsert(self.config.collection, data=[existing.to_row(new_vector)])
        return existing, "replaced" if on_conflict == "replace" else "merged"

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
            elif key == "content":
                _check_size(value)
                mem.content = value
            elif key == "source":
                mem.source = value

        mem.updated_at = now_ts()
        vector = self._embedder.embed_one(mem.topic) if topic_changed else self._vector_of(memory_id)
        self.client.upsert(self.config.collection, data=[mem.to_row(vector)])
        return mem

    def delete(self, memory_ids: int | list[int]) -> int:
        ids = [memory_ids] if isinstance(memory_ids, int) else list(memory_ids)
        if not ids:
            return 0
        # Milvus echoes back every requested id whether or not it matched, so
        # count the ids that actually exist; a miss must report 0, not success.
        rows = self.client.get(self.config.collection, ids=ids, output_fields=["id"])
        existing = [int(r["id"]) for r in rows]
        if not existing:
            return 0
        self.client.delete(self.config.collection, ids=existing)
        return len(existing)

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
        """Memories ordered by recency of update, most recent first.

        Milvus returns query() rows in primary-key (insertion) order and Milvus
        Lite silently ignores order_by, so sorting server-side is not an option.
        Reading only limit/offset rows and sorting that page would make "recent
        first" true only within one arbitrary slice of oldest-inserted rows. So
        filter with the index where possible, then rank in Python: iterate every
        row (a query_iterator over the local store), sort by updated_at, and slice.
        """
        expr = _build_filter(tags, source) or "id >= 0"
        fields = list(Memory._PERSISTED_FIELDS)

        by_id: dict[int, dict] = {}
        # Keyed by primary key while iterating: without an mvcc timestamp to pin
        # the read, the Milvus Lite iterator can hand back a row twice across a
        # page boundary. Keying dedupes that at no cost.
        client = self.client
        logger = logging.getLogger("pymilvus")
        previous = logger.level
        logger.setLevel(max(previous, logging.ERROR))
        try:
            iterator = client.query_iterator(
                collection_name=self.config.collection,
                filter=expr,
                output_fields=fields,
                batch_size=500,
            )
            try:
                while True:
                    page = iterator.next()
                    if not page:
                        break
                    for entity in page:
                        by_id[int(entity["id"])] = entity
            finally:
                iterator.close()
        finally:
            logger.setLevel(previous)

        mems = [Memory.from_entity(r) for r in by_id.values()]
        mems.sort(key=lambda m: m.updated_at, reverse=True)
        return mems[offset : offset + limit]

    def count(self) -> int:
        stats = self.client.get_collection_stats(self.config.collection)
        return int(stats.get("row_count", 0))

    # -- replication ------------------------------------------------------

    def export_rows(self, *, batch: int = 500) -> list[dict]:
        """Every stored memory as a plain row, for syncing with another store.

        The vector is left out: it is derived from the topic, so the receiving side
        recomputes it with its own model instead of trusting ours.

        Paging uses the query iterator rather than limit/offset because Milvus caps
        the offset+limit window, which a store this is meant to grow would hit.
        """
        fields = list(Memory._PERSISTED_FIELDS)
        # Keyed by primary key while iterating: without an mvcc timestamp to pin the
        # read, the Milvus Lite iterator can hand back a row twice across a page
        # boundary. Keying dedupes that at no cost.
        by_id: dict[int, dict] = {}

        # That same missing timestamp makes the iterator log a warning per page about
        # falling back to a client-side one. It is expected, and would otherwise be
        # the loudest thing a sync prints. Connect before muting: pymilvus reinstates
        # its own logger level when a client is constructed, which would undo this.
        client = self.client
        logger = logging.getLogger("pymilvus")
        previous = logger.level
        logger.setLevel(max(previous, logging.ERROR))
        try:
            iterator = client.query_iterator(
                collection_name=self.config.collection,
                filter="id >= 0",
                output_fields=fields,
                batch_size=batch,
            )
            try:
                while True:
                    page = iterator.next()
                    if not page:
                        break
                    for entity in page:
                        by_id[int(entity["id"])] = {
                            k: entity[k] for k in fields if k in entity
                        }
            finally:
                iterator.close()
        finally:
            logger.setLevel(previous)

        # A short export is dangerous rather than merely wrong: a sync compares this
        # list against the peer's, so silently missing memories read as deletions and
        # would be propagated. Fail loudly instead.
        expected = self.count()
        if len(by_id) < expected:
            raise RuntimeError(
                f"exported only {len(by_id)} of {expected} memories from "
                f"{self.config.db_path}; refusing to hand back a partial export"
            )
        return list(by_id.values())

    def import_rows(self, rows: list[dict]) -> int:
        """Write rows from another store verbatim, keeping their ids and timestamps.

        This deliberately bypasses store(): replication has to preserve identity and
        history, where store() would mint a new id and dedup-merge the content.
        """
        if not rows:
            return 0
        payload = []
        for row in rows:
            mem = Memory.from_entity(row)
            payload.append(mem.to_row(self._embedder.embed_one(mem.topic)))
        self.client.upsert(self.config.collection, data=payload)
        return len(payload)

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
            # Flush pending writes before tearing the client down. Milvus Lite
            # buffers upserts and deletes in a WAL, and the server is released
            # right after this (the MCP layer closes after every op). Flushing
            # first is cheap and makes sure those writes are persisted rather
            # than relying on WAL replay in the next process.
            try:
                self._client.flush(self.config.collection)
            except Exception:
                # Best effort: never let a flush failure mask the caller's
                # error or keep the server (and its data-dir lock) held.
                pass
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
    _close_milvus_connections()
    try:
        from milvus_lite.server_manager import server_manager_instance
    except Exception:
        return
    try:
        server_manager_instance.release_all()
    except Exception:
        pass


def _close_milvus_connections() -> None:
    """Close pymilvus's pooled gRPC connections.

    In pymilvus 3.x, ``MilvusClient.close()`` only drops the client's reference
    to a connection shared through ``ConnectionManager``; the gRPC channel stays
    open even with no clients left. Each op here starts Milvus Lite on a fresh
    port, so every op registers a new connection, and a long-lived server leaks
    its channel pipes until it hits the open-file limit.
    """
    try:
        from pymilvus.client.connection_manager import ConnectionManager
    except Exception:
        return
    try:
        ConnectionManager.get_instance().close_all()
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
