"""bigbrain MCP server.

Exposes the shared MemoryStore as MCP tools so an agent can store and recall
knowledge natively in any chat. This is the central, persistent knowledge base:
prefer recalling from it at the start of a task, and store durable facts,
decisions, and learnings as you go.

Runs over stdio.
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable, Optional, TypeVar

from mcp.server.fastmcp import FastMCP

from .config import Config

_config = Config.from_env()

# `stateless_http=True` suits many short-lived clients (each Cursor workbench is
# its own MCP client): every request is self-contained, so there is no per-client
# session state to track when dozens connect and disconnect.
mcp = FastMCP(
    "bigbrain",
    host=_config.http_host,
    port=_config.http_port,
    stateless_http=True,
)

# Import lazily inside store() so the stdio initialize handshake stays instant
# and no embedding model / DB work happens until the first tool call.
_store: Any = None

# All tool bodies are serialized: the process keeps a single MemoryStore and
# releases the Milvus data-dir lock after each op. Serializing makes the
# open -> op -> release cycle atomic, so concurrent requests (the shared HTTP
# server handles them in a threadpool) can never release the in-process Milvus
# server out from under an in-flight op, while the per-op release still lets the
# CLI grab the lock in the gaps between operations.
_lock = threading.Lock()

_F = TypeVar("_F", bound=Callable[..., Any])


def store() -> Any:
    global _store
    if _store is None:
        from .store import MemoryStore

        _store = MemoryStore(_config)
    return _store


def _release_after(fn: _F) -> _F:
    """Serialize the call and close the Milvus client afterwards so the data-dir
    lock is held only for the operation, letting the CLI and any sibling process
    share the store."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with _lock:
            try:
                return fn(*args, **kwargs)
            finally:
                if _store is not None:
                    _store.close()

    return wrapper  # type: ignore[return-value]


@mcp.tool(
    description=(
        "Store a durable piece of knowledge in long-term memory. `topic` is a "
        "short semantic key (it becomes the searchable embedding); `content` is "
        "the detailed knowledge. Near-duplicate topics are merged by default so "
        "the same fact is not stored twice. Use this to remember decisions, "
        "facts, preferences, and learnings worth recalling later. Returns the "
        "stored memory and the action taken (created/merged/replaced/skipped)."
    )
)
@_release_after
def memory_store(
    topic: str,
    content: str,
    tags: Optional[list[str]] = None,
    source: str = "",
    importance: float = 0.5,
    on_conflict: str = "merge",
    dedup: bool = True,
) -> dict[str, Any]:
    mem, action = store().store(
        topic,
        content,
        tags=tags or [],
        source=source,
        importance=importance,
        on_conflict=on_conflict,  # type: ignore[arg-type]
        dedup=dedup,
    )
    return {"action": action, "memory": mem.to_dict()}


@mcp.tool(
    description=(
        "Recall relevant knowledge by meaning. Provide a natural-language query "
        "describing what you want to remember; results are ranked by a blend of "
        "semantic similarity, recency, and importance. Optionally filter by tags "
        "or source. Call this proactively at the start of a task to check what is "
        "already known."
    )
)
@_release_after
def memory_recall(
    query: str,
    limit: int = 5,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    min_similarity: float = 0.0,
) -> list[dict[str, Any]]:
    results = store().recall(
        query,
        limit=limit,
        tags=tags or None,
        source=source,
        min_similarity=min_similarity,
    )
    return [m.to_dict() for m in results]


@mcp.tool(description="Fetch a single memory by its id. Returns null if not found.")
@_release_after
def memory_get(memory_id: int) -> Optional[dict[str, Any]]:
    mem = store().get(memory_id)
    return mem.to_dict() if mem else None


@mcp.tool(
    description=(
        "List stored memories, most recently updated first. Optionally filter by "
        "tags or source. Useful for browsing rather than semantic search."
    )
)
@_release_after
def memory_list(
    limit: int = 50,
    offset: int = 0,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
) -> list[dict[str, Any]]:
    results = store().list(limit=limit, offset=offset, tags=tags or None, source=source)
    return [m.to_dict() for m in results]


@mcp.tool(
    description=(
        "Update fields of an existing memory by id. Only provided fields change; "
        "changing the topic re-embeds the search key. Returns the updated memory "
        "or null if the id does not exist."
    )
)
@_release_after
def memory_update(
    memory_id: int,
    topic: Optional[str] = None,
    content: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    importance: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    mem = store().update(
        memory_id,
        topic=topic,
        content=content,
        tags=tags,
        source=source,
        importance=importance,
    )
    return mem.to_dict() if mem else None


@mcp.tool(description="Delete one or more memories by id. Returns the number deleted.")
@_release_after
def memory_delete(memory_ids: list[int]) -> dict[str, int]:
    return {"deleted": store().delete(memory_ids)}


@mcp.tool(description="Return the total number of stored memories.")
@_release_after
def memory_count() -> dict[str, int]:
    return {"count": store().count()}


def serve_http(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Run the shared long-lived server over streamable HTTP.

    This is the recommended way to run bigbrain: one loopback-bound process that
    every Cursor workbench connects to by URL, instead of Cursor spawning a
    stdio subprocess per workbench.
    """
    if host is not None:
        mcp.settings.host = host
    if port is not None:
        mcp.settings.port = port
    mcp.run(transport="streamable-http")


def main() -> None:
    """stdio entrypoint (`bigbrain-mcp`), kept for backward compatibility.

    Prefer the shared HTTP server (`bigbrain serve` / `bigbrain install-server`).
    """
    mcp.run()


if __name__ == "__main__":
    main()
