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
from typing import Any, Callable, Optional, TypeVar, Union

from mcp.server.fastmcp import FastMCP

from .config import Config
from .store import REWRITE_THRESHOLD_CHARS, ContentTooLargeError

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

# Memory ids are random int64s, mostly above 2**53. JSON clients that parse
# numbers as doubles (JavaScript) silently round them, so the tools take and
# return ids as strings. Integers are still accepted for older callers.
MemoryId = Union[str, int]


def _parse_id(value: MemoryId) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid memory id: {value!r}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text.lstrip("-").isdigit():
        raise ValueError(f"invalid memory id: {value!r}")
    return int(text)


def _wire(mem: Any) -> dict[str, Any]:
    data = mem.to_dict()
    data["id"] = str(data["id"])
    return data


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


_REWRITE_HINT = (
    "Not written: this memory is past the append threshold "
    f"({REWRITE_THRESHOLD_CHARS} chars). Re-store the same topic with "
    "on_conflict='replace' and content that merges your new learning into a "
    "compact current-truth rewrite of the existing content (keep live facts, "
    "drop superseded history)."
)


@mcp.tool(
    description=(
        "Store a durable piece of knowledge in long-term memory. `topic` is a "
        "short semantic key (it becomes the searchable embedding); `content` is "
        "the detailed knowledge. A near-duplicate topic updates the existing "
        "memory: on_conflict='auto' (default) appends while the entry is small, "
        "'replace' overwrites it with the content you pass, 'merge' always appends, "
        "'skip' leaves it, 'new' stores a separate entry. Returns the memory and "
        "the action taken (created/merged/replaced/skipped). If the action is "
        "'needs_rewrite', NOTHING was written: the existing entry is too large to "
        "keep appending to, so re-store the same topic with on_conflict='replace' "
        "and content that folds the new learning into a compact current-truth "
        "rewrite of the returned memory. An action of 'rejected' means the content "
        "exceeds the hard size limit."
    )
)
@_release_after
def memory_store(
    topic: str,
    content: str,
    tags: Optional[list[str]] = None,
    source: str = "",
    importance: float = 0.5,
    on_conflict: str = "auto",
    dedup: bool = True,
) -> dict[str, Any]:
    try:
        mem, action = store().store(
            topic,
            content,
            tags=tags or [],
            source=source,
            importance=importance,
            on_conflict=on_conflict,  # type: ignore[arg-type]
            dedup=dedup,
        )
    except ContentTooLargeError as err:
        return {"action": "rejected", "error": "content_too_large", "detail": str(err)}
    result: dict[str, Any] = {"action": action, "memory": _wire(mem)}
    if action == "needs_rewrite":
        result["hint"] = _REWRITE_HINT
    return result


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
    return [_wire(m) for m in results]


@mcp.tool(
    description=(
        "Fetch a single memory by its id (pass the id as a string). Returns null "
        "if not found."
    )
)
@_release_after
def memory_get(memory_id: MemoryId) -> Optional[dict[str, Any]]:
    mem = store().get(_parse_id(memory_id))
    return _wire(mem) if mem else None


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
    return [_wire(m) for m in results]


@mcp.tool(
    description=(
        "Update fields of an existing memory by id (pass the id as a string). "
        "Only provided fields change; "
        "changing the topic re-embeds the search key. Returns the updated memory "
        "or null if the id does not exist."
    )
)
@_release_after
def memory_update(
    memory_id: MemoryId,
    topic: Optional[str] = None,
    content: Optional[str] = None,
    tags: Optional[list[str]] = None,
    source: Optional[str] = None,
    importance: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    try:
        mem = store().update(
            _parse_id(memory_id),
            topic=topic,
            content=content,
            tags=tags,
            source=source,
            importance=importance,
        )
    except ContentTooLargeError as err:
        return {"action": "rejected", "error": "content_too_large", "detail": str(err)}
    return _wire(mem) if mem else None


@mcp.tool(
    description=(
        "Delete one or more memories by id (pass ids as strings). Returns the "
        "number that actually existed and were deleted."
    )
)
@_release_after
def memory_delete(memory_ids: list[MemoryId]) -> dict[str, int]:
    return {"deleted": store().delete([_parse_id(i) for i in memory_ids])}


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
