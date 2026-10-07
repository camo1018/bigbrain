"""Regression test: deleted memories must not come back via get/recall/list.

Observed against a real store (~/.bigbrain/memory.db) that had been through
many store/update/delete cycles: after MemoryStore.delete([id]) returned 1,
`get(id)` still returned the memory, `recall(...)` kept surfacing it, and its
`access_count` continued to increase — i.e. the delete never actually reached
persistent state.

The most likely cause is id precision, not persistence: ids are random int64s,
mostly above 2**53, so a JSON client that parses numbers as doubles rounds them
and the delete targets a nonexistent id. delete() used to report len(ids) even
on a miss, which hid that. These tests pin the invariant at the store level,
across a close/reopen, and for misses; the MCP tests pin string-id handling.
"""
from __future__ import annotations

import pytest

from bigbrain.config import Config
from bigbrain.store import MemoryStore


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(Config(home=tmp_path))
    yield s
    s.close()


def _assert_gone(store: MemoryStore, mem_id: int) -> None:
    assert store.get(mem_id) is None, "get() still returns the deleted memory"
    hits = store.recall("a memory that must stay deleted", limit=10, min_similarity=0.0)
    assert all(h.id != mem_id for h in hits), "recall() still surfaces the deleted memory"
    listed = list(store.list(limit=100))
    assert all(m.id != mem_id for m in listed), "list() still includes the deleted memory"


def test_deleted_memory_disappears_from_get_recall_and_list(store: MemoryStore) -> None:
    mem, action = store.store(
        topic="bigbrain delete-recall regression: a memory that must stay deleted",
        content="if this entry shows up after test teardown, delete is broken",
        tags=["repro"],
        importance=0.1,
    )
    assert action == "created"
    assert store.get(mem.id) is not None

    deleted = store.delete(mem.id)
    assert deleted == 1

    _assert_gone(store, mem.id)


def test_delete_survives_close_and_reopen(tmp_path) -> None:
    """The close() path is what the MCP server runs after every op, so the delete
    has to be visible to a fresh MemoryStore on the same data dir."""
    first = MemoryStore(Config(home=tmp_path))
    mem, _ = first.store(
        topic="bigbrain delete-recall regression: a memory that must stay deleted",
        content="must not come back after a reopen",
        importance=0.1,
    )
    keep, _ = first.store(
        topic="unrelated memory that should survive the reopen",
        content="control row",
        importance=0.1,
        dedup=False,
    )
    assert first.delete(mem.id) == 1
    first.close()

    second = MemoryStore(Config(home=tmp_path))
    try:
        _assert_gone(second, mem.id)
        assert second.get(keep.id) is not None
    finally:
        second.close()


def test_delete_of_missing_id_returns_zero(store: MemoryStore) -> None:
    mem, _ = store.store(topic="present", content="stays", importance=0.1)
    # A rounded id (what a double-precision JSON client sends) must not count.
    assert store.delete(mem.id + 1) == 0
    assert store.delete([mem.id + 1, mem.id]) == 1
    assert store.delete([]) == 0


def test_delete_is_idempotent(store: MemoryStore) -> None:
    mem, _ = store.store(
        topic="bigbrain delete-recall regression: idempotent delete",
        content="deleting twice must not error or resurrect",
        tags=["repro"],
        importance=0.1,
    )
    assert store.delete(mem.id) == 1
    # Second delete is a no-op that reports 0 removed, not an error.
    assert store.delete(mem.id) == 0
    assert store.get(mem.id) is None
