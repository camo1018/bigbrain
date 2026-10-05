"""Regression tests: deleted memories must not come back via get/recall/list.

Once delete() reports success, the id must be gone from get(), recall(), and
list(). That must hold within the same session and after the store is closed
and reopened, which is what the MCP server does after every op (close() flushes
and then releases the Milvus Lite server).
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


def _assert_gone(store: MemoryStore, memory_id: int, query: str) -> None:
    assert store.get(memory_id) is None, "get() still returns the deleted memory"
    hits = store.recall(query, limit=10, min_similarity=0.0)
    assert all(h.id != memory_id for h in hits), "recall() still surfaces the deleted memory"
    listed = store.list(limit=100)
    assert all(m.id != memory_id for m in listed), "list() still includes the deleted memory"


def test_deleted_memory_disappears_from_get_recall_and_list(store: MemoryStore) -> None:
    mem, action = store.store(
        topic="delete regression: a memory that must stay deleted",
        content="if this entry shows up after delete, delete is broken",
        tags=["repro"],
        importance=0.1,
    )
    assert action == "created"
    assert store.get(mem.id) is not None

    assert store.delete(mem.id) == 1
    _assert_gone(store, mem.id, "a memory that must stay deleted")


def test_delete_survives_close_and_reopen(tmp_path) -> None:
    first = MemoryStore(Config(home=tmp_path))
    keep, _ = first.store(topic="delete regression: survivor", content="stays", dedup=False)
    gone, _ = first.store(topic="delete regression: tombstoned", content="goes", dedup=False)
    first.close()

    # Mirror the MCP server: each op runs against a freshly opened client and
    # closes (releasing the Milvus Lite server) right after.
    assert first.delete(gone.id) == 1
    first.close()

    reopened = MemoryStore(Config(home=tmp_path))
    try:
        assert reopened.get(keep.id) is not None
        _assert_gone(reopened, gone.id, "delete regression: tombstoned")
    finally:
        reopened.close()
