"""Content size handling: auto stops appending to large entries, and writes over
the Milvus VARCHAR limit fail with a clear error instead of a raw Milvus one."""

from __future__ import annotations

import pytest

from bigbrain import mcp_server
from bigbrain.config import Config
from bigbrain.store import (
    MAX_CONTENT_BYTES,
    REWRITE_THRESHOLD_CHARS,
    ContentTooLargeError,
    MemoryStore,
)

_TOPIC = "content size probe: a memory that keeps growing"
_TOO_BIG = "x" * (MAX_CONTENT_BYTES + 1)


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(Config(home=tmp_path))
    yield s
    s.close()


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    s = MemoryStore(Config(home=tmp_path))
    monkeypatch.setattr(mcp_server, "_store", s)
    yield mcp_server
    s.close()


def _seed(store: MemoryStore, size: int) -> int:
    mem, action = store.store(_TOPIC, "a" * size)
    assert action == "created"
    return mem.id


@pytest.mark.parametrize(
    "existing_size, addition, expected_action",
    [
        (100, "small addition", "merged"),
        (REWRITE_THRESHOLD_CHARS, "pushes it over", "needs_rewrite"),
        (REWRITE_THRESHOLD_CHARS + 5000, "already over", "needs_rewrite"),
    ],
)
def test_auto_appends_only_while_small(store, existing_size, addition, expected_action) -> None:
    mem_id = _seed(store, existing_size)
    before = store.get(mem_id).content

    _, action = store.store(_TOPIC, addition)

    assert action == expected_action
    after = store.get(mem_id).content
    if expected_action == "merged":
        assert after.endswith(addition)
    else:
        assert after == before, "needs_rewrite must not write anything"


def test_auto_is_noop_when_content_already_present(store) -> None:
    mem_id = _seed(store, REWRITE_THRESHOLD_CHARS + 10)
    existing = store.get(mem_id).content

    _, action = store.store(_TOPIC, existing[:50])

    assert action == "merged"
    assert store.get(mem_id).content == existing


@pytest.mark.parametrize("on_conflict", ["replace", "merge"])
def test_large_entry_accepts_replace_and_explicit_merge(store, on_conflict) -> None:
    mem_id = _seed(store, REWRITE_THRESHOLD_CHARS + 10)

    _, action = store.store(_TOPIC, "compacted rewrite", on_conflict=on_conflict)

    assert action == ("replaced" if on_conflict == "replace" else "merged")
    content = store.get(mem_id).content
    assert content.endswith("compacted rewrite")


@pytest.mark.parametrize("on_conflict", ["replace", "merge", "new"])
def test_writes_over_the_hard_limit_raise(store, on_conflict) -> None:
    mem_id = _seed(store, 100)
    before = store.get(mem_id).content

    with pytest.raises(ContentTooLargeError):
        store.store(_TOPIC, _TOO_BIG, on_conflict=on_conflict)

    assert store.get(mem_id).content == before


def test_create_and_update_over_the_hard_limit_raise(store) -> None:
    with pytest.raises(ContentTooLargeError):
        store.store("brand new oversized memory", _TOO_BIG)

    mem_id = _seed(store, 100)
    with pytest.raises(ContentTooLargeError):
        store.update(mem_id, content=_TOO_BIG)


def test_limit_counts_utf8_bytes(store) -> None:
    # Each character is 3 bytes, so this fits by length but not by bytes.
    wide = "€" * (MAX_CONTENT_BYTES // 3 + 1)
    assert len(wide) < MAX_CONTENT_BYTES
    with pytest.raises(ContentTooLargeError):
        store.store("utf-8 width probe", wide)


def test_mcp_store_reports_needs_rewrite_with_hint(tools) -> None:
    first = tools.memory_store(topic=_TOPIC, content="a" * (REWRITE_THRESHOLD_CHARS + 1))
    assert first["action"] == "created"

    out = tools.memory_store(topic=_TOPIC, content="new learning")

    assert out["action"] == "needs_rewrite"
    assert "replace" in out["hint"]
    assert out["memory"]["id"] == first["memory"]["id"]


@pytest.mark.parametrize("call", ["store", "update"])
def test_mcp_reports_rejected_instead_of_raising(tools, call) -> None:
    created = tools.memory_store(topic=_TOPIC, content="small")
    if call == "store":
        out = tools.memory_store(topic=_TOPIC, content=_TOO_BIG, on_conflict="replace")
    else:
        out = tools.memory_update(memory_id=created["memory"]["id"], content=_TOO_BIG)

    assert out["action"] == "rejected"
    assert out["error"] == "content_too_large"
