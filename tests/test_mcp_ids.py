"""MCP tools take and return memory ids as strings.

Ids are random int64s, mostly above 2**53, so a JSON client that parses numbers
as doubles (JavaScript) rounds them. These tests call the tool functions
directly against a temp store.
"""

from __future__ import annotations

import pytest

from bigbrain import mcp_server
from bigbrain.config import Config
from bigbrain.store import MemoryStore

_UNSAFE = 1 << 53


@pytest.fixture()
def tools(tmp_path, monkeypatch):
    s = MemoryStore(Config(home=tmp_path))
    monkeypatch.setattr(mcp_server, "_store", s)
    yield mcp_server
    s.close()


def _store_unsafe_id(tools) -> str:
    """Store a memory and rewrite its id to one a double cannot represent."""
    out = tools.memory_store(topic="precision probe", content="c", dedup=False)
    s = tools.store()
    mem = s.get(int(out["memory"]["id"]))
    vector = s._vector_of(mem.id)
    s.delete(mem.id)
    mem.id = 5465341493168734911
    assert float(mem.id) != mem.id
    s.client.upsert(s.config.collection, data=[mem.to_row(vector)])
    return str(mem.id)


def test_ids_are_returned_as_strings(tools) -> None:
    out = tools.memory_store(topic="string id", content="c")
    assert isinstance(out["memory"]["id"], str)
    assert all(isinstance(m["id"], str) for m in tools.memory_list())
    assert all(isinstance(m["id"], str) for m in tools.memory_recall("string id"))


def test_string_ids_round_trip_above_2_53(tools) -> None:
    mid = _store_unsafe_id(tools)
    assert int(mid) > _UNSAFE

    got = tools.memory_get(mid)
    assert got is not None and got["id"] == mid
    assert tools.memory_update(mid, content="updated")["content"] == "updated"

    # What a double-precision client would have sent instead.
    rounded = int(float(int(mid)))
    assert tools.memory_delete([rounded]) == {"deleted": 0}
    assert tools.memory_delete([mid]) == {"deleted": 1}
    assert tools.memory_get(mid) is None


def test_integer_ids_still_accepted(tools) -> None:
    mid = tools.memory_store(topic="int id", content="c")["memory"]["id"]
    assert tools.memory_get(int(mid))["id"] == mid
    assert tools.memory_delete([int(mid)]) == {"deleted": 1}


@pytest.mark.parametrize("bad", ["abc", "", "1.5", True])
def test_invalid_ids_rejected(tools, bad) -> None:
    with pytest.raises(ValueError):
        tools.memory_get(bad)
