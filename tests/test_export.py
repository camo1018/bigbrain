"""Tests for the replication export.

A stub client stands in for Milvus so these stay fast, and so the awkward paging
behaviour that export_rows has to defend against can be reproduced on demand.
"""

from __future__ import annotations

import pytest

from bigbrain.config import Config
from bigbrain.store import MemoryStore


def entity(mid: int, topic: str = "topic") -> dict:
    return {
        "id": mid,
        "topic": topic,
        "content": "content",
        "tags": [],
        "source": "",
        "importance": 0.5,
        "created_at": 1,
        "updated_at": 2,
        "access_count": 0,
        "last_accessed": 0,
        "vector": [0.0] * 384,
    }


class FakeIterator:
    def __init__(self, pages: list[list[dict]]) -> None:
        self._pages = list(pages)
        self.closed = False

    def next(self) -> list[dict]:
        return self._pages.pop(0) if self._pages else []

    def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, pages: list[list[dict]], row_count: int) -> None:
        self.pages = pages
        self.row_count = row_count
        self.iterator: FakeIterator | None = None

    def query_iterator(self, **_kwargs):
        self.iterator = FakeIterator(self.pages)
        return self.iterator

    def get_collection_stats(self, _name: str) -> dict:
        return {"row_count": self.row_count}


def store_with(tmp_path, pages: list[list[dict]], row_count: int) -> MemoryStore:
    store = MemoryStore(Config(home=tmp_path))
    store._client = FakeClient(pages, row_count)
    return store


@pytest.mark.parametrize(
    "pages,row_count,want_ids",
    [
        pytest.param([[entity(1), entity(2)]], 2, {1, 2}, id="single-page"),
        pytest.param([[entity(1)], [entity(2)]], 2, {1, 2}, id="two-pages"),
        pytest.param([], 0, set(), id="empty-store"),
        pytest.param(
            [[entity(1), entity(2)], [entity(2)]],
            2,
            {1, 2},
            id="row-repeated-across-a-page-boundary",
        ),
        pytest.param(
            [[entity(1)], [entity(1)], [entity(1)]],
            1,
            {1},
            id="row-repeated-many-times",
        ),
    ],
)
def test_export_returns_each_memory_once(tmp_path, pages, row_count, want_ids):
    rows = store_with(tmp_path, pages, row_count).export_rows()
    assert {int(r["id"]) for r in rows} == want_ids
    assert len(rows) == len(want_ids)


def test_export_omits_the_vector(tmp_path):
    """Vectors are derived from the topic and recomputed on import, so sending them
    would only bloat the payload and risk crossing embedding models."""
    rows = store_with(tmp_path, [[entity(1)]], 1).export_rows()
    assert "vector" not in rows[0]
    assert rows[0]["topic"] == "topic"


@pytest.mark.parametrize(
    "pages,row_count",
    [
        pytest.param([[entity(1)]], 2, id="one-of-two"),
        pytest.param([], 5, id="none-of-five"),
        pytest.param([[entity(1), entity(1)]], 2, id="duplicates-hiding-a-short-read"),
    ],
)
def test_a_short_export_raises(tmp_path, pages, row_count):
    """A sync reads a missing memory as a deletion and propagates it, so a partial
    export has to fail rather than be reported as complete."""
    with pytest.raises(RuntimeError, match="partial export"):
        store_with(tmp_path, pages, row_count).export_rows()


def test_export_closes_the_iterator(tmp_path):
    store = store_with(tmp_path, [[entity(1)]], 1)
    store.export_rows()
    assert store._client.iterator.closed
