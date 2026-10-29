"""Tests for full reads that avoid Milvus Lite's short/duplicated pages.

`export_rows` and `list` both read the whole collection, and both used to break
against a real store in ways a scheduler-level fake cannot show:

- `export_rows` collected the page lists and copied fields out afterwards. Once a
  later `query()` had run on the client, those stale entity dicts read `tags`
  back as None, so an export carried no tags at all and a sync pushed tag-less
  rows over the peer's good ones.
- `list` walked `query_iterator(batch_size=500)`, which on a store past ~600 rows
  returns duplicated rows and stops early (622 unique of 734 observed), so
  browsing silently hid ~15% of the store.

Both now go through `MemoryStore._read_all`: range-split `query()` pages, copied
into plain dicts before the next query runs.
"""

from __future__ import annotations

import re

import pytest

from bigbrain.config import Config
from bigbrain.store import MemoryStore

_MAX_ID = (1 << 63) - 1
_RANGE = re.compile(r"id >= (\d+) and id <= (\d+)(?: and (.+))?")
_TAG_LIKE = re.compile(r"""tags like '%"(.+?)"%'""")


def _sql_like_on_tags(row: dict, expr: str) -> bool:
    """Evaluate the tiny subset of filter expressions `_build_filter` emits."""
    match = _TAG_LIKE.fullmatch(expr.strip())
    assert match is not None, expr
    return match.group(1) in (row.get("tags") or [])


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


class FakeClient:
    def __init__(
        self,
        rows: list[dict],
        row_count: int | None = None,
        *,
        reverse: bool = False,
        count_override: int | None = None,
    ) -> None:
        self.rows = rows
        self.row_count = len(rows) if row_count is None else row_count
        self.reverse = reverse
        self.count_override = count_override
        self.queries: list[dict] = []
        # Every dict handed out by a non-count query, so a later query can wipe
        # their JSON field the way the real Milvus Lite client does.
        self._handed_out: list[dict] = []

    def query(self, **kwargs) -> list[dict]:
        self.queries.append(kwargs)
        if kwargs["output_fields"] != ["count(*)"]:
            self._invalidate_previous_pages()
        matches = self._matches(kwargs["filter"])
        if kwargs["output_fields"] == ["count(*)"]:
            # Milvus Lite's exact match count; respects the filter but not limit.
            total = len(matches) if self.count_override is None else self.count_override
            return [{"count(*)": total}]
        fields = kwargs["output_fields"]
        page = [
            {key: row[key] for key in fields if key in row}
            for row in matches[: kwargs["limit"]]
        ]
        self._handed_out.extend(page)
        return page

    def _invalidate_previous_pages(self) -> None:
        """Simulate Milvus Lite entity dicts going stale after a later query.

        Against a real store, rows returned by one `query()` read their JSON
        `tags` column back as None once another query has run on the same client.
        Copying each page into plain dicts before the next query is what makes a
        full read safe, so the fake has to reproduce the hazard.
        """
        for row in self._handed_out:
            if "tags" in row:
                row["tags"] = None
        self._handed_out.clear()

    def _matches(self, expr: str) -> list[dict]:
        if expr == "id >= 0":
            matches = list(self.rows)
        else:
            match = _RANGE.fullmatch(expr)
            assert match is not None, expr
            lower, upper = (int(value) for value in match.groups()[:2])
            extra = match.group(3)
            matches = [
                row
                for row in self.rows
                if lower <= int(row["id"]) <= upper
                and (extra is None or _sql_like_on_tags(row, extra))
            ]
        matches.sort(key=lambda row: int(row["id"]), reverse=self.reverse)
        return matches

    def range_queries(self) -> list[dict]:
        return [
            call
            for call in self.queries
            if call["output_fields"] != ["count(*)"]
        ]

    def get_collection_stats(self, _name: str) -> dict:
        return {"row_count": self.row_count}


def store_with(
    tmp_path,
    rows: list[dict],
    row_count: int | None = None,
    *,
    reverse: bool = False,
    count_override: int | None = None,
) -> MemoryStore:
    store = MemoryStore(Config(home=tmp_path))
    store._client = FakeClient(
        rows,
        row_count,
        reverse=reverse,
        count_override=count_override,
    )
    return store


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param([], id="empty-store"),
        pytest.param([1], id="one-row"),
        pytest.param([1, 2, 3], id="fewer-than-window"),
        pytest.param([1, _MAX_ID], id="int64-boundaries"),
    ],
)
def test_export_returns_each_memory_once(tmp_path, ids):
    rows = store_with(tmp_path, [entity(mid) for mid in ids]).export_rows()
    assert [int(row["id"]) for row in rows] == ids


def test_export_splits_a_full_result_window(tmp_path):
    ids = [1, 1 << 62, _MAX_ID]
    store = store_with(
        tmp_path,
        [entity(mid) for mid in ids],
        reverse=True,
    )

    rows = store.export_rows(batch=2)

    assert [int(row["id"]) for row in rows] == ids
    assert len(store._client.range_queries()) > 1
    assert all(call["limit"] == 2 for call in store._client.range_queries())


def test_export_over_default_500_row_window(tmp_path):
    step = _MAX_ID // 521
    ids = [step * index for index in range(1, 521)]
    store = store_with(tmp_path, [entity(mid) for mid in ids])

    rows = store.export_rows()

    assert [int(row["id"]) for row in rows] == ids
    assert len(store._client.range_queries()) == 3


def test_export_splits_when_count_exactly_fills_window(tmp_path):
    ids = [1, _MAX_ID]
    store = store_with(tmp_path, [entity(mid) for mid in ids])

    rows = store.export_rows(batch=2)

    assert [int(row["id"]) for row in rows] == ids
    assert len(store._client.range_queries()) == 3


def test_export_omits_the_vector(tmp_path):
    """Vectors are derived from the topic and recomputed on import, so sending them
    would only bloat the payload and risk crossing embedding models."""
    rows = store_with(tmp_path, [entity(1)]).export_rows()
    assert "vector" not in rows[0]
    assert rows[0]["topic"] == "topic"


def test_a_short_export_raises(tmp_path):
    """Missing rows look like deletions, so an incomplete export must fail."""
    store = store_with(tmp_path, [entity(1)], count_override=2)

    with pytest.raises(RuntimeError, match="partial export"):
        store.export_rows()


def test_export_preserves_tags(tmp_path):
    """Regression: pages were copied after the read, so tags came back None."""
    rows = [entity(1), entity(2)]
    rows[0]["tags"] = ["gotcha"]
    rows[1]["tags"] = ["datalake", "metrics"]
    store = store_with(tmp_path, rows)

    exported = store.export_rows(batch=1)

    assert {int(r["id"]): r["tags"] for r in exported} == {
        1: ["gotcha"],
        2: ["datalake", "metrics"],
    }


def test_list_returns_every_memory(tmp_path):
    """Regression: query_iterator lost ~15% of the store on a real database."""
    ids = [(_MAX_ID // 800) * index for index in range(1, 801)]
    store = store_with(tmp_path, [entity(mid, topic=f"topic {mid}") for mid in ids])

    listed = store.list(limit=10_000)

    assert sorted(m.id for m in listed) == ids


def test_list_ranks_by_update_recency_across_the_whole_store(tmp_path):
    """A page-local sort would only order the slice it happened to read."""
    rows = [entity(mid, topic=f"topic {mid}") for mid in range(1, 601)]
    for index, row in enumerate(rows):
        row["updated_at"] = 1_000 - index
    store = store_with(tmp_path, rows)

    listed = store.list(limit=5)

    assert [m.id for m in listed] == [1, 2, 3, 4, 5]


def test_list_respects_a_tag_filter_across_the_whole_store(tmp_path):
    """The range split has to carry the caller's filter into every sub-query.

    A split that re-queries with only the id range would happily return rows the
    filter excludes, so a tag lookup would come back wrong as soon as the store
    grows past one page.
    """
    rows = [entity(mid, topic=f"topic {mid}") for mid in range(1, 601)]
    for index, row in enumerate(rows):
        row["tags"] = ["gotcha"] if index % 100 == 0 else ["other"]
    store = store_with(tmp_path, rows)

    tagged = store.list(limit=10_000, tags=["gotcha"])

    assert [m.id for m in tagged] == [1, 101, 201, 301, 401, 501]
    assert all("gotcha" in m.tags for m in tagged)


@pytest.mark.parametrize("batch", [0, -1])
def test_export_rejects_non_positive_batch(tmp_path, batch):
    store = store_with(tmp_path, [])

    with pytest.raises(ValueError, match="batch must be at least 1"):
        store.export_rows(batch=batch)
