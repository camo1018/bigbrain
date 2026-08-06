"""Tests for peer reconciliation.

The planner is pure -- rows in, decisions out -- so all of this runs without Milvus
or an embedding model.
"""

from __future__ import annotations

import gzip
import json

import pytest

from bigbrain import sync as s


def row(mid: int, **overrides) -> dict:
    base = {
        "id": mid,
        "topic": f"topic-{mid}",
        "content": f"content-{mid}",
        "tags": [],
        "source": "",
        "importance": 0.5,
        "created_at": 50,
        "updated_at": 100,
        "access_count": 0,
        "last_accessed": 0,
    }
    base.update(overrides)
    return base


def moved(plan: s.Plan) -> dict:
    return {
        "to_peer": {int(r["id"]) for r in plan.upsert_peer},
        "to_local": {int(r["id"]) for r in plan.upsert_local},
        "del_peer": set(plan.delete_peer),
        "del_local": set(plan.delete_local),
    }


def expect(**kw) -> dict:
    out = {"to_peer": set(), "to_local": set(), "del_peer": set(), "del_local": set()}
    out.update(kw)
    return out


# Each case is (local rows, peer rows, last-sync snapshot, expected movement).
SCENARIOS = [
    pytest.param([row(1)], [row(1)], {}, expect(), id="identical-never-synced"),
    pytest.param([row(1)], [row(1)], {1: 100}, expect(), id="identical-already-synced"),
    pytest.param([row(1)], [], {}, expect(to_peer={1}), id="new-on-local"),
    pytest.param([], [row(1)], {}, expect(to_local={1}), id="new-on-peer"),
    pytest.param(
        [row(1, content="edited", updated_at=200)],
        [row(1)],
        {1: 100},
        expect(to_peer={1}),
        id="edited-on-local-only",
    ),
    pytest.param(
        [row(1)],
        [row(1, content="edited", updated_at=200)],
        {1: 100},
        expect(to_local={1}),
        id="edited-on-peer-only",
    ),
    pytest.param(
        [row(1, content="mine", updated_at=300)],
        [row(1, content="theirs", updated_at=200)],
        {1: 100},
        expect(to_peer={1}),
        id="edited-both-local-newer",
    ),
    pytest.param(
        [row(1, content="mine", updated_at=200)],
        [row(1, content="theirs", updated_at=300)],
        {1: 100},
        expect(to_local={1}),
        id="edited-both-peer-newer",
    ),
    pytest.param(
        [row(1)],
        [],
        {1: 100},
        expect(del_local={1}),
        id="deleted-on-peer-propagates",
    ),
    pytest.param(
        [],
        [row(1)],
        {1: 100},
        expect(del_peer={1}),
        id="deleted-on-local-propagates",
    ),
    pytest.param(
        [row(1, content="revived", updated_at=200)],
        [],
        {1: 100},
        expect(to_peer={1}),
        id="deleted-on-peer-but-edited-here-wins",
    ),
    pytest.param(
        [],
        [row(1, content="revived", updated_at=200)],
        {1: 100},
        expect(to_local={1}),
        id="deleted-here-but-edited-on-peer-wins",
    ),
    pytest.param([], [], {1: 100}, expect(), id="deleted-on-both-sides"),
    pytest.param(
        [row(1, updated_at=100)],
        [row(1, updated_at=250)],
        {1: 100},
        expect(),
        id="same-content-different-timestamps",
    ),
    pytest.param(
        [row(1, importance=0.5)],
        [row(1, importance=0.5000001)],
        {1: 100},
        expect(),
        id="importance-float-noise-is-not-an-edit",
    ),
    pytest.param(
        [row(1, tags=["b", "a"])],
        [row(1, tags=["a", "b"])],
        {1: 100},
        expect(),
        id="tag-order-is-not-an-edit",
    ),
    pytest.param(
        [row(1, access_count=9, last_accessed=999)],
        [row(1, access_count=3)],
        {1: 100},
        expect(),
        id="telemetry-alone-is-not-an-edit",
    ),
    pytest.param(
        [row(1), row(2, content="mine", updated_at=200), row(4)],
        [row(1), row(2), row(3)],
        {1: 100, 2: 100},
        expect(to_peer={2, 4}, to_local={3}),
        id="mixed-batch",
    ),
]


@pytest.mark.parametrize("local,peer,snapshot,want", SCENARIOS)
def test_build_plan_movement(local, peer, snapshot, want):
    plan = s.build_plan(local, peer, snapshot)
    assert moved(plan) == want


@pytest.mark.parametrize("local,peer,snapshot,want", SCENARIOS)
def test_no_id_is_written_in_both_directions(local, peer, snapshot, want):
    plan = s.build_plan(local, peer, snapshot)
    outbound = {int(r["id"]) for r in plan.upsert_peer} | set(plan.delete_peer)
    inbound = {int(r["id"]) for r in plan.upsert_local} | set(plan.delete_local)
    assert not (outbound & inbound)


@pytest.mark.parametrize(
    "local,peer,snapshot,conflicts,notes",
    [
        pytest.param([row(1)], [row(1)], {1: 100}, 0, 0, id="clean"),
        pytest.param(
            [row(1, content="a", updated_at=300)],
            [row(1, content="b", updated_at=200)],
            {1: 100},
            1,
            0,
            id="concurrent-edit-is-a-conflict",
        ),
        pytest.param(
            [row(1, content="a", updated_at=300)],
            [row(1, content="b", updated_at=200)],
            {},
            1,
            0,
            id="divergent-content-with-no-snapshot-is-a-conflict",
        ),
        pytest.param(
            [row(1, content="revived", updated_at=200)],
            [],
            {1: 100},
            0,
            1,
            id="edit-beating-a-delete-is-a-note",
        ),
        pytest.param([row(1)], [], {}, 0, 0, id="plain-new-memory-is-silent"),
    ],
)
def test_reporting(local, peer, snapshot, conflicts, notes):
    plan = s.build_plan(local, peer, snapshot)
    assert len(plan.conflicts) == conflicts
    assert len(plan.notes) == notes


@pytest.mark.parametrize(
    "direction,want,deferred",
    [
        pytest.param("both", expect(to_peer={1}, to_local={2}), 0, id="both"),
        pytest.param("push", expect(to_peer={1}), 1, id="push-withholds-local-writes"),
        pytest.param("pull", expect(to_local={2}), 1, id="pull-withholds-peer-writes"),
    ],
)
def test_direction_filters(direction, want, deferred):
    local = [row(1)]
    peer = [row(2)]
    plan = s.build_plan(local, peer, {}, direction=direction)
    assert moved(plan) == want
    assert plan.counts()["deferred"] == deferred


@pytest.mark.parametrize(
    "direction,want",
    [
        pytest.param("push", expect(del_peer={1}), id="push-still-propagates-its-deletions"),
        pytest.param("pull", expect(), id="pull-ignores-them"),
    ],
)
def test_direction_and_deletions(direction, want):
    plan = s.build_plan([], [row(1)], {1: 100}, direction=direction)
    assert moved(plan) == want


def test_deferred_changes_keep_their_old_snapshot_entry():
    """A withheld change must not be recorded as converged.

    Claiming it synced would make the next run read the still-present difference as
    a brand-new edit, and silently pick a winner.
    """
    plan = s.build_plan(
        [row(1, content="mine", updated_at=200)], [row(1)], {1: 100}, direction="pull"
    )
    assert plan.next_snapshot == {1: 100}
    assert moved(plan) == expect()


@pytest.mark.parametrize(
    "local,peer,snapshot,want",
    [
        pytest.param([row(1)], [row(1)], {}, {1: 100}, id="converged-records-the-version"),
        pytest.param(
            [row(1, updated_at=100)],
            [row(1, updated_at=250)],
            {},
            {1: 250},
            id="same-content-records-the-newer-stamp",
        ),
        pytest.param(
            [row(1, content="x", updated_at=200)],
            [row(1)],
            {1: 100},
            {1: 200},
            id="push-records-the-winner",
        ),
        pytest.param([], [], {1: 100}, {}, id="fully-deleted-ids-are-forgotten"),
        pytest.param([row(1)], [], {1: 100}, {}, id="propagated-deletes-are-forgotten"),
    ],
)
def test_next_snapshot(local, peer, snapshot, want):
    assert s.build_plan(local, peer, snapshot).next_snapshot == want


def test_snapshot_converges_after_a_second_run():
    """Applying a plan and re-planning must find nothing left to do."""
    local = [row(1, content="mine", updated_at=200)]
    peer = [row(1)]

    first = s.build_plan(local, peer, {1: 100})
    assert moved(first) == expect(to_peer={1})

    # The peer now holds exactly what was sent to it.
    peer_after = [dict(first.upsert_peer[0])]
    second = s.build_plan(local, peer_after, first.next_snapshot)
    assert second.is_empty


@pytest.mark.parametrize(
    "field,winner_value,loser_value,want",
    [
        pytest.param("access_count", 2, 7, 7, id="reads-on-the-losing-side-still-count"),
        pytest.param("last_accessed", 10, 90, 90, id="latest-read-wins"),
        pytest.param("created_at", 80, 20, 20, id="earliest-creation-wins"),
    ],
)
def test_telemetry_is_merged_into_the_winner(field, winner_value, loser_value, want):
    local = [row(1, content="mine", updated_at=300, **{field: winner_value})]
    peer = [row(1, content="theirs", updated_at=200, **{field: loser_value})]
    plan = s.build_plan(local, peer, {1: 100})
    assert plan.upsert_peer[0][field] == want


def test_winner_keeps_its_own_knowledge():
    local = [row(1, content="mine", topic="mine", updated_at=300)]
    peer = [row(1, content="theirs", topic="theirs", updated_at=200)]
    sent = s.build_plan(local, peer, {1: 100}).upsert_peer[0]
    assert (sent["topic"], sent["content"]) == ("mine", "mine")


# ----------------------------------------------------------------- wire format


def test_payload_round_trip():
    payload = s.make_patch([row(1)], [2, 3])
    assert s.decode_payload(s.encode_payload(payload)) == payload


@pytest.mark.parametrize(
    "noise",
    [
        pytest.param(b"", id="clean"),
        pytest.param(b"version mismatch, download v2.33.8\n", id="ssh-wrapper-banner"),
        pytest.param(b"Warning: Permanently added a host key\n" * 3, id="multiline-banner"),
    ],
)
def test_decode_tolerates_noise_before_the_payload(noise):
    """ssh wrappers print banners and notices, and one stray line must not break a sync."""
    payload = s.make_dump([row(1)], s.Config())
    assert s.decode_payload(noise + s.encode_payload(payload)) == payload


@pytest.mark.parametrize(
    "noise",
    [
        pytest.param(b"\n", id="trailing-newline"),
        pytest.param(b"Connection to host closed.\n", id="ssh-teardown-notice"),
    ],
)
def test_decode_tolerates_noise_after_the_payload(noise):
    payload = s.make_dump([row(1)], s.Config())
    assert s.decode_payload(s.encode_payload(payload) + noise) == payload


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"command not found: bigbrain", id="remote-error-text"),
        pytest.param(b'{"memories": []}', id="uncompressed-json"),
    ],
)
def test_decode_rejects_a_non_payload(raw):
    with pytest.raises(s.SyncError):
        s.decode_payload(raw)


def test_decode_rejects_corrupt_gzip():
    with pytest.raises(s.SyncError):
        s.decode_payload(gzip.compress(b"not json"))


@pytest.mark.parametrize(
    "payload,kind",
    [
        pytest.param({"kind": "patch", "version": s.WIRE_VERSION}, "dump", id="wrong-kind"),
        pytest.param({"kind": "dump", "version": 999}, "dump", id="future-wire-version"),
        pytest.param({"kind": "dump"}, "dump", id="missing-version"),
    ],
)
def test_check_rejects_incompatible_payloads(payload, kind):
    with pytest.raises(s.SyncError):
        s._check(payload, kind)


def test_dump_advertises_the_embedding_model():
    """run_sync refuses to merge stores built with different models, so the dump has
    to say which one produced it."""
    payload = s.make_dump([], s.Config())
    assert payload["embed_model"] == s.Config().embed_model
    assert json.loads(json.dumps(payload)) == payload


# ----------------------------------------------------------------- sync state


def test_snapshot_survives_a_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRAIN_HOME", str(tmp_path))
    cfg = s.Config.from_env()
    s.save_snapshot(cfg, "peer-a", {1: 100, 2: 200})
    assert s.load_snapshot(cfg, "peer-a") == {1: 100, 2: 200}


def test_snapshots_are_kept_per_peer(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRAIN_HOME", str(tmp_path))
    cfg = s.Config.from_env()
    s.save_snapshot(cfg, "peer-a", {1: 100})
    s.save_snapshot(cfg, "peer-b", {2: 200})
    assert s.load_snapshot(cfg, "peer-a") == {1: 100}
    assert s.load_snapshot(cfg, "peer-b") == {2: 200}


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param("", id="empty-file"),
        pytest.param("not json at all", id="corrupt"),
        pytest.param("[]", id="wrong-shape"),
    ],
)
def test_unreadable_state_is_treated_as_a_first_sync(tmp_path, monkeypatch, contents):
    monkeypatch.setenv("BIGBRAIN_HOME", str(tmp_path))
    cfg = s.Config.from_env()
    s.state_path(cfg).write_text(contents)
    assert s.load_snapshot(cfg, "peer-a") == {}


def test_missing_state_is_treated_as_a_first_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("BIGBRAIN_HOME", str(tmp_path))
    assert s.load_snapshot(s.Config.from_env(), "peer-a") == {}
