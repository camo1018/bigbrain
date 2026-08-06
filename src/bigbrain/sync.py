"""Reconcile two bigbrain stores that have been accumulating memories separately.

Running a store on more than one machine means both sides gain memories the other
has never seen, and neither is complete. This module converges them.

The reconciliation is three-way: each side is compared not just against the other
but against a snapshot of the last successful sync. Without that snapshot, a
memory missing from one side is ambiguous -- it could be newly created on the
other, or deliberately deleted here -- and a two-way merge has to guess, which
means it either resurrects deletions forever or loses new memories. The snapshot
resolves the ambiguity, and it is what lets deletions propagate at all.

Identity is the memory id. That works because ids are random 63-bit integers, so
a store copied to a second machine keeps its ids, and memories created
independently on each side will not collide. Ids are therefore a durable shared
identity that needs no coordination between peers.

Topic vectors are not sent over the wire. They are derived from the topic, so the
receiving side recomputes them, which keeps every vector consistent with that
store's own embedding model and keeps the payload small and readable.
"""

from __future__ import annotations

import gzip
import json
import socket
import subprocess
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from .config import Config

WIRE_VERSION = 1

Direction = Literal["both", "push", "pull"]

# Knowledge-bearing fields, and the only ones an edit can be detected in.
# access_count and last_accessed are usage telemetry that recall() bumps on every
# read, so comparing them would leave both sides looking permanently edited;
# created_at never changes after insert.
CONTENT_FIELDS = ("topic", "content", "tags", "source", "importance")

# Floats make the round trip through a Milvus FLOAT column and JSON, so compare
# importance at a tolerance rather than exactly.
_IMPORTANCE_PRECISION = 6

# Resolved by the peer's login shell, so a plain name works when bigbrain is on the
# remote PATH. Point --peer-cmd at an absolute path (e.g. ~/src/bigbrain/.venv/bin/bigbrain)
# when it is not: ssh runs a non-interactive shell, which often has a leaner PATH.
DEFAULT_PEER_COMMAND = "bigbrain"

_GZIP_MAGIC = b"\x1f\x8b"


class SyncError(RuntimeError):
    """A sync could not be completed. Neither store has been left inconsistent."""


# --------------------------------------------------------------------- planning


@dataclass
class Decision:
    """What should happen to one memory id, before direction filtering."""

    action: Literal["converged", "push", "pull", "delete_local", "delete_peer"]
    row: dict | None = None
    updated_at: int | None = None
    conflict: str | None = None
    note: str | None = None


@dataclass
class Plan:
    """The writes needed to converge both sides."""

    upsert_local: list[dict] = field(default_factory=list)
    upsert_peer: list[dict] = field(default_factory=list)
    delete_local: list[int] = field(default_factory=list)
    delete_peer: list[int] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    deferred: int = 0
    next_snapshot: dict[int, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.upsert_local or self.upsert_peer or self.delete_local or self.delete_peer
        )

    def counts(self) -> dict[str, int]:
        return {
            "to_local": len(self.upsert_local),
            "to_peer": len(self.upsert_peer),
            "delete_local": len(self.delete_local),
            "delete_peer": len(self.delete_peer),
            "conflicts": len(self.conflicts),
            "deferred": self.deferred,
        }


def _content_key(row: dict) -> tuple:
    parts: list[Any] = []
    for name in CONTENT_FIELDS:
        value = row.get(name)
        if name == "tags":
            parts.append(tuple(sorted(value or [])))
        elif name == "importance":
            parts.append(round(float(value or 0.0), _IMPORTANCE_PRECISION))
        else:
            parts.append(value or "")
    return tuple(parts)


def _merge_telemetry(winner: dict, loser: dict) -> dict:
    """Carry usage counters across from the losing copy.

    The winner decides the knowledge, but reads that happened on the other side
    are still real reads, and the earliest creation time is still the truth.
    """
    row = dict(winner)
    born = [int(r.get("created_at") or 0) for r in (winner, loser)]
    known = [b for b in born if b > 0]
    row["created_at"] = min(known) if known else 0
    row["access_count"] = max(
        int(winner.get("access_count") or 0), int(loser.get("access_count") or 0)
    )
    row["last_accessed"] = max(
        int(winner.get("last_accessed") or 0), int(loser.get("last_accessed") or 0)
    )
    return row


def decide(mid: int, local: dict | None, peer: dict | None, seen: int | None) -> Decision | None:
    """Resolve one memory id. `seen` is its updated_at at the last sync, if any."""
    if local and peer:
        l_at, p_at = int(local["updated_at"]), int(peer["updated_at"])
        if _content_key(local) == _content_key(peer):
            return Decision(action="converged", updated_at=max(l_at, p_at))

        local_edited = seen is None or l_at > seen
        peer_edited = seen is None or p_at > seen

        if local_edited and not peer_edited:
            row = _merge_telemetry(local, peer)
            return Decision(action="push", row=row, updated_at=l_at)
        if peer_edited and not local_edited:
            row = _merge_telemetry(peer, local)
            return Decision(action="pull", row=row, updated_at=p_at)

        # Edited on both sides since the last sync, so there is no safe automatic
        # answer. Newest wins because that is the least surprising rule, but say
        # so loudly -- the other version is about to be overwritten.
        if l_at >= p_at:
            row = _merge_telemetry(local, peer)
            return Decision(
                action="push",
                row=row,
                updated_at=l_at,
                conflict=(
                    f"id={mid} {local['topic']!r} was edited on both sides; "
                    f"kept the local copy (newer by {l_at - p_at}s)"
                ),
            )
        row = _merge_telemetry(peer, local)
        return Decision(
            action="pull",
            row=row,
            updated_at=p_at,
            conflict=(
                f"id={mid} {peer['topic']!r} was edited on both sides; "
                f"kept the peer copy (newer by {p_at - l_at}s)"
            ),
        )

    if local:
        at = int(local["updated_at"])
        if seen is None:
            return Decision(action="push", row=dict(local), updated_at=at)
        if at > seen:
            # The peer dropped it, but this side has since edited it. An edit is a
            # stronger signal of intent than a stale delete, so keep it and say so.
            return Decision(
                action="push",
                row=dict(local),
                updated_at=at,
                note=(
                    f"id={mid} {local['topic']!r} was deleted on the peer but edited "
                    "here afterwards; restoring it to the peer"
                ),
            )
        return Decision(action="delete_local", updated_at=at)

    if peer:
        at = int(peer["updated_at"])
        if seen is None:
            return Decision(action="pull", row=dict(peer), updated_at=at)
        if at > seen:
            return Decision(
                action="pull",
                row=dict(peer),
                updated_at=at,
                note=(
                    f"id={mid} {peer['topic']!r} was deleted here but edited on the "
                    "peer afterwards; restoring it locally"
                ),
            )
        return Decision(action="delete_peer", updated_at=at)

    # Only in the snapshot, so it is gone from both sides. Stop tracking it.
    return None


def build_plan(
    local_rows: Iterable[dict],
    peer_rows: Iterable[dict],
    snapshot: dict[int, int] | None = None,
    *,
    direction: Direction = "both",
) -> Plan:
    """Work out the writes that would converge the two sides."""
    local = {int(r["id"]): r for r in local_rows}
    peer = {int(r["id"]): r for r in peer_rows}
    snapshot = {int(k): int(v) for k, v in (snapshot or {}).items()}

    plan = Plan()
    for mid in sorted(set(local) | set(peer) | set(snapshot)):
        decision = decide(mid, local.get(mid), peer.get(mid), snapshot.get(mid))
        if decision is None:
            continue

        writes_local = decision.action in ("pull", "delete_local")
        if direction == "push" and writes_local:
            plan.deferred += 1
        elif direction == "pull" and not writes_local and decision.action != "converged":
            plan.deferred += 1
        else:
            _record(plan, mid, decision)
            continue

        # Skipped by the direction filter, so the id is still divergent. Keep the
        # old snapshot entry rather than inventing one: claiming it converged
        # would make the next sync misread the difference as a fresh edit.
        if mid in snapshot:
            plan.next_snapshot[mid] = snapshot[mid]

    return plan


def _record(plan: Plan, mid: int, decision: Decision) -> None:
    if decision.conflict:
        plan.conflicts.append(decision.conflict)
    if decision.note:
        plan.notes.append(decision.note)

    if decision.action == "push":
        plan.upsert_peer.append(decision.row or {})
    elif decision.action == "pull":
        plan.upsert_local.append(decision.row or {})
    elif decision.action == "delete_local":
        plan.delete_local.append(mid)
        return
    elif decision.action == "delete_peer":
        plan.delete_peer.append(mid)
        return

    # Present on both sides once applied, so record the version they now share.
    plan.next_snapshot[mid] = int(decision.updated_at or 0)


# ------------------------------------------------------------------ wire format


def encode_payload(payload: dict) -> bytes:
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode())


def decode_payload(raw: bytes) -> dict:
    """Decode a gzipped JSON payload out of a possibly noisy stream.

    The payload is located by its gzip magic rather than assumed to start at byte
    zero: ssh wrappers are fond of printing banners and upgrade notices, and one
    stray line would otherwise break every sync.
    """
    start = raw.find(_GZIP_MAGIC)
    if start < 0:
        preview = raw[:400].decode("utf-8", "replace").strip()
        raise SyncError(f"no payload found in peer response. Got: {preview or '(nothing)'}")
    try:
        # A decompressobj rather than gzip.decompress, because it stops cleanly at the
        # end of the stream and leaves anything printed afterwards in unused_data,
        # where gzip.decompress would treat those trailing bytes as corruption.
        obj = zlib.decompressobj(16 + zlib.MAX_WBITS)
        body = obj.decompress(raw[start:]) + obj.flush()
        return json.loads(body)
    except (OSError, zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SyncError(f"peer payload is corrupt: {exc}") from exc


def make_dump(rows: list[dict], config: Config) -> dict:
    return {
        "version": WIRE_VERSION,
        "kind": "dump",
        "host": socket.gethostname(),
        "generated_at": int(time.time()),
        "embed_model": config.embed_model,
        "collection": config.collection,
        "memories": rows,
    }


def make_patch(upsert: list[dict], delete: list[int]) -> dict:
    return {
        "version": WIRE_VERSION,
        "kind": "patch",
        "upsert": upsert,
        "delete": [int(i) for i in delete],
    }


def _check(payload: dict, kind: str) -> dict:
    if payload.get("kind") != kind:
        raise SyncError(f"expected a {kind} payload, got {payload.get('kind')!r}")
    version = int(payload.get("version", 0))
    if version != WIRE_VERSION:
        raise SyncError(
            f"peer speaks wire version {version}, this side speaks {WIRE_VERSION}. "
            "Update bigbrain on the older side."
        )
    return payload


# ------------------------------------------------------------------------- peer


@dataclass
class Peer:
    """A bigbrain store on another host, reachable over ssh."""

    host: str
    command: str = DEFAULT_PEER_COMMAND
    key: str | None = None

    @property
    def snapshot_key(self) -> str:
        return self.key or self.host

    def _run(self, args: list[str], payload: bytes | None = None) -> bytes:
        # ssh passes the command through the remote login shell, so `~` in the
        # configured path expands there.
        cmd = ["ssh", self.host, f"{self.command} {' '.join(args)}"]
        try:
            proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=300)
        except FileNotFoundError as exc:
            raise SyncError("ssh is not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise SyncError(f"peer {self.host} timed out after 300s") from exc
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            tail = detail[-1] if detail else f"exit {proc.returncode}"
            raise SyncError(f"peer {self.host} failed: {tail}")
        return proc.stdout

    def dump(self) -> dict:
        return _check(decode_payload(self._run(["sync-dump"])), "dump")

    def apply(self, patch: dict) -> dict:
        raw = self._run(["sync-apply"], payload=encode_payload(patch))
        for line in reversed(raw.decode("utf-8", "replace").strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        raise SyncError(f"peer {self.host} returned no apply summary")


# ------------------------------------------------------------------ sync state


def state_path(config: Config) -> Path:
    return config.home / "sync-state.json"


def load_snapshot(config: Config, key: str) -> dict[int, int]:
    path = state_path(config)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    entry = (data.get("peers") or {}).get(key) or {}
    return {int(k): int(v) for k, v in (entry.get("ids") or {}).items()}


def save_snapshot(config: Config, key: str, ids: dict[int, int]) -> Path:
    path = state_path(config)
    data: dict = {"version": WIRE_VERSION, "peers": {}}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if isinstance(existing, dict):
                data = {**data, **existing}
                data.setdefault("peers", {})
        except json.JSONDecodeError:
            pass
    data["peers"][key] = {
        "synced_at": int(time.time()),
        "ids": {str(k): int(v) for k, v in ids.items()},
    }
    config.ensure_home()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------- driver


@dataclass
class Report:
    plan: Plan
    local_count: int
    peer_count: int
    peer_host: str
    dry_run: bool
    applied_local: int = 0
    applied_peer: int = 0
    deleted_local: int = 0
    deleted_peer: int = 0


def run_sync(
    store_factory,
    peer: Peer,
    *,
    direction: Direction = "both",
    dry_run: bool = False,
) -> Report:
    """Converge the local store with `peer`.

    `store_factory` is called to build a MemoryStore each time one is needed, and
    the store is closed in between: Milvus Lite takes an exclusive lock on its data
    directory, so holding it open across the network calls would block the local
    server for the whole sync.
    """
    config = Config.from_env()

    store = store_factory()
    try:
        local_rows = store.export_rows()
    finally:
        store.close()

    peer_dump = peer.dump()
    peer_rows = peer_dump.get("memories") or []

    peer_model = peer_dump.get("embed_model")
    if peer_model and peer_model != config.embed_model:
        raise SyncError(
            f"peer embeds with {peer_model!r} but this side uses {config.embed_model!r}. "
            "Vectors from different models are not comparable; align them before syncing."
        )

    snapshot = load_snapshot(config, peer.snapshot_key)
    plan = build_plan(local_rows, peer_rows, snapshot, direction=direction)

    report = Report(
        plan=plan,
        local_count=len(local_rows),
        peer_count=len(peer_rows),
        peer_host=peer.host,
        dry_run=dry_run,
    )
    if dry_run or plan.is_empty:
        return report

    # The peer goes first so that a network failure leaves the local store
    # untouched. Either way the snapshot is only written after both sides succeed,
    # so a partial sync is simply redone on the next run rather than losing data.
    if plan.upsert_peer or plan.delete_peer:
        summary = peer.apply(make_patch(plan.upsert_peer, plan.delete_peer))
        report.applied_peer = int(summary.get("upserted", 0))
        report.deleted_peer = int(summary.get("deleted", 0))

    if plan.upsert_local or plan.delete_local:
        store = store_factory()
        try:
            report.applied_local = store.import_rows(plan.upsert_local)
            report.deleted_local = store.delete(plan.delete_local) if plan.delete_local else 0
        finally:
            store.close()

    save_snapshot(config, peer.snapshot_key, plan.next_snapshot)
    return report


def apply_patch(store, payload: dict) -> dict:
    """Apply a patch payload to a store. Used by the `sync-apply` entry point."""
    _check(payload, "patch")
    upserted = store.import_rows(payload.get("upsert") or [])
    ids = [int(i) for i in (payload.get("delete") or [])]
    deleted = store.delete(ids) if ids else 0
    return {"upserted": upserted, "deleted": deleted}
