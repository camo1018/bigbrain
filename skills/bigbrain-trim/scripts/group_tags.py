#!/usr/bin/env python3
"""Report group/provenance tag coverage, and flag tags whose group is gone.

A host may tag memories with a group/provenance tag (`<prefix><group-id>`) and
inject group-scoped context by filtering recall on exactly that tag. Trimming
rewrites and deletes memories, so it is the operation most likely to cause one
of those tags to be lost — and a lost tag fails silently: the memory simply
stops appearing in that group, with no error on either side.

This script is READ-ONLY. Run it before and after a trim:

    python group_tags.py --snapshot before.json
    ...trim...
    python group_tags.py --baseline before.json     # exits 1 on any loss

Default tag prefix is `pi-group:`; override with --prefix. Group ids are
resolved against a registry JSON whose `groups[]` entries carry `id` (and
optionally `name`) when you have one; without --groups-json, every tag is
reported without a live/dead verdict.

Two independent checks:
  1. census delta - every group present before must still be present, with a
     count no lower than before. A SHRINK or disappearance means someone lost
     data. Groups normally only gain memories.
  2. live-group check - a tag whose id is not in the registry is dead
     provenance from a deleted group. Those are the only ones safe to drop.

`--baseline` treats a shrink as failure rather than merging counts, because
consolidation should move a tag onto the keeper, not delete it. Legit
consolidation of two memories in the SAME group reduces that group's count by
one; pass --allow-consolidation N to tolerate N such reductions, and read the
per-group diff it prints rather than trusting the exit code alone.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import load_memories, resolve_bin  # noqa: E402

DEFAULT_PREFIX = "pi-group:"


def census(mems: list[dict], prefix: str) -> dict[str, dict]:
    """group tag -> {count, memory ids}, in one pass over the store."""
    out: dict[str, dict] = collections.defaultdict(lambda: {"count": 0, "ids": []})
    for m in mems:
        for tag in m.get("tags") or []:
            if tag.startswith(prefix):
                out[tag]["count"] += 1
                out[tag]["ids"].append(str(m["id"]))
    return dict(out)


def known_groups(path: Path) -> dict[str, str]:
    """group id -> name, from a registry JSON with groups[].id."""
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    groups = data.get("groups", data) if isinstance(data, dict) else data
    if not isinstance(groups, list):
        return {}
    return {g["id"]: g.get("name", "") for g in groups if isinstance(g, dict) and "id" in g}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--bin", help="Path to the bigbrain executable.")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help=f"Group tag prefix to census (default: {DEFAULT_PREFIX}).")
    p.add_argument("--groups-json", metavar="PATH",
                   help="Registry JSON with groups[].id to resolve live group ids.")
    p.add_argument("--snapshot", metavar="PATH",
                   help="Write the census to PATH for a later --baseline comparison.")
    p.add_argument("--baseline", metavar="PATH",
                   help="Compare against a snapshot; exit 1 if any group shrank or vanished.")
    p.add_argument("--allow-consolidation", type=int, default=0, metavar="N",
                   help="Tolerate up to N per-group count reductions (same-group merges).")
    args = p.parse_args()

    cur = census(load_memories(resolve_bin(args.bin), 100000), args.prefix)
    groups = known_groups(Path(args.groups_json).expanduser()) if args.groups_json else {}

    if args.snapshot:
        Path(args.snapshot).write_text(json.dumps(
            {k: {"count": v["count"], "ids": v["ids"]} for k, v in cur.items()},
            indent=1))
        print(f"census written to {args.snapshot} ({sum(v['count'] for v in cur.values())} "
              f"applications across {len(cur)} groups)")

    print(f"{args.prefix} tags: {sum(v['count'] for v in cur.values())} applications "
          f"across {len(cur)} groups")
    dead = []
    for tag in sorted(cur):
        gid = tag[len(args.prefix):]
        if groups:
            name = groups.get(gid)
            if name is None:
                dead.append(tag)
                status = "ORPHAN (group id not in registry)"
            else:
                status = f"live: {name or gid}"
        else:
            status = ""
        print(f"  {cur[tag]['count']:4d}  {tag}  {status}".rstrip())

    if dead:
        print("\nOrphaned group tags (safe to drop; their group no longer exists):")
        for tag in dead:
            print(f"  {tag}  ids={cur[tag]['ids']}")
    elif groups:
        print("\nNo orphaned group tags: every group tag resolves to a live group.")

    if not args.baseline:
        return 0

    base = json.loads(Path(args.baseline).read_text())
    lost, shrank, changed = [], [], []
    for tag, info in base.items():
        now = cur.get(tag, {}).get("count", 0)
        if now == 0:
            lost.append((tag, info["count"]))
        elif now < info["count"] and (info["count"] - now) > args.allow_consolidation:
            shrank.append((tag, info["count"], now, info["count"] - now))
    for tag in cur:
        if tag not in base:
            changed.append((tag, 0, cur[tag]["count"], cur[tag]["count"]))

    print("\n=== census delta vs baseline ===")
    if lost:
        print(f"LOST GROUP TAGS ({len(lost)}) — every memory in these groups "
              f"silently left the group's context:")
        for tag, was in lost:
            print(f"  {tag}  had {was}, now 0  ids_before={base[tag]['ids']}")
    if shrank:
        print(f"SHRUNK beyond --allow-consolidation={args.allow_consolidation} ({len(shrank)}):")
        for tag, was, now, drop in shrank:
            still = set(cur.get(tag, {}).get("ids", []))
            missing = sorted(set(base[tag].get("ids", [])) - still)
            note = f"  ids no longer carrying it: {missing}" if missing else ""
            print(f"  {tag}  {was} -> {now} (-{drop}){note}")
    for tag, was, now, delta in changed:
        print(f"  grew/new  {tag}  {was} -> {now} (+{delta})")
    if not lost and not shrank:
        print("OK: no group lost memories.")

    return 1 if (lost or shrank) else 0


if __name__ == "__main__":
    sys.exit(main())
