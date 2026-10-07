#!/usr/bin/env python3
"""Read-only audit of the bigbrain memory store to plan a trim.

Emits a ranked worklist of compaction / consolidation / eviction candidates so
the agent can build a concrete trim proposal. This script NEVER writes: it only
reads via `bigbrain list --json` and analyses the result in memory.

Usage:
    python audit.py [--limit N] [--json] [--bin PATH]

Binary resolution order:
    1. --bin / $BIGBRAIN_BIN
    2. <repo>/.venv/bin/bigbrain
    3. `bigbrain` on PATH
    4. `uv run --directory <repo> --no-sync bigbrain`

where <repo> is $BIGBRAIN_REPO, the path stamped in at install time, or the
checkout this script lives in.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Markers that indicate append-churn: content rewritten as stacked corrections
# rather than a single current-truth narrative.
CHURN_MARKERS = ("ADDENDUM", "SUPERSEDES", "SUPERSEDED", "CORRECTION", "AS-BUILT", "UPDATE ")
# A section header that carries a date is an appended update (e.g.
# "=== 2026-07-28: STATUS ==="); undated headers are just structure, which a
# compacted current-truth entry uses too, so they do not count as churn.
DATED_SECTION_RE = re.compile(r"^(?:===|\*\*\*).*\b\d{4}-\d{2}-\d{2}\b.*$", re.MULTILINE)
# A horizontal-rule divider between appended blocks.
DIVIDER_RE = re.compile(r"^---\s*$", re.MULTILINE)

# Tokens too generic to signal that two topics are about the same subject.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "by", "with",
    "is", "are", "vs", "via", "how", "what", "not", "no", "add", "new", "into",
    "from", "at", "as", "it", "its", "that", "this", "be", "do", "does",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# `bigbrain install-hooks` rewrites this placeholder to the checkout path when it
# copies this script outside the repo; it stays a placeholder in the repo itself.
_INSTALLED_REPO = "__BIGBRAIN_REPO__"


def repo_root() -> Path | None:
    env = os.environ.get("BIGBRAIN_REPO")
    if env:
        return Path(env).expanduser()
    if not _INSTALLED_REPO.startswith("__"):
        return Path(_INSTALLED_REPO)
    checkout = Path(__file__).resolve().parents[3]
    return checkout if (checkout / "pyproject.toml").exists() else None


def resolve_bin(explicit: str | None) -> list[str]:
    if explicit:
        return [explicit]
    env = os.environ.get("BIGBRAIN_BIN")
    if env:
        return [env]
    repo = repo_root()
    if repo and (repo / ".venv" / "bin" / "bigbrain").exists():
        return [str(repo / ".venv" / "bin" / "bigbrain")]
    which = shutil.which("bigbrain")
    if which:
        return [which]
    if repo:
        return ["uv", "run", "--directory", str(repo), "--no-sync", "bigbrain"]
    raise SystemExit(
        "error: cannot find the bigbrain executable — set $BIGBRAIN_BIN or $BIGBRAIN_REPO"
    )


def load_memories(binary: list[str], limit: int) -> list[dict]:
    out = subprocess.run(
        [*binary, "list", "--json", "--limit", str(limit)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def churn_markers(content: str) -> int:
    upper = content.upper()
    return (
        sum(upper.count(m) for m in CHURN_MARKERS)
        + len(DATED_SECTION_RE.findall(content))
        + len(DIVIDER_RE.findall(content))
    )


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def analyse(mems: list[dict]) -> dict:
    now = time.time()
    for m in mems:
        c = m.get("content", "") or ""
        m["_chars"] = len(c)
        m["_tokens_est"] = len(c) // 4
        m["_markers"] = churn_markers(c)
        m["_age_days"] = int((now - m.get("updated_at", now)) / 86400)
        m["_topic_tokens"] = tokens(m.get("topic", ""))

    tag_freq: dict[str, int] = {}
    for m in mems:
        for t in m.get("tags", []) or []:
            tag_freq[t] = tag_freq.get(t, 0) + 1

    # Tier 1: append-churn compaction. Rank by markers, then size. Size alone is
    # not churn: a large current-truth entry with no append markers is fine.
    compaction = sorted(
        (m for m in mems if m["_markers"] >= 2 or (m["_chars"] >= 12000 and m["_markers"] >= 1)),
        key=lambda m: (m["_markers"], m["_chars"]), reverse=True,
    )

    # Tier 2: consolidation. Pairs about the same subject that never auto-merged
    # (topic embeddings stayed < the 0.92 dedup threshold). Heuristic proxy:
    # high topic-token overlap and/or shared rare tags.
    rare = {t for t, f in tag_freq.items() if f <= 6}
    pairs = []
    for i in range(len(mems)):
        for j in range(i + 1, len(mems)):
            a, b = mems[i], mems[j]
            jac = jaccard(a["_topic_tokens"], b["_topic_tokens"])
            shared_rare = (set(a.get("tags", []) or []) & set(b.get("tags", []) or []) & rare)
            if jac >= 0.4 or len(shared_rare) >= 3:
                pairs.append((jac, len(shared_rare), a, b))
    pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)

    # Tier 3: eviction signals (NEVER auto-delete; these are hints for review).
    eviction = sorted(
        (m for m in mems
         if m.get("access_count", 0) == 0
         and m["_age_days"] >= 21
         and m.get("importance", 1.0) < 0.7),
        key=lambda m: (m.get("importance", 0), -m["_age_days"]),
    )

    singleton_tags = sorted(t for t, f in tag_freq.items() if f == 1)

    total_chars = sum(m["_chars"] for m in mems)
    sizes = sorted(m["_chars"] for m in mems)
    return {
        "count": len(mems),
        "total_chars": total_chars,
        "total_tokens_est": total_chars // 4,
        "median_chars": sizes[len(sizes) // 2] if sizes else 0,
        "max_chars": sizes[-1] if sizes else 0,
        "tag_count": len(tag_freq),
        "compaction": compaction,
        "consolidation": pairs,
        "eviction": eviction,
        "singleton_tags": singleton_tags,
    }


def _row(m: dict) -> str:
    return (f"  id={m['id']}  {m['_chars']:>6,}c (~{m['_tokens_est']:>5,}t)  "
            f"markers={m['_markers']:>2}  acc={m.get('access_count', 0):>3}  "
            f"age={m['_age_days']:>3}d  imp={m.get('importance', 0):.2f}  "
            f"{m.get('topic', '')[:72]}")


def print_report(a: dict) -> None:
    print("=" * 78)
    print(f"bigbrain audit: {a['count']} memories, ~{a['total_tokens_est']:,} tokens "
          f"({a['total_chars']:,} chars); median {a['median_chars']:,}c, "
          f"max {a['max_chars']:,}c; {a['tag_count']} distinct tags")
    print("=" * 78)

    print(f"\n[TIER 1] COMPACTION — rewrite to current-truth (replace in place, no delete)"
          f"  ({len(a['compaction'])})")
    for m in a["compaction"][:20]:
        print(_row(m))

    print(f"\n[TIER 2] CONSOLIDATION — likely same-subject, never auto-merged"
          f"  ({len(a['consolidation'])} pairs)")
    for jac, nrare, x, y in a["consolidation"][:20]:
        print(f"  jaccard={jac:.2f} sharedRareTags={nrare}")
        print(f"      id={x['id']} acc={x.get('access_count',0):>3} :: {x.get('topic','')[:64]}")
        print(f"      id={y['id']} acc={y.get('access_count',0):>3} :: {y.get('topic','')[:64]}")

    print(f"\n[TIER 3] EVICTION SIGNALS — review only, never auto-delete"
          f"  ({len(a['eviction'])})")
    for m in a["eviction"][:20]:
        print(_row(m))

    print(f"\n[TIER 4] TAG HYGIENE — {len(a['singleton_tags'])} singleton tags "
          f"(fold into vocabulary during rewrites, don't mass-retag)")
    if a["singleton_tags"]:
        print("  " + ", ".join(a["singleton_tags"][:40]) + (" ..." if len(a["singleton_tags"]) > 40 else ""))

    print("\nNext: build a written proposal from the above, get explicit approval, then apply.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a report.")
    p.add_argument("--bin", default=None, help="Path to the bigbrain executable.")
    args = p.parse_args()

    binary = resolve_bin(args.bin)
    try:
        mems = load_memories(binary, args.limit)
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: could not load memories via {' '.join(binary)}: {exc}", file=sys.stderr)
        raise SystemExit(1)

    a = analyse(mems)
    if args.json:
        slim = {k: v for k, v in a.items() if k not in ("compaction", "consolidation", "eviction")}
        slim["compaction"] = [{"id": m["id"], "topic": m["topic"], "chars": m["_chars"],
                               "markers": m["_markers"]} for m in a["compaction"]]
        slim["consolidation"] = [{"jaccard": round(j, 3), "shared_rare_tags": n,
                                  "a": {"id": x["id"], "topic": x["topic"]},
                                  "b": {"id": y["id"], "topic": y["topic"]}}
                                 for j, n, x, y in a["consolidation"]]
        slim["eviction"] = [{"id": m["id"], "topic": m["topic"], "age_days": m["_age_days"],
                             "importance": m.get("importance")} for m in a["eviction"]]
        print(json.dumps(slim, indent=2))
    else:
        print_report(a)


if __name__ == "__main__":
    main()
