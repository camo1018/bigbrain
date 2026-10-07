#!/usr/bin/env python3
"""Automated Tier-1 compaction for the bigbrain-trim skill.

Rewrites append-churned memories to a single current-truth entry with an LLM
(through headless `pi -p`), validates each rewrite, and applies only the ones
that pass. Writes keep the exact topic (on_conflict="replace"), tags,
importance, and source, so ids and recall keys do not move.

Safety rails:
  - Originals are saved to <workdir>/orig/ and a combined backup JSON before any write.
  - A rewrite is applied only if it is smaller (<= --max-ratio of the original),
    not suspiciously small (>= --min-ratio), has no preamble, and keeps at least
    --retention of the original's distinctive identifiers (paths, dotted names,
    snake_case / CONSTANT names, ticket ids, PR numbers, backticked code).
  - A rewrite under the retention bar is retried once, telling the model exactly
    which identifiers it dropped.
  - Right before writing, the memory is re-read; if its content changed since it
    was loaded (e.g. the maintenance hook updated it), it is skipped.
  - Writes are serialized. --dry-run validates and saves rewrites without writing.

Usage:
  python compact.py [--model provider/id] [--pi-arg ARG ...] [--workers 6]
                    [--min-chars 3000] [--ids 1,2,3] [--limit N] [--dry-run]

Run audit.py first and review the plan with the user: this rewrites memories in
place. Each run writes <workdir>/log.jsonl (one line per memory with status,
sizes and retention) and prints a summary.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import churn_markers, resolve_bin  # noqa: E402

PROMPT = """You are compacting one entry of a long-term engineering memory store. Rewrite it as a single self-contained CURRENT-TRUTH document.

The entry grew by appending dated addenda, corrections, status updates and supersessions. Your job:
- Fold every correction into the fact it corrects: when a later section supersedes or corrects an earlier claim, keep ONLY the corrected version. Never state both.
- KEEP every fact that is still true: current state, decisions with their rationale (why X over Y), gotchas and their fixes, how-to steps and commands, measured numbers (with their date when the date matters), and every identifier: file paths, function/class/constant names, config keys, metric names and labels, PR numbers, commits, ticket ids, URLs, queue ids, people.
- DROP: superseded intermediate states, duplicated restatements, session narration ("I found", "this session", "asked by"), apologies, and process chatter.
- When unsure whether something is still true, KEEP it.
- Prefer compact structure: short sections with === HEADERS ===, dense bullet lines. No markdown code fences around the whole output.
- Do not invent anything. Do not add commentary about the rewrite. Do not repeat the topic line.
- Aim for roughly a third of the original length, but never drop a live fact to hit a size.

Output ONLY the rewritten entry text, nothing before or after it.

TOPIC: {topic}

ENTRY CONTENT:
<<<
{content}
>>>
"""

RETRY_NOTE = (
    "\nA previous rewrite dropped these identifiers/recipes from the original. Re-include every one "
    "that is still true and useful (exact spelling), dropping only ones that were superseded:\n{missing}\n"
)

_TOKEN_RES = [
    re.compile(r"`([^`\n]{3,120})`"),
    re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"),
    re.compile(r"#\d{2,}\b"),
    re.compile(r"\b[\w.-]*/[\w./{}<>*-]{3,}"),
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b"),
    re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+){1,}\b"),
    re.compile(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w.]*\b"),
]
_PREAMBLE_RE = re.compile(r"^(here|sure|below|okay|i\b)", re.IGNORECASE)
_ECHO_RE = re.compile(r"^\s*(?:TOPIC:[^\n]*\n+|ENTRY CONTENT:\s*\n+|<<<\s*\n)")


def identifiers(text: str) -> set[str]:
    """Distinctive tokens a faithful rewrite should keep."""
    out: set[str] = set()
    for rx in _TOKEN_RES:
        for m in rx.finditer(text):
            t = (m.group(1) if m.groups() else m.group(0)).strip()
            if len(t) >= 4:
                out.add(t)
    return out


def retention(old_ids: set[str], new: str) -> float:
    return sum(1 for t in old_ids if t in new) / len(old_ids) if old_ids else 1.0


def clean_output(text: str) -> str:
    out = text.strip()
    out = re.sub(r"^```[a-zA-Z]*\n", "", out)
    out = re.sub(r"\n```\s*$", "", out)
    while True:
        stripped = _ECHO_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped
    if out.rstrip().endswith(">>>"):
        out = out.rstrip()[:-3]
    return out.strip()


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bin = resolve_bin(args.bin)
        self.workdir = Path(args.workdir)
        (self.workdir / "orig").mkdir(parents=True, exist_ok=True)
        (self.workdir / "new").mkdir(parents=True, exist_ok=True)
        self.log_path = self.workdir / "log.jsonl"
        self.write_lock = threading.Lock()
        self.log_lock = threading.Lock()

    # -- store access -----------------------------------------------------

    def load_all(self) -> dict[str, dict]:
        out = subprocess.run(
            [*self.bin, "list", "--json", "--limit", "100000"],
            capture_output=True, text=True, check=True,
        ).stdout
        data = json.loads(out)
        data = data if isinstance(data, list) else data.get("memories", data)
        return {str(m["id"]): m for m in data}

    def write(self, mem: dict, content: str) -> bool:
        cmd = [*self.bin, "store", mem["topic"], content, "--on-conflict", "replace",
               "--importance", str(round(float(mem.get("importance", 0.5)), 2))]
        if mem.get("tags"):
            cmd += ["--tags", ",".join(mem["tags"])]
        if mem.get("source"):
            cmd += ["--source", mem["source"]]
        subprocess.run(cmd, capture_output=True, text=True)
        after = self.load_all().get(str(mem["id"]))
        return after is not None and after["content"].strip() == content.strip()

    # -- llm ----------------------------------------------------------------

    def rewrite(self, mem: dict, missing: list[str] | None = None) -> str:
        prompt = PROMPT.format(topic=mem["topic"], content=mem["content"])
        if missing:
            prompt += RETRY_NOTE.format(missing="\n".join(missing))
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            path = f.name
        cmd = ["pi", "-p", "--no-session", "--no-tools", "--no-extensions", "--no-skills",
               "--no-context-files", "--thinking", "low", *self.args.pi_arg]
        if self.args.model:
            cmd += ["--model", self.args.model]
        cmd += [f"@{path}", "Follow the instructions in the attached file."]
        try:
            last = ""
            for attempt in range(3):
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=tempfile.gettempdir())
                if r.returncode == 0 and r.stdout.strip():
                    return clean_output(r.stdout)
                last = (r.stderr or r.stdout)[-300:]
                time.sleep(3 * (attempt + 1))
            raise RuntimeError(f"pi failed: {last}")
        finally:
            os.unlink(path)

    # -- pipeline -----------------------------------------------------------

    def log(self, rec: dict) -> None:
        with self.log_lock, open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def verdict(self, old: str, new: str, ret: float) -> str:
        a = self.args
        if len(new) > a.max_ratio * len(old):
            return "skip_not_smaller"
        if len(new) < a.min_ratio * len(old):
            return "skip_too_small"
        if ret < a.retention:
            return "skip_low_retention"
        if _PREAMBLE_RE.match(new):
            return "skip_preamble"
        return "ready"

    def process(self, mem: dict) -> dict:
        mid, old = str(mem["id"]), mem["content"]
        rec = {"id": mid, "topic": mem["topic"][:100], "old": len(old)}
        old_ids = identifiers(old)
        try:
            new = self.rewrite(mem)
            ret = retention(old_ids, new)
            if ret < self.args.retention and len(new) <= self.args.max_ratio * len(old):
                missing = sorted(t for t in old_ids if t not in new)[:120]
                retry = self.rewrite(mem, missing)
                if retention(old_ids, retry) > ret:
                    new, ret = retry, retention(old_ids, retry)
                    rec["retried"] = True
        except Exception as exc:  # noqa: BLE001
            rec.update(status="llm_error", detail=str(exc)[:300])
            self.log(rec)
            return rec
        (self.workdir / "new" / f"{mid}.txt").write_text(new)
        rec.update(new=len(new), retention=round(ret, 3), identifiers=len(old_ids))
        rec["status"] = self.verdict(old, new, ret)
        if rec["status"] == "skip_low_retention":
            rec["missing_sample"] = sorted(t for t in old_ids if t not in new)[:25]
        if rec["status"] == "ready" and not self.args.dry_run:
            with self.write_lock:
                current = self.load_all().get(mid)
                if current is None:
                    rec["status"] = "skip_deleted"
                elif current["content"] != old or current.get("updated_at") != mem.get("updated_at"):
                    rec["status"] = "skip_changed_meanwhile"
                else:
                    rec["status"] = "applied" if self.write(mem, new) else "write_failed"
        elif rec["status"] == "ready":
            rec["status"] = "dry_run_ready"
        self.log(rec)
        return rec

    def candidates(self, mems: dict[str, dict]) -> list[dict]:
        a = self.args
        if a.ids:
            picked = [mems[i] for i in a.ids.split(",") if i in mems]
        else:
            picked = [m for m in mems.values()
                      if len(m["content"]) >= a.min_chars and churn_markers(m["content"]) >= a.min_markers]
            picked.sort(key=lambda m: -len(m["content"]))
        if a.limit:
            picked = picked[: a.limit]
        return picked

    def run(self) -> None:
        mems = self.load_all()
        todo = self.candidates(mems)
        for m in todo:
            (self.workdir / "orig" / f"{m['id']}.json").write_text(json.dumps(m))
        backup = self.workdir / "originals.json"
        backup.write_text(json.dumps(todo))
        print(f"{len(todo)} candidates; originals saved to {backup}", flush=True)
        results: list[dict] = []
        with cf.ThreadPoolExecutor(self.args.workers) as ex:
            futures = [ex.submit(self.process, m) for m in todo]
            for i, fut in enumerate(cf.as_completed(futures), 1):
                results.append(fut.result())
                if i % 10 == 0 or i == len(todo):
                    print(f"{i}/{len(todo)} done", flush=True)
        counts: dict[str, int] = {}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        saved = sum(r["old"] - r["new"] for r in results if r["status"] in ("applied", "dry_run_ready"))
        print(json.dumps({"statuses": counts, "chars_saved": saved, "log": str(self.log_path)}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", help="Path to the bigbrain executable (see audit.py for resolution).")
    p.add_argument("--model", default=os.environ.get("BIGBRAIN_TRIM_MODEL"),
                   help="pi model pattern (provider/id). Prefer a strong model; small models drop facts.")
    p.add_argument("--pi-arg", action="append", default=[],
                   help="Extra argument passed to pi (repeatable), e.g. --pi-arg=-e --pi-arg=/path/provider.ts")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--min-chars", type=int, default=3000)
    p.add_argument("--min-markers", type=int, default=1,
                   help="Only consider memories with at least this many churn markers (see audit.py).")
    p.add_argument("--ids", help="Comma-separated memory ids to process instead of auto-selecting.")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retention", type=float, default=0.80)
    p.add_argument("--max-ratio", type=float, default=0.85)
    p.add_argument("--min-ratio", type=float, default=0.10)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workdir", default=str(Path(tempfile.gettempdir()) / f"bigbrain-compact-{time.strftime('%Y%m%d-%H%M%S')}"))
    args = p.parse_args()
    if shutil.which("pi") is None:
        sys.exit("pi not found on PATH; compact.py drives headless `pi -p` for the rewrites")
    Runner(args).run()


if __name__ == "__main__":
    main()
