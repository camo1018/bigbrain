---
name: bigbrain-trim
description: >-
  Trim and reconcile the bigbrain long-term memory store: compact append-churned
  memories to current-truth, consolidate duplicate/overlapping entries, evict
  dead weight, and tidy tags. Use when the user asks to trim, prune, compact,
  clean up, reconcile, garbage-collect, or "reduce the fat" in bigbrain memory,
  or when the store has grown bloated. This is a destructive, on-demand
  maintenance pass (distinct from the per-turn memory-maintenance hook) and
  always requires explicit user approval before writing.
disable-model-invocation: true
---

# bigbrain-trim: memory store trimming

Trimming is **destructive** and needs human review. Unlike the per-turn
memory-maintenance hook (which appends learnings), this is an explicit,
occasional pass that rewrites and deletes. **Never mutate without an approved
plan.**

## Why bigbrain accumulates fat

bigbrain has store-time dedup-merge (topic embedding cosine ≥ 0.92) and
recency/importance recall ranking, but **no automatic compaction, eviction, or
cross-memory consolidation**. So two things bloat the store:

1. **Append-churn.** `on_conflict="merge"` and the maintenance hook *append*
   (`=== ADDENDUM ===`, `SUPERSEDES`, `CORRECTION`, `AS-BUILT`). Big memories
   grow monotonically as stacked, partly-obsolete corrections. (`auto`, the
   default, now stops appending past 16,000 chars and returns `needs_rewrite`,
   but entries built before that, or written with explicit `merge`, still need
   compaction. Content has a hard 65,535-byte cap.)
2. **Silent duplicates.** Same-subject memories phrased differently stay below
   the 0.92 auto-merge threshold and coexist.

## The two mechanics (know these cold)

- **Compaction = rewrite in place, no delete, no data loss.** Re-`memory_store`
  with the **exact same `topic`** and `on_conflict="replace"`. Same topic →
  overwrites the existing row and re-embeds. This is the safe workhorse.
- **Delete = use the CLI, not MCP.** bigbrain ids are int64 and can round-trip
  lossily through a JS-number MCP transport, so a numeric-id `memory_delete`/
  `memory_update` may hit the wrong row or no-op. Delete via the CLI, where ids
  are shell args parsed as ints:

```bash
"$BIGBRAIN_REPO"/.venv/bin/bigbrain delete <id> <id> ...   # or just: bigbrain delete <id> ...
```

  The MCP server releases the Milvus lock after each op, so the CLI runs fine
  alongside it. Read ids from the audit / `memory_list`; pass them to the CLI.

## Workflow

```
Trim progress:
- [ ] 1. Audit (read-only)
- [ ] 2. Build proposal
- [ ] 3. Get explicit approval  ← hard gate, do not skip
- [ ] 4. Apply (compact via MCP replace, delete via CLI)
- [ ] 5. Verify
```

### 1. Audit (read-only)

```bash
python ~/.cursor/skills/bigbrain-trim/scripts/audit.py
```

(Run it from `skills/bigbrain-trim/scripts/audit.py` in the repo if not installed. It
finds the `bigbrain` executable via `$BIGBRAIN_BIN`, `$BIGBRAIN_REPO`, or `PATH`.)
It prints a ranked worklist across four tiers plus store
stats. It only reads (`bigbrain list --json`) — it never writes. Read the full
content of any candidate you plan to change with `memory_get` before rewriting.

### 2. Build the proposal

Turn the audit into a concrete plan, in priority order:

**Tier 1 — Compaction (biggest win, do first).** For each append-churned memory,
draft a single current-truth rewrite:
- **Keep:** what is still true — current state, durable gotchas + fixes, source
  refs, PR/commit/ticket ids, file paths, ownership, rationale ("why X over Y").
- **Drop:** superseded intermediate states (`NOT yet committed` → later
  `pushed as PR #123` collapses to just the final state), duplicated
  restatements, and any chat/session scaffolding.
- **When unsure whether a fact is still true, keep it.** Compaction must not
  lose live information — only obsolete scaffolding.

**Tier 2 — Consolidation.** For each flagged group that is genuinely the *same
subject*: pick the canonical keeper (usually highest `access_count` /
`importance` / most complete), rewrite it to absorb the others' unique facts
(Tier-1 style), then delete the losers. Do **not** merge distinct-but-related
memories — over-merging hurts recall precision.

**Tier 3 — Eviction.** Treat access/age/importance as *signals, not rules* — a
0-access entry can still be valuable reference. Propose deletions only for truly
transient, fully-superseded, or no-longer-true entries.

**Tier 4 — Tag hygiene (optional).** Fold synonym/singleton tags toward whatever
vocabulary your store has settled on (e.g. `project-map`, `gotcha`,
`conventions`, `infra`, ...) *while* doing Tier-1/2 rewrites. Don't run a
separate mass re-tag pass.

### 3. Get explicit approval (hard gate)

Present the plan: which memories get compacted (with before/after token
estimate), which get consolidated into which keeper, which get deleted, and the
projected reclaimed tokens. **Stop and wait for the user to approve.** Never
delete or rewrite before sign-off. Let the user approve tiers or items
selectively.

### 4. Apply (approved items only)

Preserve the `topic` string exactly on every rewrite — it is the embedding key;
drifting it can orphan recall or spawn a near-duplicate. Work one memory at a
time.

- **Compact / rewrite keeper** (MCP):

```
memory_store(
  topic="<exact existing topic>",
  content="<rewritten current-truth>",
  tags=[<union of relevant tags>],
  importance=<preserve or raise>,
  on_conflict="replace",
  dedup=true,
)
```

- **Consolidate:** rewrite the keeper as above, then delete the losers via the
  CLI (step-2 command). Confirm the ids against the audit first.
- **Evict:** delete via the CLI.

If a `topic` itself must change (rare), that is effectively a new key: store the
new entry, then CLI-delete the old id — do not rely on `replace` to move a key.

### 5. Verify

- `memory_recall` each rewritten topic to confirm it is intact and still
  findable at the top.
- `bigbrain count` (or `memory_count`) and re-run the audit to confirm the
  store shrank and no live memory was dropped.
- Report: memories compacted / consolidated / deleted, and tokens reclaimed.

## Guardrails

- **Approval before every write.** No exceptions.
- **Current-truth, never lossy.** Compaction removes obsolete scaffolding, not
  live facts.
- **Same topic on replace.** Changing the topic is a key move, not an edit.
- **CLI for deletes.** Never trust numeric-id deletes over MCP.
- **Consolidate real duplicates only.** Keep distinct memories distinct.
- **On-demand only.** This is not the per-turn hook; run it when asked.
