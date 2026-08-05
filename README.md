# bigbrain

A hand-rolled **semantic agent-memory store** — a central, persistent knowledge base
that an AI agent (or you) can write to and recall from by meaning.

It works like human memory: a short **topic** is embedded into a vector (the fast
index / key), and the detailed **content** is the knowledge retrieved once a topic
matches. Recall blends **semantic similarity**, **recency**, and **importance** so the
most relevant, fresh, and important knowledge surfaces first.

## Design

- **Storage:** [Milvus Lite](https://milvus.io/docs/milvus_lite.md) — an embedded,
  single-file vector database (`$BIGBRAIN_HOME/memory.db`). Zero infra.
- **Embeddings:** local [`fastembed`](https://github.com/qdrant/fastembed) model
  (`BAAI/bge-small-en-v1.5`, 384-dim). No API key; offline after the one-time download.
  Vectors are unit-normalized and the store uses the inner-product metric, so the
  returned score is a cosine similarity in `[0, 1]`.
- **One core library, two front-doors:** a CLI and an MCP server both wrap the same
  `MemoryStore`.

### What "rich" scope includes

- `store` / `recall` / `get` / `list` / `update` / `delete` / `count`
- **Dedup-merge:** storing a near-identical topic (cosine ≥ `0.92`) merges into the
  existing memory instead of creating a duplicate (`on_conflict`: `merge` | `replace`
  | `skip` | `new`).
- **Recency + importance reranking:** vector candidates are reranked by
  `0.6·similarity + 0.25·recency + 0.15·importance` (recency decays with a 30-day
  half-life). All weights are configurable via env vars.
- **Tags & source** metadata, with filtering on recall/list.
- **Access tracking:** `access_count` / `last_accessed` bump on recall.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.13.

```bash
cd /path/to/bigbrain
uv sync                 # install dependencies
uv run bigbrain setup   # download + cache the embedding model (one-time)
```

> TLS-interception note: if you're behind a proxy whose CA isn't in Python's trust
> store, the one-time model download will fail verification. `bigbrain setup` falls
> back to an unverified HTTP client scoped strictly to that fetch (see
> `src/bigbrain/_bootstrap.py`); point `SSL_CERT_FILE` at the CA bundle to avoid that.
> Normal runtime does no network I/O.

## CLI usage

```bash
uv run bigbrain store "Milvus Lite setup" "Use MilvusClient(uri=path); IP metric on \
  unit-normalized vectors; FLAT index." --tags milvus,infra --importance 0.9 --source chat

uv run bigbrain recall "how do I store vectors locally?" --limit 5
uv run bigbrain recall "deploy process" --tags infra --json

uv run bigbrain list
uv run bigbrain get <id>
uv run bigbrain update <id> --importance 0.8 --tags milvus,infra,notes
uv run bigbrain delete <id> [<id> ...]
uv run bigbrain count
```

## MCP server

The server exposes these tools: `memory_store`, `memory_recall`, `memory_get`,
`memory_list`, `memory_update`, `memory_delete`, `memory_count`.

There are two ways to run it. **Prefer the shared HTTP server** (below) — it's the
durable fix for the per-workbench spawn storm described in "Why a shared server".

### Recommended: one shared HTTP server (macOS LaunchAgent)

Install once. This starts a single loopback-bound server as a LaunchAgent (starts on
login, respawns if it dies) and repoints Cursor's `mcp.json` at it by URL:

```bash
uv run bigbrain install-server            # default 127.0.0.1:8765
uv run bigbrain install-server --port 9001
```

Then reload Cursor (or toggle the server in Settings → MCP) to pick up the URL. The
resulting `~/.cursor/mcp.json` entry is just:

```json
{ "mcpServers": { "bigbrain": { "url": "http://127.0.0.1:8765/mcp" } } }
```

Manage it with `launchctl` (label `com.bigbrain.mcp`); server logs are under
`$BIGBRAIN_HOME/logs/`. Run it in the foreground for debugging with
`uv run bigbrain serve --port 8765`. To revert to stdio:

```bash
uv run bigbrain uninstall-server          # stops the agent + reverts mcp.json to stdio
```

#### Why a shared server (the problem it solves)

Cursor spawns a **separate MCP host per workbench** (each agent tab / background agent).
With the stdio config, every workbench forks its own `bigbrain-mcp` subprocess. When
many workbenches accumulate in a long-lived window and (re)create their clients at once,
they all try to spawn the same stdio server simultaneously and saturate Cursor's internal
client-creation IPC — you see a burst of `Error creating client: Timeout waiting for
EverythingProvider with command 'mcp.createClient'`, and only one workbench connects while
the rest time out. A single always-on HTTP server sidesteps this entirely: each workbench
just opens a cheap HTTP connection to the already-running server (no per-workbench
subprocess), and one process owns the DB so there's no data-dir lock contention either.

### Alternative: stdio (per-workbench subprocess)

```json
{
  "mcpServers": {
    "bigbrain": {
      "command": "uv",
      "args": ["run", "--no-sync", "--directory", "/path/to/bigbrain", "bigbrain-mcp"],
      "env": { "BIGBRAIN_HOME": "/path/to/your/.bigbrain" }
    }
  }
}
```

Reload Cursor (or toggle the server in Settings → MCP) to pick it up. The agent can then
call `memory_recall` at the start of a task and `memory_store` to persist durable
knowledge.

## Agent memory rule (make recall/store automatic)

What turns bigbrain from "a tool you invoke" into "memory that just works" is a **global
Cursor rule** that tells the agent to recall at the start of a task and store durable
learnings as it finishes — without being asked.

- **Location:** `~/.cursor/rules/bigbrain-memory.mdc` (a user-level rule, so it applies in
  **every chat and every repo**, not just this one).
- **Why `alwaysApply: true`:** the rule is loaded into every session automatically.
- **When changes take effect:** rules load at chat start, so edits apply to **new chats**.

### How to add to / edit the rule

It's a plain `.mdc` file: YAML frontmatter followed by Markdown instructions. Edit it to
change what the agent recalls/stores, add tag conventions, tune the importance scale, or
add do/don't guidance. Keep it concise (aim for < 50 lines) and actionable.

```markdown
---
description: Use the bigbrain memory store to recall and persist durable knowledge across chats
alwaysApply: true
---

# bigbrain: persistent agent memory

## Recall at the start of a task
Before non-trivial work, call `memory_recall` to check what is already known.

## Store durable learnings as you finish
After resolving something reusable, call `memory_store` with:
- `topic`: short descriptive key (becomes the search embedding)
- `content`: full, self-contained detail
- `tags`: lowercase, reusable (e.g. `infra`, `conventions`, `gotcha`)
- `importance`: 0.9–1.0 core, 0.5 default, <0.3 niche

## Don't
- Don't store secrets, credentials, or tokens.
- Don't store one-off or purely conversational context.
```

**Examples of edits you might make:**

- Add a project-specific tag convention: *"Tag anything about the billing pipeline
  with `billing`."*
- Bias what gets remembered: *"Always store database table schemas and working queries."*
- Add an exclusion: *"Never store anything from scratch/throwaway queries."*

After editing, start a new chat to pick up the changes. You can confirm what's stored with
`uv run bigbrain list` or by asking the agent to recall a topic.

## Memory-maintenance hooks (auto-run the recall/store loop)

The rule tells the agent *what* to do; **Cursor hooks** make sure it actually happens. This
repo ships a pair of [user-level hooks](https://cursor.com/docs/hooks) that trigger a
memory-maintenance pass after every **substantive** turn (any turn that used a tool — pure
conversational turns are skipped):

- `hooks/bigbrain-mark-substantive.sh` — a `postToolUse` hook that drops a per-conversation
  marker whenever a tool runs.
- `hooks/bigbrain-maintenance.sh` — a `stop` hook that, when that marker is present, injects a
  follow-up telling the agent to run the recall → decide → store/replace loop. A `loop_count`
  guard (plus `loop_limit: 1`) makes it fire at most once per message and never loop.

Because bigbrain memory is cross-repo, these install at the **user level** (`~/.cursor/`), not
per-project. Install them (and the memory rule) with:

```bash
uv run bigbrain install-hooks            # scripts + hooks.json merge + rule + skills
uv run bigbrain install-hooks --no-rule  # hooks only
uv run bigbrain install-hooks --no-skills # skip the bundled skills
```

The installer copies the scripts to `~/.cursor/hooks/`, **merges** the entries into
`~/.cursor/hooks.json` (preserving any hooks you already have, backing up to
`hooks.json.bak`), installs the rule to `~/.cursor/rules/`, and installs the bundled skills to
`~/.cursor/skills/`. It's idempotent. The hook scripts require
[`jq`](https://jqlang.github.io/jq/) on `PATH`. After installing, reload Cursor and approve the
new hooks.

## Trimming memory (`bigbrain-trim` skill)

The hook *appends* learnings, and store-time dedup only merges near-identical topics — so over
time the store accumulates append-churned entries and silent duplicates. The bundled
[`bigbrain-trim`](skills/bigbrain-trim/SKILL.md) skill is an **on-demand, destructive**
maintenance pass that compacts append-churned memories to current-truth, consolidates
duplicates, evicts dead weight, and tidies tags — always behind an explicit approval gate. It
ships a read-only [`audit.py`](skills/bigbrain-trim/scripts/audit.py) that ranks trim
candidates. Invoke it by asking the agent to "trim bigbrain memory".

## Configuration (env vars)

| Variable | Default | Meaning |
| --- | --- | --- |
| `BIGBRAIN_HOME` | `~/.bigbrain` | Data directory (holds `memory.db`) |
| `BIGBRAIN_HTTP_HOST` | `127.0.0.1` | Bind address for `bigbrain serve` (shared HTTP server) |
| `BIGBRAIN_HTTP_PORT` | `8765` | Bind port for `bigbrain serve` |
| `BIGBRAIN_COLLECTION` | `memories` | Milvus collection name |
| `BIGBRAIN_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `BIGBRAIN_EMBED_DIM` | `384` | Embedding dimension (must match model) |
| `BIGBRAIN_DEDUP_THRESHOLD` | `0.92` | Cosine sim above which topics merge |
| `BIGBRAIN_W_SIMILARITY` | `0.6` | Rerank weight: similarity |
| `BIGBRAIN_W_RECENCY` | `0.25` | Rerank weight: recency |
| `BIGBRAIN_W_IMPORTANCE` | `0.15` | Rerank weight: importance |
| `BIGBRAIN_RECENCY_HALF_LIFE_DAYS` | `30` | Recency decay half-life |
| `BIGBRAIN_CANDIDATE_MULTIPLIER` | `6` | Vector candidates fetched per recall = `limit × this` |

## Tests

```bash
uv run python tests/mcp_smoke.py   # drives the MCP server over stdio
```
