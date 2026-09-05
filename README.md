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
- **Host-independent core, two front-doors:** a CLI and an MCP server both wrap the
  same `MemoryStore`. Bigbrain itself does not depend on Cursor, `cursor-agent`, or
  Claude Code.

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

## Quick start per host

Every host uses the same three layers: the shared HTTP server, an MCP (or bridged) client
so the agent can call `memory_*` tools, and an optional post-turn maintenance pass. Run
the server step once per machine, then the block for each agent you use.

**Server (once per machine)**

```bash
uv run bigbrain install-server        # macOS: LaunchAgent on 127.0.0.1:8765, starts on login
# Linux: see systemd/bigbrain.service.template (steps in its header)
```

**Cursor**

```bash
uv run bigbrain install-hooks         # hooks + rule + skills → ~/.cursor
cursor agent                          # installs cursor-agent, used by the maintenance pass
```

`install-server` already pointed `~/.cursor/mcp.json` at the server. Reload Cursor.

**Claude Code**

```bash
claude mcp add --scope user --transport http bigbrain http://127.0.0.1:8765/mcp
uv run bigbrain install-hooks --target claude    # hooks + rule + skills → ~/.claude
```

**Pi coding agent**

```bash
uv run bigbrain install-hooks --target pi        # extension + runner + AGENTS.md section + skills → ~/.pi/agent
```

Pi has no MCP client, so the installed extension registers the `memory_*` tools natively
and runs the maintenance pass through the direct runner, which needs `node` and a
`GEMINI_API_KEY` or `ANTHROPIC_API_KEY` (in Pi's environment or in `~/.bigbrain/env`).
Run `/reload` in an open Pi session, or start a new one.

**Verify**: start a new chat, do one turn that uses a tool, then `tail ~/.bigbrain/maintenance.log`.
A healthy line ends in `ok … :: NOOP` or `:: STORED <topic>`; a `skipped:` line names what is missing.

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

### Using bigbrain from different hosts

The memory store and its automatic maintenance are separate layers:

1. **Bigbrain service** — the host-independent store, available through the CLI or MCP.
2. **Interactive client integration** — an agent calls the MCP tools while it works.
3. **Automatic post-turn maintenance** — optional, host-specific hooks launch a headless
   agent or direct runner to decide what durable knowledge should be stored.

Any MCP-capable application can use bigbrain by connecting to the shared endpoint:

```text
http://127.0.0.1:8765/mcp
```

- **Cursor** connects via `mcp.json` (`install-server` writes the entry).
- **Claude Code** registers via `claude mcp add --scope user --transport http bigbrain http://127.0.0.1:8765/mcp`.
- **Pi coding agent** has no MCP client; `install-hooks --target pi` installs a TypeScript
  extension ([`hooks/pi/bigbrain.ts`](hooks/pi/bigbrain.ts)) that registers native
  `memory_*` tools and forwards each call to the endpoint.

Neither `cursor-agent` nor `claude` is required to run the service, use the CLI, or call
the MCP tools. For the automatic background-maintenance pass:
- **Cursor hooks** launch `cursor-agent -p`.
- **Claude Code hooks** launch `claude -p`.
- **Pi** runs the direct runner on `agent_settled`.
- **Direct runner** (`hooks/bigbrain-maintenance-direct.mjs`) is a standalone Node.js
  script that calls Gemini or Anthropic directly and drives the memory tools over HTTP. It
  is what Pi uses, and what the Cursor / Claude Code worker falls back to when the host's
  CLI is not installed.

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

### On Linux

`install-server` is macOS-only. On Linux, run the same server as a systemd user unit using
[`systemd/bigbrain.service.template`](systemd/bigbrain.service.template), which has the
install steps in its header. For Claude Code, register it once at user scope:

```bash
claude mcp add --scope user --transport http bigbrain http://127.0.0.1:8765/mcp
```

`--scope user` matters: the default is project-local, which would bind the server to
whatever directory you happened to run the command from.

## Agent memory rule (make recall consistent)

What turns bigbrain from "a tool you invoke" into "memory that just works" is a **global
agent rule** that tells the agent to recall at the start of a task. Durable writes at the
end of a turn are handled separately by the optional background-maintenance hooks below.

The single source is [`rules/bigbrain-memory.mdc`](rules/bigbrain-memory.mdc); `install-hooks`
places it where each host auto-loads user-level instructions:

- **Cursor:** `~/.cursor/rules/bigbrain-memory.mdc` (as is; `alwaysApply: true`)
- **Claude Code:** `~/.claude/rules/bigbrain-memory.md` (as is; `~/.claude/rules/*.md` loads every session)
- **Pi:** a managed section inside `~/.pi/agent/AGENTS.md`, frontmatter stripped, between
  `<!-- bigbrain-memory:begin -->` / `<!-- bigbrain-memory:end -->` markers. Anything you keep
  outside the markers is preserved on reinstall.
- All are user-level, so they apply in **every chat and every repo**, not just this one.
- **When changes take effect:** rules load at session start, so edits apply to **new chats**.

### How to add to / edit the rule

It's a plain `.mdc` file: YAML frontmatter followed by Markdown instructions. Edit it to
change what the agent recalls/stores, add tag conventions, tune the importance scale, or
add do/don't guidance. Keep it concise (aim for < 50 lines) and actionable. Re-run
`install-hooks` for each host afterwards.

```markdown
---
description: Use the bigbrain memory store to recall and persist durable knowledge across chats
alwaysApply: true
---

# bigbrain: persistent agent memory

## Recall at the start of a task
Before non-trivial work, call `memory_recall` to check what is already known.

## Writing is handled off-session
A background hook runs the maintenance pass after every substantive turn. Do not run
the same maintenance loop inline unless the user explicitly asks to remember something.

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

The rule tells the agent *what* to do; **hooks** make sure it actually happens. This repo
ships scripts that run a memory-maintenance pass after every **substantive** turn (any
turn that used a tool — pure conversational turns are skipped).

- `hooks/bigbrain-mark-substantive.sh` — a `postToolUse` / `PostToolUse` hook (Cursor / Claude)
  that drops a per-session marker whenever a tool runs.
- `hooks/bigbrain-maintenance.sh` — a `stop` / `Stop` hook (Cursor / Claude) that, when that
  marker is present, hands the pass to a detached background worker and returns immediately.
- `hooks/bigbrain-maintenance-run.sh` — the background worker for Claude Code (`claude -p`)
  and Cursor (`cursor-agent -p`). It reads the turn out of the session transcript (including
  Cursor side chats nested under a parent session) and runs a headless agent session to execute
  the recall → decide → store/replace loop over the bigbrain MCP server. If the host's CLI is
  not on `PATH` it uses whichever one is, and failing both, the direct runner.
- `hooks/bigbrain-maintenance-direct.mjs` — the standalone direct runner (Node.js ≥ 18, no
  packages). It calls the LLM API directly — Gemini with `GEMINI_API_KEY` (default model
  `gemini-3.7-flash`) or Anthropic with `ANTHROPIC_API_KEY` (default `claude-sonnet-5`) — and
  executes the model's `memory_recall` / `memory_store` calls against the MCP endpoint over
  plain HTTP. Pi uses it for every pass; Cursor and Claude Code use it as the fallback.
- `hooks/pi/bigbrain.ts` — the Pi extension. Registers the seven `memory_*` tools natively and,
  on `agent_settled`, renders the finished turn and spawns the direct runner detached.

The pass runs entirely **outside** the session, so it never appears as a turn in the transcript
you're reading, and the worker cleans up the headless session it creates. Every pass appends one
line to `~/.bigbrain/maintenance.log`. Nothing loops: the worker exports `BIGBRAIN_MAINT=1`, and
both hooks short-circuit when they see it.

Because bigbrain memory is cross-repo, these install at the **user level**, not per-project:

```bash
uv run bigbrain install-hooks                    # Cursor (default) → ~/.cursor
uv run bigbrain install-hooks --target claude    # Claude Code → ~/.claude
uv run bigbrain install-hooks --target pi        # Pi → ~/.pi/agent
uv run bigbrain install-hooks --no-rule          # hooks only
uv run bigbrain install-hooks --no-skills        # skip the bundled skills
```

For Cursor and Claude Code the installer copies the scripts to `<config>/hooks/`, **merges**
the entries into the host's config (`~/.cursor/hooks.json` or `~/.claude/settings.json`,
preserving anything already there and backing up alongside it), installs the rule to
`<config>/rules/`, and installs the bundled skills to `<config>/skills/`. For Pi it copies the
extension to `~/.pi/agent/extensions/`, the direct runner to `~/.pi/agent/hooks/`, writes the
rule into `~/.pi/agent/AGENTS.md` as a managed section, and installs the skills to
`~/.pi/agent/skills/`. All of it is idempotent, and the installer warns about anything the
pass will need at runtime that it cannot find (`jq`, the host CLI, `node`, an API key).

The hook scripts require [`jq`](https://jqlang.github.io/jq/) on `PATH`, plus the headless
agent CLI for the selected host — or, without one, `node` and an API key for the direct runner.

#### API keys for the direct runner

Hooks launched by a GUI application (Cursor started from the Dock, for instance) inherit no
shell profile, so a key exported in `.zshrc` never reaches them. The worker and the runner
therefore also read `~/.bigbrain/env`, a plain `KEY=VALUE` file (comments and `export` prefixes
are fine), filling in only variables that are not already set:

```bash
mkdir -p ~/.bigbrain && chmod 700 ~/.bigbrain
echo 'GEMINI_API_KEY=...' >> ~/.bigbrain/env && chmod 600 ~/.bigbrain/env
```

Gemini is picked when both keys are present; force one with `BIGBRAIN_MAINT_PROVIDER=gemini|anthropic`.

### Installing Cursor Agent

The `cursor` desktop CLI includes an `agent` subcommand that downloads Cursor Agent:

```bash
cursor agent
```

On macOS, this normally installs the executable at `~/.local/bin/cursor-agent`. If the
installation succeeds but `command -v cursor-agent` prints nothing, add that directory to
your shell `PATH` and start a new login shell:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
exec zsh -l

command -v cursor-agent
cursor-agent --help
```

The Bigbrain worker invokes `cursor-agent` directly, rather than `cursor agent`. GUI-launched
hooks do not reliably inherit `.zshrc`, so the worker explicitly prepends `~/.local/bin` to
its own `PATH`. Re-run `install-hooks` after updating Bigbrain so the installed worker has
that behavior:

```bash
uv run bigbrain install-hooks
```

Then verify the next substantive turn in:

```bash
tail ~/.bigbrain/maintenance.log
```

A healthy pass ends with `ok`; a `skipped: neither claude nor cursor-agent is on PATH …`
line means the installed worker is stale, Cursor Agent is not under `~/.local/bin`, and
no direct-runner key was found either.

For Claude Code, verify its headless CLI separately:

```bash
command -v claude
```

Installing `cursor-agent` is a requirement of the **Cursor maintenance hooks**, not of
bigbrain itself. If the relevant CLI is missing from the detached hook's `PATH`, the memory
service and interactive MCP calls continue to work, but automatic post-turn updates are
skipped. The reason is recorded in `~/.bigbrain/maintenance.log`.

The worker is tunable through the environment: `BIGBRAIN_MAINT_MODEL` (defaults to `sonnet` on
Claude Code, `composer-2.5` on Cursor — small models fail this task, answering in prose without
calling the tools), `BIGBRAIN_MAINT_DIRECT_MODEL` (the direct runner's model, kept separate
because the host CLIs and the raw APIs share no model ids), `BIGBRAIN_MAINT_HOST` (force
`claude`, `cursor`, or `direct`), `BIGBRAIN_MAINT_TIMEOUT`, `BIGBRAIN_MAINT_LOG`,
`BIGBRAIN_MAINT_MAX_CHARS`, `BIGBRAIN_MCP_URL`, and `BIGBRAIN_ENV_FILE`. Set
`BIGBRAIN_MAINT_DRYRUN=1` to print the assembled prompt instead of running the pass; it works
on the worker and on the direct runner alike.

## Trimming memory (`bigbrain-trim` skill)

The hook *appends* learnings, and store-time dedup only merges near-identical topics — so over
time the store accumulates append-churned entries and silent duplicates. The bundled
[`bigbrain-trim`](skills/bigbrain-trim/SKILL.md) skill is an **on-demand, destructive**
maintenance pass that compacts append-churned memories to current-truth, consolidates
duplicates, evicts dead weight, and tidies tags — always behind an explicit approval gate. It
ships a read-only [`audit.py`](skills/bigbrain-trim/scripts/audit.py) that ranks trim
candidates. Invoke it by asking the agent to "trim bigbrain memory".

## Syncing two machines (`bigbrain sync`)

Running a store on a laptop *and* a remote dev machine means each one learns things the
other never sees. `bigbrain sync` reconciles them over ssh, in both directions:

```bash
bigbrain sync myhost --dry-run   # show what would move
bigbrain sync myhost             # converge both sides
```

It needs bigbrain installed on the peer, and only ssh in between — there is no daemon or
shared database. The peer's entry point defaults to whatever `bigbrain` resolves to on the
remote `PATH`; ssh runs a non-interactive shell, so pass an absolute path with `--peer-cmd`
(e.g. `--peer-cmd '~/src/bigbrain/.venv/bin/bigbrain'`) if that misses.

### How it decides

Memories are matched by **id**, which works because ids are random 63-bit integers: a store
copied to a second machine keeps its ids, and memories created independently on each side
never collide.

The comparison is **three-way** — each side against the other *and* against a snapshot of
the last sync, kept in `$BIGBRAIN_HOME/sync-state.json`. That snapshot is what makes
deletions possible: without it, a memory present on only one side is ambiguous (created
there, or deleted here?), and a two-way merge has to either resurrect deletions forever or
lose new memories.

| Situation | Result |
| --- | --- |
| Only on one side, never synced | Copied to the other |
| Edited on one side | That version wins |
| Edited on both since last sync | Newer `updated_at` wins, and the conflict is reported |
| Deleted on one side, untouched on the other | Deletion propagates |
| Deleted on one side, edited on the other | Survives — an edit outranks a stale delete |

Two things are deliberately *not* treated as edits: `access_count` / `last_accessed`, which
every recall bumps, and float noise in `importance`. Otherwise both sides would look
permanently dirty and sync would never settle.

Use `--direction push` or `pull` for one-way runs. Withheld changes are left out of the
snapshot, so a later two-way sync still sees them as differences rather than assuming they
converged.

### Safety

- Topic vectors are **not** sent; the receiving side recomputes them, so every vector stays
  consistent with that store's own model. A sync aborts if the two sides embed with
  different models.
- The snapshot is written only after **both** sides apply cleanly, so an interrupted sync is
  simply redone next time rather than losing data.
- A partial read from the database raises instead of returning, because missing memories
  would otherwise look like deletions and be propagated to the peer.

### Manual plumbing

`sync` drives these two on the far side; run them directly only when debugging:

```bash
bigbrain sync-dump --json    # this store's memories, human-readable
bigbrain sync-apply          # apply a patch from stdin
```

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
uv run pytest                      # sync planner, export, installer (no Milvus or model needed)
uv run python tests/mcp_smoke.py   # drives the MCP server over stdio
```
