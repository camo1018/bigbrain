"""bigbrain command-line interface.

A thin front-door over MemoryStore, sharing the exact same core library as the
MCP server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import sync as sync_mod
from .config import Config
from .store import MemoryStore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_SCRIPTS = (
    "bigbrain-mark-substantive.sh",
    "bigbrain-maintenance.sh",
    "bigbrain-maintenance-run.sh",
    "bigbrain-maintenance-direct.mjs",
)
_DIRECT_RUNNER = "bigbrain-maintenance-direct.mjs"
_PI_EXTENSION = "pi/bigbrain.ts"
_PI_RUNNER_DIR = "bigbrain"
_RULE_SOURCE = "bigbrain-memory.mdc"
_SKILLS_DIR = "skills"
_MCP_URL = "http://127.0.0.1:8765/mcp"

_REPO_PLACEHOLDER = "__BIGBRAIN_REPO__"

# Markers delimiting the bigbrain section the installer owns inside Pi's AGENTS.md, so a
# reinstall replaces that section and leaves the rest of the file alone.
_AGENTS_BEGIN = "<!-- bigbrain-memory:begin (managed by `bigbrain install-hooks`) -->"
_AGENTS_END = "<!-- bigbrain-memory:end -->"

# Cursor and Claude Code take the same hook scripts but disagree on where config lives,
# how hook entries nest, and what extension an auto-loaded rule needs. Pi has no hook
# config at all: an extension registers the tools and runs the pass, and the rule goes
# into its global AGENTS.md.
_TARGETS = {
    "cursor": {
        "config_dir": ".cursor",
        "config_file": "hooks.json",
        "template": "cursor/hooks.json.template",
        "rule_file": "bigbrain-memory.mdc",
        "cli": "cursor-agent",
        "reload_hint": "Reload Cursor (or it will hot-reload hooks.json) and approve the new hooks.",
        "mcp_hint": (
            "Point Cursor at the server: `uv run bigbrain install-server` "
            f"(or add {{\"url\": \"{_MCP_URL}\"}} to ~/.cursor/mcp.json)."
        ),
    },
    "claude": {
        "config_dir": ".claude",
        "config_file": "settings.json",
        "template": "claude/settings.json.template",
        "rule_file": "bigbrain-memory.md",
        "cli": "claude",
        "reload_hint": "Start a new Claude Code session so the hooks and rule load.",
        "mcp_hint": (
            "Register the server once, at user scope: "
            f"`claude mcp add --scope user --transport http bigbrain {_MCP_URL}`"
        ),
    },
    "pi": {
        "config_dir": ".pi/agent",
        "config_file": None,
        "template": None,
        "rule_file": "AGENTS.md",
        "cli": None,
        "reload_hint": "Run /reload in an open Pi session (or start a new one) to load the extension.",
        "mcp_hint": (
            f"Nothing to register: the extension talks to {_MCP_URL} directly, and the "
            "maintenance pass runs through Pi's own model config (no API key). "
            "Just keep the server running (`uv run bigbrain install-server` on macOS)."
        ),
    },
}

_LAUNCHD_LABEL = "com.bigbrain.mcp"
_PLIST_TEMPLATE = "com.bigbrain.mcp.plist.template"

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="bigbrain — a hand-rolled semantic memory store (Milvus Lite + local embeddings).",
)
console = Console()
err_console = Console(stderr=True)


def _store() -> MemoryStore:
    return MemoryStore(Config.from_env())


def _parse_tags(tags: Optional[str]) -> list[str]:
    if not tags:
        return []
    return [t.strip() for t in tags.split(",") if t.strip()]


@app.command()
def setup() -> None:
    """Download and cache the local embedding model (one-time)."""
    from ._bootstrap import download_model

    with console.status("Downloading embedding model..."):
        name = download_model()
    console.print(f"[green]Model ready:[/green] {name}")


@app.command()
def store(
    topic: str = typer.Argument(..., help="Short semantic key (gets embedded)."),
    content: str = typer.Argument(..., help="The detailed knowledge to remember."),
    tags: Optional[str] = typer.Option(None, "--tags", "-t", help="Comma-separated tags."),
    source: str = typer.Option("", "--source", "-s", help="Where this came from."),
    importance: float = typer.Option(0.5, "--importance", "-i", min=0.0, max=1.0),
    on_conflict: str = typer.Option(
        "merge", "--on-conflict", help="merge | replace | skip | new"
    ),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Always insert a new memory."),
) -> None:
    """Store a memory (auto-merges near-duplicate topics)."""
    s = _store()
    mem, action = s.store(
        topic,
        content,
        tags=_parse_tags(tags),
        source=source,
        importance=importance,
        on_conflict=on_conflict,  # type: ignore[arg-type]
        dedup=not no_dedup,
    )
    color = {"created": "green", "merged": "yellow", "replaced": "yellow", "skipped": "dim"}
    console.print(f"[{color.get(action, 'white')}]{action}[/] id={mem.id}  topic={mem.topic!r}")
    s.close()


@app.command()
def recall(
    query: str = typer.Argument(..., help="What you're trying to remember."),
    limit: int = typer.Option(5, "--limit", "-n", min=1),
    tags: Optional[str] = typer.Option(None, "--tags", "-t"),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    min_similarity: float = typer.Option(0.0, "--min-similarity", min=0.0, max=1.0),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
) -> None:
    """Semantically recall memories, reranked by similarity + recency + importance."""
    s = _store()
    results = s.recall(
        query,
        limit=limit,
        tags=_parse_tags(tags),
        source=source,
        min_similarity=min_similarity,
    )
    if as_json:
        console.print_json(json.dumps([m.to_dict() for m in results]))
    else:
        _print_memories(results, show_scores=True)
    s.close()


@app.command(name="list")
def list_(
    limit: int = typer.Option(50, "--limit", "-n", min=1),
    offset: int = typer.Option(0, "--offset", min=0),
    tags: Optional[str] = typer.Option(None, "--tags", "-t"),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List stored memories (most recently updated first)."""
    s = _store()
    results = s.list(limit=limit, offset=offset, tags=_parse_tags(tags), source=source)
    if as_json:
        console.print_json(json.dumps([m.to_dict() for m in results]))
    else:
        _print_memories(results, show_scores=False)
    s.close()


@app.command()
def get(memory_id: int = typer.Argument(...)) -> None:
    """Show one memory in full."""
    s = _store()
    mem = s.get(memory_id)
    if mem is None:
        err_console.print(f"[red]No memory with id {memory_id}[/red]")
        raise typer.Exit(1)
    console.print_json(json.dumps(mem.to_dict(), indent=2))
    s.close()


@app.command()
def update(
    memory_id: int = typer.Argument(...),
    topic: Optional[str] = typer.Option(None, "--topic"),
    content: Optional[str] = typer.Option(None, "--content"),
    tags: Optional[str] = typer.Option(None, "--tags", "-t"),
    source: Optional[str] = typer.Option(None, "--source", "-s"),
    importance: Optional[float] = typer.Option(None, "--importance", "-i", min=0.0, max=1.0),
) -> None:
    """Update fields of an existing memory (re-embeds if the topic changes)."""
    s = _store()
    mem = s.update(
        memory_id,
        topic=topic,
        content=content,
        tags=_parse_tags(tags) if tags is not None else None,
        source=source,
        importance=importance,
    )
    if mem is None:
        err_console.print(f"[red]No memory with id {memory_id}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]updated[/green] id={mem.id}")
    s.close()


@app.command()
def delete(
    memory_ids: list[int] = typer.Argument(..., help="One or more memory ids."),
) -> None:
    """Delete one or more memories by id."""
    s = _store()
    n = s.delete(memory_ids)
    console.print(f"[green]deleted[/green] {n}")
    s.close()


@app.command()
def count() -> None:
    """Print the number of stored memories."""
    s = _store()
    console.print(s.count())
    s.close()


@app.command(name="sync-dump")
def sync_dump(
    as_json: bool = typer.Option(
        False, "--json", help="Emit readable JSON instead of the gzipped wire format."
    ),
) -> None:
    """Emit this store's memories for a peer to reconcile against.

    Writes the gzipped payload to stdout and nothing else, so a peer can invoke this
    over ssh and read the result straight off the pipe.
    """
    cfg = Config.from_env()
    s = MemoryStore(cfg)
    try:
        payload = sync_mod.make_dump(s.export_rows(), cfg)
    finally:
        s.close()

    if as_json:
        console.print_json(json.dumps(payload))
        return
    sys.stdout.buffer.write(sync_mod.encode_payload(payload))
    sys.stdout.buffer.flush()


@app.command(name="sync-apply")
def sync_apply() -> None:
    """Apply a sync patch read from stdin, printing a one-line JSON summary.

    Invoked on the far side by `bigbrain sync`; rarely run by hand.
    """
    s = MemoryStore(Config.from_env())
    try:
        payload = sync_mod.decode_payload(sys.stdin.buffer.read())
        summary = sync_mod.apply_patch(s, payload)
    except sync_mod.SyncError as exc:
        err_console.print(f"sync-apply failed: {exc}")
        raise typer.Exit(1)
    finally:
        s.close()
    sys.stdout.write(json.dumps(summary) + "\n")


def _print_sync_report(report: "sync_mod.Report", *, dry_run: bool) -> None:
    plan = report.plan
    counts = plan.counts()
    console.print(
        f"[bold]local[/bold] {report.local_count} memories    "
        f"[bold]{report.peer_host}[/bold] {report.peer_count} memories"
    )

    if plan.is_empty:
        console.print("[green]already in sync[/green]")
    else:
        arrow = "would send" if dry_run else "sent"
        console.print(
            f"  {arrow} to peer   [green]{counts['to_peer']}[/green] memories, "
            f"[red]{counts['delete_peer']}[/red] deletions"
        )
        console.print(
            f"  {arrow} to local  [green]{counts['to_local']}[/green] memories, "
            f"[red]{counts['delete_local']}[/red] deletions"
        )

    for note in plan.notes:
        console.print(f"  [yellow]note[/yellow] {note}")
    for conflict in plan.conflicts:
        console.print(f"  [magenta]conflict[/magenta] {conflict}")
    if counts["deferred"]:
        console.print(
            f"  [dim]{counts['deferred']} change(s) left alone by the direction filter[/dim]"
        )

    if dry_run:
        console.print("[dim]dry run — nothing was written[/dim]")
    elif not plan.is_empty:
        console.print(f"[dim]state: {sync_mod.state_path(Config.from_env())}[/dim]")


@app.command()
def sync(
    host: str = typer.Argument(..., help="ssh host of the peer, e.g. myhost or user@myhost."),
    peer_cmd: str = typer.Option(
        sync_mod.DEFAULT_PEER_COMMAND, "--peer-cmd", help="bigbrain entry point on the peer."
    ),
    name: Optional[str] = typer.Option(
        None, "--name", help="Track this peer under this key (defaults to the host)."
    ),
    direction: str = typer.Option("both", "--direction", help="both | push | pull"),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Report what would change and write nothing."
    ),
) -> None:
    """Reconcile this store with a peer's over ssh, in both directions.

    Compares both sides against a snapshot of the last sync, so a memory only on one
    side is correctly read as created there or deleted here. Deletions propagate,
    and a memory edited on both sides keeps the newer version and is reported.
    """
    if direction not in ("both", "push", "pull"):
        err_console.print("[red]--direction must be one of: both, push, pull[/red]")
        raise typer.Exit(2)

    peer = sync_mod.Peer(host=host, command=peer_cmd, key=name)
    try:
        report = sync_mod.run_sync(
            lambda: MemoryStore(Config.from_env()),
            peer,
            direction=direction,  # type: ignore[arg-type]
            dry_run=dry_run,
        )
    except sync_mod.SyncError as exc:
        err_console.print(f"[red]sync failed:[/red] {exc}")
        raise typer.Exit(1)

    _print_sync_report(report, dry_run=dry_run)


def _merge_hook_entries(existing: dict, template: dict) -> tuple[dict, list[str]]:
    """Merge template hook entries into an existing hooks.json dict, idempotently.

    Adds only entries whose `command` is not already present, preserving any
    other hooks the user has configured. Returns (merged, added_descriptions).
    """
    existing.setdefault("version", template.get("version", 1))
    hooks = existing.setdefault("hooks", {})
    added: list[str] = []
    for event, entries in template.get("hooks", {}).items():
        bucket = hooks.setdefault(event, [])
        for entry in entries:
            if any(e.get("command") == entry.get("command") for e in bucket):
                continue
            bucket.append(entry)
            added.append(f"{event}: {entry['command']}")
    return existing, added


def _merge_claude_hook_entries(existing: dict, template: dict) -> tuple[dict, list[str]]:
    """Merge template hook groups into a Claude Code settings.json dict, idempotently.

    Claude nests hooks one level deeper than Cursor: each event holds matcher groups and
    each group holds the actual entries. Dedup is on the inner `command`, so reinstalling
    neither duplicates entries nor disturbs unrelated settings such as `permissions`.
    """
    hooks = existing.setdefault("hooks", {})
    added: list[str] = []
    for event, groups in template.get("hooks", {}).items():
        bucket = hooks.setdefault(event, [])
        present = {
            entry.get("command") for group in bucket for entry in group.get("hooks", [])
        }
        for group in groups:
            keep = [e for e in group.get("hooks", []) if e.get("command") not in present]
            if not keep:
                continue
            bucket.append({**group, "hooks": keep})
            added.extend(f"{event}: {e['command']}" for e in keep)
    return existing, added


def _stamp_repo_path(path: Path) -> None:
    """Resolve the repo-path placeholder in a script installed outside the checkout."""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return
    if _REPO_PLACEHOLDER in text:
        path.write_text(text.replace(_REPO_PLACEHOLDER, str(_REPO_ROOT)))


def _rule_body() -> str:
    """The memory rule's Markdown with its Cursor-specific YAML frontmatter stripped."""
    text = (_REPO_ROOT / "rules" / _RULE_SOURCE).read_text()
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + len("\n---"):]
    return text.strip() + "\n"


def _merge_agents_md(path: Path, body: str) -> str:
    """Write the rule into an AGENTS.md as a marked section; replace it if already there.

    Returns "created", "updated", or "unchanged". Anything outside the markers is kept,
    so users can hold their own instructions in the same file.
    """
    section = f"{_AGENTS_BEGIN}\n{body}{_AGENTS_END}\n"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(section)
        return "created"

    existing = path.read_text()
    start = existing.find(_AGENTS_BEGIN)
    end = existing.find(_AGENTS_END)
    if start != -1 and end != -1 and end > start:
        rest = existing[end + len(_AGENTS_END):].lstrip("\n")
        merged = existing[:start] + section + ("\n" + rest if rest else "")
    else:
        merged = existing.rstrip("\n") + "\n\n" + section if existing.strip() else section
    if merged == existing:
        return "unchanged"
    shutil.copyfile(path, path.with_suffix(".md.bak"))
    path.write_text(merged)
    return "updated"


def _api_key_available() -> bool:
    """Whether the direct runner would find an LLM key in the env or ~/.bigbrain/env."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env_file = Path(os.environ.get("BIGBRAIN_ENV_FILE", Config.from_env().home / "env"))
    if not env_file.is_file():
        return False
    text = env_file.read_text()
    return "GEMINI_API_KEY=" in text or "ANTHROPIC_API_KEY=" in text


def _pi_pass_available() -> bool:
    """Whether the direct runner can run the pass through Pi's own configured providers."""
    return shutil.which("pi") is not None


def _install_skills(dest_root: Path) -> list[str]:
    """Copy bundled skills into the host's skills dir, preserving executable bits.

    Each `skills/<name>/` directory is mirrored under the user's skills dir; script
    files keep their +x bit so the agent can run them directly. Returns the names of
    the skills installed.
    """
    src_root = _REPO_ROOT / _SKILLS_DIR
    if not src_root.is_dir():
        return []
    installed: list[str] = []
    for skill in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dest = dest_root / "skills" / skill.name
        shutil.copytree(skill, dest, dirs_exist_ok=True)
        for script in (dest / "scripts").glob("*"):
            if script.is_file():
                _stamp_repo_path(script)
                os.chmod(script, 0o755)
        installed.append(skill.name)
    return installed


@app.command(name="install-hooks")
def install_hooks(
    target: str = typer.Option(
        "cursor", "--target", "-t", help="Agent host to install into: cursor, claude, or pi."
    ),
    config_dir: Optional[Path] = typer.Option(
        None,
        "--config-dir",
        "--cursor-dir",
        help="Target config dir (defaults to ~/.cursor, ~/.claude, or ~/.pi/agent).",
    ),
    with_rule: bool = typer.Option(
        True, "--with-rule/--no-rule", help="Also install the bigbrain memory rule."
    ),
    with_skills: bool = typer.Option(
        True, "--with-skills/--no-skills", help="Also install the bundled bigbrain skills."
    ),
) -> None:
    """Install the memory-maintenance hooks (rule + skills) for Cursor, Claude Code, or Pi.

    Cursor / Claude Code: copies the hook scripts, merges the hook entries into the
    host's config file (preserving anything already there), installs the agent memory
    rule, and installs the bundled skills (e.g. bigbrain-trim).

    Pi: installs the bigbrain extension (native memory tools + post-turn pass) and the
    direct runner it launches, writes the rule into ~/.pi/agent/AGENTS.md as a managed
    section, and installs the skills.

    Everything is user-level, so it applies in every chat and repo.
    """
    spec = _TARGETS.get(target)
    if spec is None:
        err_console.print(f"[red]--target must be one of: {', '.join(_TARGETS)}[/red]")
        raise typer.Exit(2)

    default_root = Path.home() / spec["config_dir"]
    dest_root = config_dir or default_root
    src_hooks = _REPO_ROOT / "hooks"

    if target == "pi":
        _install_pi(dest_root, src_hooks)
    else:
        _install_hook_host(target, spec, dest_root, default_root, src_hooks)

    if with_rule:
        src_rule = _REPO_ROOT / "rules" / _RULE_SOURCE
        if not src_rule.exists():
            err_console.print(f"[yellow]rule not found at {src_rule}; skipped[/yellow]")
        elif target == "pi":
            agents_md = dest_root / spec["rule_file"]
            outcome = _merge_agents_md(agents_md, _rule_body())
            console.print(f"[green]{outcome} rule section[/green] in {agents_md}")
        else:
            dest_rules = dest_root / "rules"
            dest_rules.mkdir(parents=True, exist_ok=True)
            dest_rule = dest_rules / spec["rule_file"]
            shutil.copyfile(src_rule, dest_rule)
            console.print(f"[green]installed rule[/green] {dest_rule}")

    if with_skills:
        installed = _install_skills(dest_root)
        if installed:
            console.print(
                f"[green]installed skills[/green] in {dest_root / 'skills'}: "
                + ", ".join(installed)
            )
        else:
            console.print("[dim]no bundled skills found; skipped[/dim]")

    console.print(
        "\n[bold]Next steps:[/bold]\n"
        f"  1. {spec['reload_hint']}\n"
        f"  2. {spec['mcp_hint']}\n"
        "  3. Start a new chat so the rule loads, then check ~/.bigbrain/maintenance.log\n"
        "     after the first turn that uses a tool."
    )


def _install_hook_host(
    target: str, spec: dict, dest_root: Path, default_root: Path, src_hooks: Path
) -> None:
    """Cursor / Claude Code: copy the hook scripts and register them in the host config."""
    template_path = src_hooks / spec["template"]
    if not template_path.exists():
        err_console.print(f"[red]Cannot find bundled hooks at {template_path}[/red]")
        raise typer.Exit(1)

    if shutil.which("jq") is None:
        err_console.print(
            "[yellow]warning:[/yellow] `jq` not found on PATH — the hook scripts "
            "require it at runtime. Install jq before relying on the hooks."
        )
    if shutil.which(spec["cli"]) is None:
        fallback_ok = shutil.which("node") is not None and (
            _api_key_available()
            or (_pi_pass_available() and (Path.home() / ".pi/agent/extensions/bigbrain.ts").is_file())
        )
        status = "so the direct runner will handle the pass" if fallback_ok else (
            "and the direct-runner fallback needs `node` plus either Pi with the bigbrain "
            "extension installed (`bigbrain install-hooks --target pi`) or GEMINI_API_KEY / "
            "ANTHROPIC_API_KEY (in the environment or ~/.bigbrain/env)"
        )
        err_console.print(
            f"[yellow]note:[/yellow] `{spec['cli']}` not found on PATH, {status}."
        )

    dest_hooks = dest_root / "hooks"
    dest_hooks.mkdir(parents=True, exist_ok=True)
    for name in _HOOK_SCRIPTS:
        dst = dest_hooks / name
        shutil.copyfile(src_hooks / name, dst)
        os.chmod(dst, 0o755)
        console.print(f"[green]installed script[/green] {dst}")

    # The Claude template spells out an absolute hook path, so a non-default target dir
    # has to be substituted in or the registered hooks would point at the wrong tree.
    raw_template = template_path.read_text()
    if dest_root != default_root:
        raw_template = raw_template.replace(
            f"$HOME/{spec['config_dir']}/hooks", str(dest_hooks)
        )
    template = json.loads(raw_template)

    config_path = dest_root / spec["config_file"]
    existing = json.loads(config_path.read_text()) if config_path.exists() else {}
    if config_path.exists():
        shutil.copyfile(config_path, config_path.with_suffix(".json.bak"))
    merge = _merge_claude_hook_entries if target == "claude" else _merge_hook_entries
    merged, added = merge(existing, template)
    config_path.write_text(json.dumps(merged, indent=2) + "\n")
    if added:
        console.print(f"[green]registered hooks[/green] in {config_path}: " + ", ".join(added))
    else:
        console.print(f"[dim]hooks already registered[/dim] in {config_path}")


def _remove_legacy_pi_runner(dest_root: Path) -> None:
    """Drop the runner older installs put in `<config>/hooks/`.

    Pi treats any `hooks/` directory as a leftover from before extensions and blocks
    startup with a warning, so remove the directory too once nothing else is in it.
    """
    legacy_dir = dest_root / "hooks"
    legacy = legacy_dir / _DIRECT_RUNNER
    if legacy.is_file():
        legacy.unlink()
        console.print(f"[green]removed legacy script[/green] {legacy}")
    try:
        legacy_dir.rmdir()
    except OSError:
        pass  # absent, or holds files that are not ours


def _install_pi(dest_root: Path, src_hooks: Path) -> None:
    """Pi: install the extension and the direct runner it spawns.

    The runner performs the pass with a headless `pi -p` that loads only the bigbrain
    extension, so it reuses Pi's configured providers and needs no separate API key.
    Pi auto-discovers `extensions/*.ts` and warns about a legacy `hooks/` directory, so the
    runner lives under `bigbrain/` instead. The extension looks for the runner there first
    and falls back to the stamped checkout path.
    """
    if shutil.which("node") is None:
        err_console.print(
            "[yellow]warning:[/yellow] `node` not found on PATH — the Pi maintenance "
            "pass runs on Node.js (Pi itself needs it too)."
        )
    # The pass normally runs through a headless Pi using Pi's own providers, so no key is
    # required. An API key is only the fallback for when the runner cannot launch Pi.
    if not _pi_pass_available() and not _api_key_available():
        err_console.print(
            "[yellow]note:[/yellow] `pi` is not on PATH here. The Pi extension passes its own "
            "CLI path to the runner, so the pass still runs from inside Pi; standalone runs "
            "would need GEMINI_API_KEY or ANTHROPIC_API_KEY (environment or ~/.bigbrain/env)."
        )

    dest_runner_dir = dest_root / _PI_RUNNER_DIR
    dest_runner_dir.mkdir(parents=True, exist_ok=True)
    runner = dest_runner_dir / _DIRECT_RUNNER
    shutil.copyfile(src_hooks / _DIRECT_RUNNER, runner)
    os.chmod(runner, 0o755)
    console.print(f"[green]installed script[/green] {runner}")
    _remove_legacy_pi_runner(dest_root)

    dest_ext = dest_root / "extensions"
    dest_ext.mkdir(parents=True, exist_ok=True)
    extension = dest_ext / Path(_PI_EXTENSION).name
    shutil.copyfile(src_hooks / _PI_EXTENSION, extension)
    _stamp_repo_path(extension)
    console.print(f"[green]installed extension[/green] {extension}")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (default 127.0.0.1)."),
    port: Optional[int] = typer.Option(None, "--port", help="Bind port (default 8765)."),
) -> None:
    """Run the shared long-lived memory server over streamable HTTP.

    One loopback-bound process that every Cursor workbench connects to by URL,
    instead of Cursor spawning a stdio subprocess per workbench. Normally managed
    by the LaunchAgent (see `install-server`); run directly for debugging.
    """
    from .mcp_server import serve_http

    cfg = Config.from_env()
    resolved_host = host or cfg.http_host
    resolved_port = port or cfg.http_port
    console.print(
        f"[green]bigbrain[/green] serving on "
        f"http://{resolved_host}:{resolved_port}/mcp  (Ctrl-C to stop)"
    )
    serve_http(resolved_host, resolved_port)


def _launchctl(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["launchctl", *args], capture_output=True, text=True, check=check
    )


def _wait_for_http(host: str, port: int, timeout: float = 15.0) -> bool:
    """Poll the server until it accepts connections (any HTTP reply counts)."""
    import socket
    import time
    import urllib.error
    import urllib.request

    connect_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{connect_host}:{port}/mcp"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1.0)
            return True
        except urllib.error.HTTPError:
            return True  # server answered (405/406/etc.) => it's up
        except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
            time.sleep(0.4)
    return False


def _set_mcp_server_entry(cursor_dir: Path, entry: dict) -> Path:
    """Rewrite the `bigbrain` entry in ~/.cursor/mcp.json, backing up first."""
    mcp_json = cursor_dir / "mcp.json"
    data = json.loads(mcp_json.read_text()) if mcp_json.exists() else {}
    if mcp_json.exists():
        shutil.copyfile(mcp_json, mcp_json.with_suffix(".json.bak"))
    data.setdefault("mcpServers", {})["bigbrain"] = entry
    mcp_json.write_text(json.dumps(data, indent=2) + "\n")
    return mcp_json


@app.command(name="install-server")
def install_server(
    host: Optional[str] = typer.Option(None, "--host", help="Bind address (default 127.0.0.1)."),
    port: Optional[int] = typer.Option(None, "--port", help="Bind port (default 8765)."),
    cursor_dir: Path = typer.Option(
        Path.home() / ".cursor", "--cursor-dir", help="Target Cursor config dir."
    ),
) -> None:
    """Install & start the shared HTTP server as a macOS LaunchAgent, and point
    Cursor's mcp.json at it by URL.

    This is the durable fix for Cursor's per-workbench stdio-spawn storm: one
    always-on server owns the DB, and each workbench connects by URL (a cheap
    HTTP handshake) instead of forking its own subprocess.
    """
    if sys.platform != "darwin":
        err_console.print("[red]install-server currently supports macOS (launchd) only.[/red]")
        raise typer.Exit(1)

    cfg = Config.from_env()
    resolved_host = host or cfg.http_host
    resolved_port = port or cfg.http_port

    template_path = _REPO_ROOT / "launchd" / _PLIST_TEMPLATE
    if not template_path.exists():
        err_console.print(f"[red]Cannot find plist template at {template_path}[/red]")
        raise typer.Exit(1)

    program = Path(sys.executable).with_name("bigbrain")
    if not program.exists():
        err_console.print(f"[red]Could not find the bigbrain console script at {program}[/red]")
        raise typer.Exit(1)

    home = cfg.ensure_home()
    logs_dir = home / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plist = (
        template_path.read_text()
        .replace("__LABEL__", _LAUNCHD_LABEL)
        .replace("__PROGRAM__", str(program))
        .replace("__HOST__", resolved_host)
        .replace("__PORT__", str(resolved_port))
        .replace("__BIGBRAIN_HOME__", str(home))
        .replace("__STDOUT__", str(logs_dir / "server.out.log"))
        .replace("__STDERR__", str(logs_dir / "server.err.log"))
    )

    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    plist_path = agents_dir / f"{_LAUNCHD_LABEL}.plist"
    plist_path.write_text(plist)
    console.print(f"[green]wrote LaunchAgent[/green] {plist_path}")

    uid = os.getuid()
    domain = f"gui/{uid}"
    # Replace any previous instance, then load and (re)start.
    _launchctl("bootout", f"{domain}/{_LAUNCHD_LABEL}")
    boot = _launchctl("bootstrap", domain, str(plist_path))
    if boot.returncode != 0:
        err_console.print(
            f"[red]launchctl bootstrap failed:[/red] {boot.stderr.strip() or boot.stdout.strip()}"
        )
        raise typer.Exit(1)
    _launchctl("enable", f"{domain}/{_LAUNCHD_LABEL}")
    _launchctl("kickstart", "-k", f"{domain}/{_LAUNCHD_LABEL}")
    console.print(f"[green]loaded & started[/green] {_LAUNCHD_LABEL}")

    if _wait_for_http(resolved_host, resolved_port):
        console.print(f"[green]server is up[/green] on http://{resolved_host}:{resolved_port}/mcp")
    else:
        err_console.print(
            "[yellow]warning:[/yellow] server did not answer yet; check "
            f"{logs_dir / 'server.err.log'}"
        )

    url = f"http://{resolved_host}:{resolved_port}/mcp"
    mcp_json = _set_mcp_server_entry(cursor_dir, {"url": url})
    console.print(f"[green]pointed Cursor at[/green] {url} (in {mcp_json})")

    steps = [
        "Reload Cursor (or toggle the bigbrain MCP server off/on) to pick up the URL.",
        "The server now starts on login and respawns automatically.",
        f"Logs: {logs_dir / 'server.err.log'}",
    ]
    if shutil.which("claude"):
        steps.append(
            "Claude Code: `claude mcp add --scope user --transport http bigbrain "
            f"{url}` (once)."
        )
    if shutil.which("pi"):
        steps.append("Pi: nothing to register; `bigbrain install-hooks --target pi` installs the bridge.")
    console.print(
        "\n[bold]Next steps:[/bold]\n"
        + "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
    )


@app.command(name="uninstall-server")
def uninstall_server(
    cursor_dir: Path = typer.Option(
        Path.home() / ".cursor", "--cursor-dir", help="Target Cursor config dir."
    ),
    restore_stdio: bool = typer.Option(
        True,
        "--restore-stdio/--no-restore-stdio",
        help="Revert Cursor's mcp.json bigbrain entry to the stdio command form.",
    ),
) -> None:
    """Stop & remove the LaunchAgent and (optionally) revert mcp.json to stdio."""
    if sys.platform != "darwin":
        err_console.print("[red]uninstall-server currently supports macOS (launchd) only.[/red]")
        raise typer.Exit(1)

    uid = os.getuid()
    _launchctl("bootout", f"gui/{uid}/{_LAUNCHD_LABEL}")
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{_LAUNCHD_LABEL}.plist"
    if plist_path.exists():
        plist_path.unlink()
        console.print(f"[green]removed[/green] {plist_path}")
    else:
        console.print(f"[dim]no LaunchAgent at {plist_path}[/dim]")

    if restore_stdio:
        entry = {
            "command": "uv",
            "args": ["run", "--no-sync", "--directory", str(_REPO_ROOT), "bigbrain-mcp"],
            "env": {"BIGBRAIN_HOME": str(Config.from_env().home)},
        }
        mcp_json = _set_mcp_server_entry(cursor_dir, entry)
        console.print(f"[green]reverted bigbrain to stdio[/green] in {mcp_json}")
    console.print("\n[bold]Reload Cursor[/bold] to apply the change.")


def _print_memories(memories, *, show_scores: bool) -> None:
    if not memories:
        console.print("[dim]no memories[/dim]")
        return
    table = Table(show_lines=True)
    table.add_column("id", style="cyan", no_wrap=True)
    if show_scores:
        table.add_column("score", justify="right")
        table.add_column("sim", justify="right")
    table.add_column("topic", style="bold")
    table.add_column("content")
    table.add_column("tags", style="magenta")
    table.add_column("imp", justify="right")
    for m in memories:
        row = [str(m.id)]
        if show_scores:
            row += [f"{m.score:.3f}" if m.score is not None else "-",
                    f"{m.similarity:.3f}" if m.similarity is not None else "-"]
        content = m.content if len(m.content) <= 200 else m.content[:197] + "..."
        row += [m.topic, content, ", ".join(m.tags), f"{m.importance:.2f}"]
        table.add_row(*row)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
