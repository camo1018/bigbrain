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

from .config import Config
from .store import MemoryStore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOK_SCRIPTS = ("bigbrain-mark-substantive.sh", "bigbrain-maintenance.sh")
_RULE_FILE = "bigbrain-memory.mdc"
_SKILLS_DIR = "skills"

_REPO_PLACEHOLDER = "__BIGBRAIN_REPO__"

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


def _stamp_repo_path(path: Path) -> None:
    """Resolve the repo-path placeholder in a script installed outside the checkout."""
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return
    if _REPO_PLACEHOLDER in text:
        path.write_text(text.replace(_REPO_PLACEHOLDER, str(_REPO_ROOT)))


def _install_skills(cursor_dir: Path) -> list[str]:
    """Copy bundled skills into ~/.cursor/skills/, preserving executable bits.

    Each `skills/<name>/` directory is mirrored under the user's Cursor skills
    dir; script files keep their +x bit so the agent can run them directly.
    Returns the names of the skills installed.
    """
    src_root = _REPO_ROOT / _SKILLS_DIR
    if not src_root.is_dir():
        return []
    installed: list[str] = []
    for skill in sorted(p for p in src_root.iterdir() if p.is_dir()):
        dest = cursor_dir / "skills" / skill.name
        shutil.copytree(skill, dest, dirs_exist_ok=True)
        for script in (dest / "scripts").glob("*"):
            if script.is_file():
                _stamp_repo_path(script)
                os.chmod(script, 0o755)
        installed.append(skill.name)
    return installed


@app.command(name="install-hooks")
def install_hooks(
    cursor_dir: Path = typer.Option(
        Path.home() / ".cursor",
        "--cursor-dir",
        help="Target Cursor config dir (user-level).",
    ),
    with_rule: bool = typer.Option(
        True, "--with-rule/--no-rule", help="Also install the bigbrain memory rule."
    ),
    with_skills: bool = typer.Option(
        True, "--with-skills/--no-skills", help="Also install the bundled bigbrain skills."
    ),
) -> None:
    """Install the memory-maintenance Cursor hooks (rule + skills) into ~/.cursor.

    Copies the hook scripts, merges the hook entries into the user's hooks.json
    (preserving any existing hooks), installs the agent memory rule, and installs
    the bundled skills (e.g. bigbrain-trim). These are user-level so they apply
    in every chat and repo.
    """
    src_hooks = _REPO_ROOT / "hooks"
    template_path = src_hooks / "hooks.json.template"
    if not template_path.exists():
        err_console.print(f"[red]Cannot find bundled hooks at {src_hooks}[/red]")
        raise typer.Exit(1)

    if shutil.which("jq") is None:
        err_console.print(
            "[yellow]warning:[/yellow] `jq` not found on PATH — the hook scripts "
            "require it at runtime. Install jq before relying on the hooks."
        )

    dest_hooks = cursor_dir / "hooks"
    dest_hooks.mkdir(parents=True, exist_ok=True)
    for name in _HOOK_SCRIPTS:
        dst = dest_hooks / name
        shutil.copyfile(src_hooks / name, dst)
        os.chmod(dst, 0o755)
        console.print(f"[green]installed script[/green] {dst}")

    template = json.loads(template_path.read_text())
    hooks_json = cursor_dir / "hooks.json"
    existing = json.loads(hooks_json.read_text()) if hooks_json.exists() else {}
    if hooks_json.exists():
        shutil.copyfile(hooks_json, hooks_json.with_suffix(".json.bak"))
    merged, added = _merge_hook_entries(existing, template)
    hooks_json.write_text(json.dumps(merged, indent=2) + "\n")
    if added:
        console.print(f"[green]registered hooks[/green] in {hooks_json}: " + ", ".join(added))
    else:
        console.print(f"[dim]hooks already registered[/dim] in {hooks_json}")

    if with_rule:
        src_rule = _REPO_ROOT / "rules" / _RULE_FILE
        if src_rule.exists():
            dest_rules = cursor_dir / "rules"
            dest_rules.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_rule, dest_rules / _RULE_FILE)
            console.print(f"[green]installed rule[/green] {dest_rules / _RULE_FILE}")
        else:
            err_console.print(f"[yellow]rule not found at {src_rule}; skipped[/yellow]")

    if with_skills:
        installed = _install_skills(cursor_dir)
        if installed:
            console.print(
                f"[green]installed skills[/green] in {cursor_dir / 'skills'}: "
                + ", ".join(installed)
            )
        else:
            console.print("[dim]no bundled skills found; skipped[/dim]")

    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Reload Cursor (or it will hot-reload hooks.json) and approve the new hooks.\n"
        "  2. Ensure the bigbrain MCP server is registered (see README).\n"
        "  3. Start a new chat so the rule loads."
    )


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

    console.print(
        "\n[bold]Next steps:[/bold]\n"
        "  1. Reload Cursor (or toggle the bigbrain MCP server off/on) to pick up the URL.\n"
        "  2. The server now starts on login and respawns automatically.\n"
        f"  3. Logs: {logs_dir / 'server.err.log'}"
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
