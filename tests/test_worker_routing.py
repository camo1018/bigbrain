"""Backend routing in hooks/bigbrain-maintenance-run.sh (the Cursor / Claude Code worker).

Runs the real worker in dry-run mode against a stub PATH, so which of `claude`,
`cursor-agent`, and `pi` "exist" is controlled per test. Dry-run prints `host=<runner>`
and exits before any agent is launched.

The rule under test: Cursor / Claude Code run the pass on their own CLI (or the other
one), and use Pi only when BIGBRAIN_MAINT_HOST=pi opts in. Pi is never a silent fallback.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "hooks" / "bigbrain-maintenance-run.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("jq") and shutil.which("bash")), reason="worker needs bash and jq"
)

CURSOR_PAYLOAD = {"conversation_id": "c1", "session_id": "c1", "workspace_roots": ["/tmp"]}
CLAUDE_PAYLOAD = {"session_id": "s1"}


def _stub(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def _link_tool(bin_dir: Path, name: str) -> None:
    real = shutil.which(name)
    if real:
        (bin_dir / name).symlink_to(real)


def run_worker(tmp_path, payload, *, clis=(), pi_extension=True, env_file_text="", **env):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    # Only the tools the worker needs, plus whichever agent CLIs this test says exist.
    for tool in ("jq", "node", "date", "dirname", "tr", "find", "sleep", "rm", "mkdir", "cat"):
        _link_tool(bin_dir, tool)
    for cli in clis:
        _stub(bin_dir, cli)
    if pi_extension:
        ext = home / ".pi" / "agent" / "extensions" / "bigbrain.ts"
        ext.parent.mkdir(parents=True, exist_ok=True)
        ext.write_text("// stub\n")

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}})
        + "\n"
    )
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps({**payload, "transcript_path": str(transcript)}))
    env_file = tmp_path / "env"
    env_file.write_text(env_file_text)
    log = tmp_path / "maintenance.log"

    proc = subprocess.run(
        [shutil.which("bash"), str(WORKER), str(payload_file)],
        env={
            "HOME": str(home),
            "PATH": str(bin_dir),
            "TMPDIR": str(tmp_path),
            "BIGBRAIN_ENV_FILE": str(env_file),
            "BIGBRAIN_MAINT_LOG": str(log),
            "BIGBRAIN_MAINT_SETTLE": "0",
            "BIGBRAIN_MAINT_DRYRUN": "1",
            **env,
        },
        capture_output=True,
        text=True,
    )
    first = proc.stdout.splitlines()[0] if proc.stdout else ""
    return first, log.read_text() if log.exists() else ""


@pytest.mark.parametrize(
    "payload, clis, expected",
    [
        (CURSOR_PAYLOAD, ["cursor-agent"], "host=cursor"),
        (CLAUDE_PAYLOAD, ["claude"], "host=claude"),
        # The other host's CLI is still preferred over Pi.
        (CURSOR_PAYLOAD, ["claude"], "host=claude"),
        (CLAUDE_PAYLOAD, ["cursor-agent"], "host=cursor"),
        # Pi being installed changes nothing without the opt-in.
        (CURSOR_PAYLOAD, ["cursor-agent", "pi"], "host=cursor"),
        (CLAUDE_PAYLOAD, ["claude", "pi"], "host=claude"),
    ],
)
def test_host_cli_is_used_by_default(tmp_path, payload, clis, expected):
    first, _ = run_worker(tmp_path, payload, clis=clis)
    assert first == expected


@pytest.mark.parametrize("payload", [CURSOR_PAYLOAD, CLAUDE_PAYLOAD])
def test_no_host_cli_skips_even_when_pi_is_installed(tmp_path, payload):
    first, log = run_worker(
        tmp_path, payload, clis=["pi"], env_file_text="GEMINI_API_KEY=abc\n"
    )
    assert first == ""
    assert "skipped: neither claude nor cursor-agent is on PATH" in log
    assert "BIGBRAIN_MAINT_HOST=pi" in log  # tells the user how to opt in


@pytest.mark.parametrize("payload", [CURSOR_PAYLOAD, CLAUDE_PAYLOAD])
def test_opt_in_routes_to_pi_even_with_host_cli_present(tmp_path, payload):
    first, _ = run_worker(
        tmp_path,
        payload,
        clis=["claude", "cursor-agent", "pi"],
        env_file_text="BIGBRAIN_MAINT_HOST=pi\n",
    )
    assert first == "host=pi"


def test_opt_in_via_environment(tmp_path):
    first, _ = run_worker(tmp_path, CURSOR_PAYLOAD, clis=["pi"], BIGBRAIN_MAINT_HOST="pi")
    assert first == "host=pi"


def test_opt_in_finds_pi_in_its_install_dir(tmp_path):
    """Pi's installer uses ~/.pi/agent/bin, which GUI-launched hooks do not have on PATH."""
    pi_bin = tmp_path / "home" / ".pi" / "agent" / "bin"
    pi_bin.mkdir(parents=True)
    _stub(pi_bin, "pi")
    first, _ = run_worker(tmp_path, CURSOR_PAYLOAD, env_file_text="BIGBRAIN_MAINT_HOST=pi\n")
    assert first == "host=pi"


def test_opt_in_without_pi_fails_instead_of_using_the_host_cli(tmp_path):
    first, log = run_worker(
        tmp_path,
        CURSOR_PAYLOAD,
        clis=["cursor-agent"],
        env_file_text="BIGBRAIN_MAINT_HOST=pi\n",
    )
    assert first == ""
    assert "skipped: BIGBRAIN_MAINT_HOST=pi but pi is not installed" in log


def test_opt_in_without_pi_extension_is_logged(tmp_path):
    first, log = run_worker(
        tmp_path,
        CLAUDE_PAYLOAD,
        clis=["pi"],
        pi_extension=False,
        env_file_text="BIGBRAIN_MAINT_HOST=pi\n",
    )
    assert first == ""
    assert "the bigbrain Pi extension is missing" in log


def test_opt_in_pins_the_pi_provider(tmp_path):
    """BIGBRAIN_MAINT_HOST=pi means Pi, even if a legacy provider is also configured."""
    first, _ = run_worker(
        tmp_path,
        CURSOR_PAYLOAD,
        clis=["pi"],
        env_file_text="BIGBRAIN_MAINT_HOST=pi\nBIGBRAIN_MAINT_PROVIDER=gemini\n",
    )
    assert first == "host=pi"


def test_direct_host_uses_the_selected_legacy_provider(tmp_path):
    first, _ = run_worker(
        tmp_path,
        CURSOR_PAYLOAD,
        env_file_text="BIGBRAIN_MAINT_HOST=direct\nBIGBRAIN_MAINT_PROVIDER=gemini\nGEMINI_API_KEY=abc\n",
    )
    assert first == "host=direct"


def test_unknown_host_is_logged(tmp_path):
    first, log = run_worker(tmp_path, CURSOR_PAYLOAD, clis=["cursor-agent"], BIGBRAIN_MAINT_HOST="nope")
    assert first == ""
    assert "unknown BIGBRAIN_MAINT_HOST 'nope'" in log
