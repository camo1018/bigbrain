"""Which messages hooks/bigbrain-maintenance-run.sh hands to the maintenance pass.

Runs the real worker in dry-run mode, which prints the assembled prompt, against a
synthetic transcript. A checkpoint file records how many transcript entries the last pass
read, so the window covers the whole run (and any skipped turn) instead of only the
messages after the last user prompt.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WORKER = Path(__file__).resolve().parents[1] / "hooks" / "bigbrain-maintenance-run.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("jq") and shutil.which("bash")), reason="worker needs bash and jq"
)


def user(text):
    return {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}


def assistant(text, tool=None):
    content = [{"type": "text", "text": text}]
    if tool:
        content.append({"type": "tool_use", "name": tool, "input": {"q": "x"}})
    return {"role": "assistant", "message": {"content": content}}


def tool_result(text):
    return {"role": "user", "message": {"content": [{"type": "tool_result", "content": text}]}}


def run(tmp_path, entries, *, dryrun=True, **env):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for tool in ("jq", "date", "dirname", "tr", "find", "sleep", "rm", "mkdir", "cat"):
        real = shutil.which(tool)
        if real and not (bin_dir / tool).exists():
            (bin_dir / tool).symlink_to(real)
    claude = bin_dir / "claude"
    # Stands in for the real CLI when not in dry-run: reports success without a model call.
    claude.write_text('#!/bin/sh\necho \'{"result":"NOOP"}\'\n')
    claude.chmod(0o755)

    transcript = tmp_path / "t.jsonl"
    transcript.write_text("".join(json.dumps(e) + "\n" for e in entries))
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps({"session_id": "s1", "transcript_path": str(transcript)}))
    log = tmp_path / "maintenance.log"
    base = {
        "HOME": str(home),
        "PATH": str(bin_dir),
        "TMPDIR": str(tmp_path),
        "BIGBRAIN_ENV_FILE": str(tmp_path / "env"),
        "BIGBRAIN_MAINT_LOG": str(log),
        "BIGBRAIN_MAINT_SETTLE": "0",
    }
    if dryrun:
        base["BIGBRAIN_MAINT_DRYRUN"] = "1"
    proc = subprocess.run(
        [shutil.which("bash"), str(WORKER), str(payload_file)],
        env={**base, **env},
        capture_output=True,
        text=True,
    )
    return proc.stdout, log.read_text() if log.exists() else ""


def checkpoint(tmp_path):
    return tmp_path / "bigbrain-hooks" / "reviewed-s1"


def test_without_checkpoint_window_starts_at_last_user_prompt(tmp_path):
    out, _ = run(tmp_path, [user("old request"), assistant("old answer"), user("new request"), assistant("ok", "Read")])
    assert "USER: new request" in out
    assert "old request" not in out


def test_checkpoint_widens_window_to_the_whole_run(tmp_path):
    """A message steered in mid-run must not hide the original request."""
    entries = [
        user("earlier turn"), assistant("done"),            # reviewed by the previous pass
        user("original request: prefer uv over pip"), assistant("working", "Bash"),
        tool_result("ran"),
        user("steer: also check the README"), assistant("finished"),
    ]
    checkpoint(tmp_path).parent.mkdir(parents=True)
    checkpoint(tmp_path).write_text("2")
    out, _ = run(tmp_path, entries)
    assert "USER: original request: prefer uv over pip" in out
    assert "USER: steer: also check the README" in out
    assert "earlier turn" not in out


def test_a_real_pass_advances_the_checkpoint(tmp_path):
    entries = [user("hi"), assistant("ok", "Read")]
    run(tmp_path, entries, dryrun=False)
    assert checkpoint(tmp_path).read_text() == "2"
    # Nothing new since then: the next stop is a logged no-op.
    _, log = run(tmp_path, entries, dryrun=False)
    assert "nothing new since the last pass" in log


def test_dry_run_leaves_the_checkpoint_alone(tmp_path):
    run(tmp_path, [user("hi"), assistant("ok", "Read")])
    assert not checkpoint(tmp_path).exists()


def test_stale_checkpoint_past_the_end_falls_back(tmp_path):
    checkpoint(tmp_path).parent.mkdir(parents=True)
    checkpoint(tmp_path).write_text("99")
    out, _ = run(tmp_path, [user("a"), assistant("b"), user("latest"), assistant("c", "Read")])
    assert "USER: latest" in out
    assert "USER: a\n" not in out


def test_user_messages_get_a_larger_budget_than_assistant_messages(tmp_path):
    long_user = "u" * 10000
    long_assistant = "a" * 10000
    out, _ = run(tmp_path, [user(long_user), assistant(long_assistant, "Read")])
    assert "USER: " + long_user in out  # 10k fits the 16k user default
    assert "ASSISTANT: " + "a" * 4000 + "\n" in out  # assistant text is still clipped at 4k


def test_user_budget_is_configurable(tmp_path):
    out, _ = run(tmp_path, [user("x" * 500), assistant("ok", "Read")], BIGBRAIN_MAINT_USER_CHARS="100")
    assert "USER: " + "x" * 100 + "\n" in out


def test_tool_output_is_dropped_before_user_text_when_over_budget(tmp_path):
    entries = [user("remember: deploys go through argo")]
    for _ in range(20):
        entries += [assistant("step", "Bash"), tool_result("r" * 600)]
    out, _ = run(tmp_path, entries, BIGBRAIN_MAINT_MAX_CHARS="5000")
    assert "USER: remember: deploys go through argo" in out
    assert "RESULT" not in out
    assert "middle of turn omitted" not in out
