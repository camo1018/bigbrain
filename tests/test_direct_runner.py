"""Backend selection in hooks/bigbrain-maintenance-direct.mjs.

Runs the real script in dry-run mode, which prints the chosen backend and model and
exits before any model or MCP call. Needs `node`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "hooks" / "bigbrain-maintenance-direct.mjs"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


@pytest.fixture
def fake_pi(tmp_path):
    cli = tmp_path / "pi-cli.js"
    cli.write_text("// stand-in for pi's CLI entry point\n")
    extension = tmp_path / "bigbrain.ts"
    extension.write_text("// stand-in for the bigbrain Pi extension\n")
    return cli, extension


def run(tmp_path, payload, env_file_text="", **env):
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps({"session_id": "t", "turn_text": "USER: hi", **payload}))
    env_file = tmp_path / "env"
    env_file.write_text(env_file_text)
    base = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("BIGBRAIN_", "GEMINI_", "ANTHROPIC_"))
    }
    # An empty PATH apart from node keeps a real `pi` from being discovered.
    base["PATH"] = str(Path(shutil.which("node")).parent)
    base.update(
        BIGBRAIN_MAINT_DRYRUN="1",
        BIGBRAIN_ENV_FILE=str(env_file),
        BIGBRAIN_MAINT_LOG=str(tmp_path / "maintenance.log"),
        HOME=str(tmp_path),
        **env,
    )
    proc = subprocess.run(
        ["node", str(RUNNER), str(payload_file)], env=base, capture_output=True, text=True
    )
    log = tmp_path / "maintenance.log"
    return proc.stdout.splitlines()[0] if proc.stdout else "", log.read_text() if log.exists() else ""


def test_pi_is_the_default_backend(tmp_path, fake_pi):
    cli, ext = fake_pi
    first, _ = run(tmp_path, {"pi_cli": str(cli), "pi_extension": str(ext), "pi_model": "gw/session"})
    assert first == "provider=pi model=gw/session"


def test_env_file_model_overrides_the_session_model(tmp_path, fake_pi):
    cli, ext = fake_pi
    first, _ = run(
        tmp_path,
        {"pi_cli": str(cli), "pi_extension": str(ext), "pi_model": "gw/session"},
        env_file_text="# comment\nexport BIGBRAIN_MAINT_PI_MODEL='gw/background'\n",
    )
    assert first == "provider=pi model=gw/background"


def test_api_key_alone_does_not_switch_away_from_pi(tmp_path, fake_pi):
    cli, ext = fake_pi
    first, _ = run(
        tmp_path,
        {"pi_cli": str(cli), "pi_extension": str(ext)},
        env_file_text="GEMINI_API_KEY=abc\nANTHROPIC_API_KEY=def\n",
    )
    assert first == "provider=pi model=default"


def test_missing_pi_is_logged_not_silently_replaced_by_an_api(tmp_path):
    first, log = run(tmp_path, {}, env_file_text="GEMINI_API_KEY=abc\n")
    assert first == ""
    assert "skipped: the pi CLI was not found" in log


def test_legacy_backend_only_when_selected(tmp_path):
    first, _ = run(
        tmp_path,
        {},
        env_file_text="BIGBRAIN_MAINT_PROVIDER=gemini\nGEMINI_API_KEY=abc\n",
    )
    assert first.startswith("provider=gemini ")


def test_selected_legacy_backend_without_key_is_logged(tmp_path):
    first, log = run(tmp_path, {}, BIGBRAIN_MAINT_PROVIDER="anthropic")
    assert first == ""
    assert "ANTHROPIC_API_KEY is not set" in log
