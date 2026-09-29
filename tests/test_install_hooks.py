"""Tests for `bigbrain install-hooks`.

The same hook scripts serve Cursor and Claude Code, but the two hosts disagree on
where config lives and how deeply hook entries nest, so most cases run against both.
Pi takes a different shape entirely (extension + AGENTS.md section) and has its own
cases at the bottom.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bigbrain.cli import (
    _AGENTS_BEGIN,
    _AGENTS_END,
    _DIRECT_RUNNER,
    _HOOK_SCRIPTS,
    _REPO_PLACEHOLDER,
    _TARGETS,
    app,
)

runner = CliRunner()

TARGETS = ["cursor", "claude"]


def install(dest, target, *extra):
    result = runner.invoke(
        app, ["install-hooks", "--target", target, "--config-dir", str(dest), *extra]
    )
    assert result.exit_code == 0, result.output
    return result


def commands_in(config: dict, target: str) -> set[str]:
    """Collect every registered hook command, flattening Claude's extra nesting."""
    found = set()
    for entries in config.get("hooks", {}).values():
        for entry in entries:
            if target == "claude":
                found.update(e["command"] for e in entry.get("hooks", []))
            else:
                found.add(entry["command"])
    return found


@pytest.mark.parametrize("target", TARGETS)
def test_installs_scripts_config_rule_and_skills(tmp_path, target):
    install(tmp_path, target)
    spec = _TARGETS[target]

    for name in _HOOK_SCRIPTS:
        script = tmp_path / "hooks" / name
        assert script.is_file()
        assert os.stat(script).st_mode & stat.S_IXUSR, f"{name} is not executable"

    config = json.loads((tmp_path / spec["config_file"]).read_text())
    commands = commands_in(config, target)
    assert any("bigbrain-maintenance.sh" in c for c in commands)
    assert any("bigbrain-mark-substantive.sh" in c for c in commands)

    assert (tmp_path / "rules" / spec["rule_file"]).is_file()
    assert (tmp_path / "skills" / "bigbrain-trim" / "SKILL.md").is_file()


@pytest.mark.parametrize("target", TARGETS)
def test_reinstall_is_idempotent(tmp_path, target):
    install(tmp_path, target)
    first = json.loads((tmp_path / _TARGETS[target]["config_file"]).read_text())
    install(tmp_path, target)
    second = json.loads((tmp_path / _TARGETS[target]["config_file"]).read_text())
    assert first == second


@pytest.mark.parametrize(
    "target, existing",
    [
        (
            "cursor",
            {"version": 1, "hooks": {"stop": [{"command": "./hooks/my-own.sh"}]}},
        ),
        (
            "claude",
            {
                "permissions": {"allow": ["Bash(ls:*)"]},
                "hooks": {
                    "Stop": [{"hooks": [{"type": "command", "command": "/my/own.sh"}]}]
                },
            },
        ),
    ],
)
def test_preserves_existing_config(tmp_path, target, existing):
    config_path = tmp_path / _TARGETS[target]["config_file"]
    config_path.write_text(json.dumps(existing))

    install(tmp_path, target)

    merged = json.loads(config_path.read_text())
    commands = commands_in(merged, target)
    assert {"./hooks/my-own.sh", "/my/own.sh"} & commands, "existing hook was dropped"
    assert any("bigbrain-maintenance.sh" in c for c in commands)
    if target == "claude":
        assert merged["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert config_path.with_suffix(".json.bak").is_file()


@pytest.mark.parametrize("target", TARGETS)
def test_registered_commands_point_at_the_target_dir(tmp_path, target):
    """A non-default --config-dir must not leave hooks pointing at the default tree."""
    install(tmp_path, target)
    config = json.loads((tmp_path / _TARGETS[target]["config_file"]).read_text())
    for command in commands_in(config, target):
        # Cursor resolves a relative command against its config dir; Claude needs an
        # absolute path, which the installer rewrites for a non-default target.
        path = tmp_path / command[2:] if command.startswith("./") else Path(command)
        assert path.is_file(), f"{command} does not resolve to an installed script"


@pytest.mark.parametrize("flag, missing", [("--no-rule", "rules"), ("--no-skills", "skills")])
def test_opt_out_flags(tmp_path, flag, missing):
    install(tmp_path, "cursor", flag)
    assert not (tmp_path / missing).exists()


def test_unknown_target_is_rejected(tmp_path):
    result = runner.invoke(
        app, ["install-hooks", "--target", "emacs", "--config-dir", str(tmp_path)]
    )
    assert result.exit_code == 2


# --- Pi -----------------------------------------------------------------------------


def agents_section(text: str) -> str:
    start, end = text.find(_AGENTS_BEGIN), text.find(_AGENTS_END)
    assert start != -1 and end > start, "managed section missing"
    return text[start : end + len(_AGENTS_END)]


def test_pi_installs_extension_runner_agents_and_skills(tmp_path):
    install(tmp_path, "pi")

    extension = tmp_path / "extensions" / "bigbrain.ts"
    assert extension.is_file()
    runner_script = tmp_path / "hooks" / _DIRECT_RUNNER
    assert runner_script.is_file()
    assert os.stat(runner_script).st_mode & stat.S_IXUSR
    # Only the extension may live under extensions/: Pi loads everything there.
    assert [p.name for p in (tmp_path / "extensions").iterdir()] == ["bigbrain.ts"]

    body = agents_section((tmp_path / "AGENTS.md").read_text())
    assert "memory_recall" in body
    assert "alwaysApply" not in body, "Cursor frontmatter leaked into AGENTS.md"

    assert (tmp_path / "skills" / "bigbrain-trim" / "SKILL.md").is_file()


def test_pi_extension_has_repo_path_stamped(tmp_path):
    install(tmp_path, "pi")
    text = (tmp_path / "extensions" / "bigbrain.ts").read_text()
    assert _REPO_PLACEHOLDER not in text
    assert str(Path(__file__).resolve().parents[1]) in text


@pytest.mark.parametrize(
    "existing",
    [
        None,
        "# My instructions\n\nAlways answer in haiku.\n",
        f"# Mine\n\n{_AGENTS_BEGIN}\nstale bigbrain text\n{_AGENTS_END}\n\n# More of mine\n",
    ],
    ids=["no-file", "unrelated-content", "stale-section"],
)
def test_pi_agents_md_merge_preserves_content_and_is_idempotent(tmp_path, existing):
    agents_md = tmp_path / "AGENTS.md"
    if existing is not None:
        agents_md.write_text(existing)

    install(tmp_path, "pi")
    first = agents_md.read_text()
    assert first.count(_AGENTS_BEGIN) == 1
    assert "stale bigbrain text" not in first
    if existing:
        for line in existing.splitlines():
            if line.startswith("#") and "bigbrain" not in line:
                assert line in first, f"user content dropped: {line!r}"

    install(tmp_path, "pi")
    assert agents_md.read_text() == first


def test_pi_opt_out_flags(tmp_path):
    install(tmp_path, "pi", "--no-rule", "--no-skills")
    assert not (tmp_path / "AGENTS.md").exists()
    assert not (tmp_path / "skills").exists()
    assert (tmp_path / "extensions" / "bigbrain.ts").is_file()


def test_pi_install_does_not_demand_an_api_key(tmp_path, monkeypatch):
    """With `pi` on PATH the pass runs on Pi's own providers, so no key note is printed."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("BIGBRAIN_ENV_FILE", str(tmp_path / "missing-env"))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_pi = fake_bin / "pi"
    fake_pi.write_text("#!/bin/sh\n")
    fake_pi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")

    result = runner.invoke(
        app, ["install-hooks", "--target", "pi", "--config-dir", str(tmp_path / "pi")]
    )
    assert result.exit_code == 0, result.output
    assert "GEMINI_API_KEY" not in result.output


def test_pi_extension_hands_runner_its_cli_and_model(tmp_path):
    install(tmp_path, "pi")
    text = (tmp_path / "extensions" / "bigbrain.ts").read_text()
    for field in ("pi_cli", "pi_node", "pi_extension", "pi_model"):
        assert field in text, f"payload field {field} missing"
