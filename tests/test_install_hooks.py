"""Tests for `bigbrain install-hooks`.

The same hook scripts serve Cursor and Claude Code, but the two hosts disagree on
where config lives and how deeply hook entries nest, so most cases run against both.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from bigbrain.cli import _HOOK_SCRIPTS, _TARGETS, app

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
