"""Which messages the Pi extension (hooks/pi/bigbrain.ts) hands to the maintenance pass.

Loads the real extension under Node's type stripping, with `typebox` stubbed and a fake
ExtensionAPI / sessionManager, fires `agent_settled`, and reads the payload it writes for
the (stubbed) direct runner. Needs a Node with --experimental-strip-types (22.6+).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[1] / "hooks" / "pi" / "bigbrain.ts"

HARNESS = r"""
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
const [extPath, scenarioPath, hookDir] = process.argv.slice(2);
const scenario = JSON.parse(readFileSync(scenarioPath, "utf8"));
const handlers = {};
const pi = { on: (ev, fn) => { handlers[ev] = fn; }, registerTool: () => {} };
(await import(extPath)).default(pi);
const out = [];
for (const branch of scenario.passes) {
  const ctx = {
    cwd: "/tmp",
    model: undefined,
    sessionManager: { getSessionId: () => "s1", getBranch: () => branch },
  };
  const before = new Set(readdirSync(hookDir).filter((f) => f.startsWith("pi-")));
  await handlers.agent_settled({}, ctx);
  const added = readdirSync(hookDir).filter((f) => f.startsWith("pi-") && !before.has(f));
  out.push(added.length ? JSON.parse(readFileSync(join(hookDir, added[0]), "utf8")).turn_text : null);
}
process.stdout.write(JSON.stringify(out));
"""


def _node_strips_types() -> bool:
    node = shutil.which("node")
    if not node:
        return False
    probe = subprocess.run(
        [node, "--experimental-strip-types", "-e", "const x: number = 1"], capture_output=True
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _node_strips_types(), reason="needs node with type stripping")


def entry(i, role, text=None, tool=None):
    if role == "user":
        content = [{"type": "text", "text": text}]
    elif role == "toolResult":
        content = [{"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": text}] if text else []
        if tool:
            content.append({"type": "toolCall", "name": tool, "arguments": {"q": "x"}})
    return {"type": "message", "id": f"e{i}", "message": {"role": role, "content": content}}


def run(tmp_path, passes, **env):
    work = tmp_path / "ext"
    (work / "node_modules" / "typebox").mkdir(parents=True)
    (work / "node_modules" / "typebox" / "package.json").write_text(
        '{"name":"typebox","type":"module","main":"index.js"}'
    )
    (work / "node_modules" / "typebox" / "index.js").write_text(
        "const f = () => ({});\n"
        "export const Type = new Proxy({}, { get: () => f });\n"
    )
    # The extension renders collapsed tool results with pi-tui's Text/truncateToWidth.
    # A stub is enough: these tests exercise the maintenance window, not rendering.
    (work / "node_modules" / "@earendil-works" / "pi-tui").mkdir(parents=True)
    (work / "node_modules" / "@earendil-works" / "pi-tui" / "package.json").write_text(
        '{"name":"@earendil-works/pi-tui","type":"module","main":"index.js"}'
    )
    (work / "node_modules" / "@earendil-works" / "pi-tui" / "index.js").write_text(
        "export class Text { constructor() {} render() { return []; } invalidate() {} }\n"
        "export const truncateToWidth = (s) => s;\n"
    )
    (work / "package.json").write_text('{"type":"module"}')
    shutil.copy(EXTENSION, work / "bigbrain.ts")
    (work / "harness.mjs").write_text(HARNESS)
    runner = tmp_path / "runner.mjs"
    runner.write_text("// stub runner: the test reads the payload file instead\n")
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"passes": passes}))
    hook_dir = tmp_path / "bigbrain-hooks"
    hook_dir.mkdir()

    proc = subprocess.run(
        [
            shutil.which("node"),
            "--experimental-strip-types",
            "--no-warnings",
            str(work / "harness.mjs"),
            str(work / "bigbrain.ts"),
            str(scenario),
            str(hook_dir),
        ],
        env={
            "PATH": str(Path(shutil.which("node")).parent),
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),
            "BIGBRAIN_MAINT_RUNNER": str(runner),
            "BIGBRAIN_MAINT_LOG": str(tmp_path / "maintenance.log"),
            **env,
        },
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_first_pass_starts_at_last_user_prompt(tmp_path):
    branch = [
        entry(1, "user", "old request"), entry(2, "assistant", "old answer"),
        entry(3, "user", "new request"), entry(4, "assistant", "ok", tool="read"),
        entry(5, "toolResult", "file body"),
    ]
    [turn] = run(tmp_path, [branch])
    assert "USER: new request" in turn
    assert "old request" not in turn


def test_next_pass_covers_the_whole_run_including_steered_messages(tmp_path):
    first = [entry(1, "user", "warmup"), entry(2, "assistant", "ok", tool="read")]
    second = first + [
        entry(3, "user", "original request: prefer uv over pip"),
        entry(4, "assistant", "working", tool="bash"),
        entry(5, "toolResult", "ran"),
        entry(6, "user", "steer: also check the README"),
        entry(7, "assistant", "done"),
    ]
    _, turn = run(tmp_path, [first, second])
    assert "USER: original request: prefer uv over pip" in turn
    assert "USER: steer: also check the README" in turn
    assert "warmup" not in turn


def test_a_skipped_tool_free_turn_is_picked_up_by_the_next_pass(tmp_path):
    first = [entry(1, "user", "warmup"), entry(2, "assistant", "ok", tool="read")]
    chat = first + [entry(3, "user", "I always want terse answers"), entry(4, "assistant", "got it")]
    work = chat + [entry(5, "user", "now fix the bug"), entry(6, "assistant", "fixed", tool="edit")]
    _, skipped, turn = run(tmp_path, [first, chat, work])
    assert skipped is None  # no tool ran, so no pass...
    assert "USER: I always want terse answers" in turn  # ...but nothing was lost


def test_user_messages_get_a_larger_budget_than_assistant_messages(tmp_path):
    branch = [entry(1, "user", "u" * 10000), entry(2, "assistant", "a" * 10000, tool="read")]
    [turn] = run(tmp_path, [branch])
    assert "USER: " + "u" * 10000 in turn
    assert "ASSISTANT: " + "a" * 4000 + "\n" in turn


def test_tool_output_is_dropped_before_user_text_when_over_budget(tmp_path):
    branch = [entry(0, "user", "remember: deploys go through argo")]
    for i in range(1, 40, 2):
        branch += [entry(i, "assistant", "step", tool="bash"), entry(i + 1, "toolResult", "r" * 600)]
    [turn] = run(tmp_path, [branch], BIGBRAIN_MAINT_MAX_CHARS="5000")
    assert "USER: remember: deploys go through argo" in turn
    assert "RESULT" not in turn
    assert "middle of turn omitted" not in turn
