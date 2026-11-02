"""How the Pi extension (hooks/pi/bigbrain.ts) launches the direct runner.

A self-contained (Bun-compiled) Pi reports itself as process.execPath. Spawning that with
the runner path does not run the script: it opens a new Pi session whose prompt is the two
file paths. These tests load the real extension under Node, optionally pretending to be a
compiled Bun binary, and check which binary the runner is spawned with and what the
payload tells the runner about launching a headless Pi.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

EXTENSION = Path(__file__).resolve().parents[1] / "hooks" / "pi" / "bigbrain.ts"
NODE = shutil.which("node")

HARNESS = r"""
const [extPath, fakeBun] = process.argv.slice(2);
if (fakeBun === "1") {
  Object.defineProperty(process.versions, "bun", { value: "1.2.0" });
  process.argv[1] = "/$bunfs/root/pi";
}
const handlers = {};
const pi = { on: (ev, fn) => { handlers[ev] = fn; }, registerTool: () => {} };
(await import(extPath)).default(pi);
const branch = [
  { type: "message", id: "e1", message: { role: "user", content: [{ type: "text", text: "hi" }] } },
  { type: "message", id: "e2", message: { role: "assistant", content: [{ type: "toolCall", name: "read", arguments: {} }] } },
];
await handlers.agent_settled({}, {
  cwd: "/tmp",
  model: undefined,
  sessionManager: { getSessionId: () => "s1", getBranch: () => branch },
});
"""


def _node_strips_types() -> bool:
    if not NODE:
        return False
    probe = subprocess.run([NODE, "--experimental-strip-types", "-e", "const x: number = 1"], capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _node_strips_types(), reason="needs node with type stripping")


def run(tmp_path, fake_bun: bool, path_node: bool = True, **env):
    work = tmp_path / "ext"
    (work / "node_modules" / "typebox").mkdir(parents=True)
    (work / "node_modules" / "typebox" / "package.json").write_text(
        '{"name":"typebox","type":"module","main":"index.js"}'
    )
    (work / "node_modules" / "typebox" / "index.js").write_text(
        "const f = () => ({});\nexport const Type = new Proxy({}, { get: () => f });\n"
    )
    # The extension renders collapsed tool results with pi-tui's Text/truncateToWidth.
    # A stub is enough: these tests exercise the maintenance launch, not rendering.
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

    # A fake `node` on PATH that records how it was invoked, so the test sees exactly
    # which binary the extension spawned and with what arguments.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    spawned = tmp_path / "spawned.txt"
    if path_node:
        fake = bindir / "node"
        fake.write_text(f'#!/bin/sh\necho "node $*" >> "{spawned}"\n')
        fake.chmod(0o755)

    runner = tmp_path / "runner.mjs"
    runner.write_text(f'import {{ appendFileSync }} from "node:fs";\nappendFileSync({json.dumps(str(spawned))}, "execPath " + process.execPath + "\\n");\n')
    hook_dir = tmp_path / "bigbrain-hooks"
    hook_dir.mkdir()

    proc = subprocess.run(
        [NODE, "--experimental-strip-types", "--no-warnings", str(work / "harness.mjs"), str(work / "bigbrain.ts"), "1" if fake_bun else "0"],
        env={
            # No real node directory on PATH: under a fake Bun host the only node the
            # extension may find is the recording one in bindir.
            "PATH": f"{bindir}:/bin:/usr/bin",
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

    # The child is detached; give it a moment to record itself.
    for _ in range(50):
        if spawned.exists() and spawned.read_text().strip():
            break
        time.sleep(0.05)
    payloads = sorted(hook_dir.glob("pi-*.json"))
    payload = json.loads(payloads[0].read_text()) if payloads else None
    log = (tmp_path / "maintenance.log").read_text() if (tmp_path / "maintenance.log").exists() else ""
    return {
        "spawned": spawned.read_text() if spawned.exists() else "",
        "payload": payload,
        "log": log,
        "runner": str(runner),
    }


def test_node_hosted_pi_spawns_runner_with_its_own_node(tmp_path):
    res = run(tmp_path, fake_bun=False)
    assert f"execPath {NODE}" in res["spawned"] or f"execPath {os.path.realpath(NODE)}" in res["spawned"]
    assert "node " not in res["spawned"]  # the PATH node was not needed
    assert res["payload"]["pi_cli"]  # node + CLI script


def test_compiled_pi_spawns_runner_with_node_from_path_not_itself(tmp_path):
    res = run(tmp_path, fake_bun=True)
    assert res["spawned"].startswith(f"node {res['runner']} "), res["spawned"]
    assert "pi_cli" not in res["payload"]  # the /$bunfs/ path is not sent
    assert res["payload"]["pi_node"]  # the compiled binary itself launches headless Pi


def test_compiled_pi_honors_explicit_node_override(tmp_path):
    res = run(tmp_path, fake_bun=True, path_node=False, BIGBRAIN_MAINT_NODE=NODE)
    assert "execPath " in res["spawned"]
    assert "skipped" not in res["log"]


def test_compiled_pi_without_any_node_logs_a_skip_and_writes_no_payload(tmp_path):
    if any(Path(d, "node").exists() for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")):
        pytest.skip("a system node exists in a fallback dir, so the extension finds it")
    res = run(tmp_path, fake_bun=True, path_node=False)
    assert res["spawned"] == ""
    assert res["payload"] is None
    assert "skipped: no node binary found" in res["log"]
