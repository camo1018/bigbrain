"""Repeated open/close cycles must not leak file descriptors.

The MCP server closes the store after every op, so each op starts Milvus Lite on
a fresh port and connects a new pymilvus client. pymilvus pools connections and
MilvusClient.close() does not close the gRPC channel, so without an explicit
pool close every op leaked its channel pipes and a long-lived server eventually
failed with "Too many open files".
"""

from __future__ import annotations

import os

import pytest

from bigbrain.config import Config
from bigbrain.store import MemoryStore

_FD_DIR = next((d for d in ("/proc/self/fd", "/dev/fd") if os.path.isdir(d)), None)


def _open_fds() -> int:
    assert _FD_DIR is not None
    return len(os.listdir(_FD_DIR))


@pytest.mark.skipif(_FD_DIR is None, reason="no per-process fd directory on this platform")
def test_store_close_cycles_do_not_leak_fds(tmp_path) -> None:
    store = MemoryStore(Config(home=tmp_path))
    store.store("fd leak probe", "content")
    store.close()

    baseline = _open_fds()
    cycles = 15
    for _ in range(cycles):
        store.count()
        store.close()

    # Allow a little slack for unrelated runtime fds; the leak was 4 per cycle.
    assert _open_fds() - baseline < cycles
