"""Vuer runtime helpers that do not import Isaac Lab."""

from __future__ import annotations

import tempfile
from typing import Any

# vuer 0.1.x awaits sleep(0.0) on an idle WebSocket. That yields the event
# loop but not the GIL, so the asyncio thread busy-waits and starves Isaac.
VUER_IDLE_SLEEP_S = 0.002
VUER_WORKSPACE_PREFIX = "maniparena_vuer_ws_"


def empty_vuer_workspace() -> str:
    """Empty temp dir so Vuer does not mount the repo as static files."""
    return tempfile.mkdtemp(prefix=VUER_WORKSPACE_PREFIX)


def patch_vuer_idle_sleep(
    vuer_server: Any, idle_s: float = VUER_IDLE_SLEEP_S
) -> None:
    """Replace vuer.server.sleep(0.0) with a short sleep to avoid GIL spin.

    Match manaenv: patch the ``vuer.server.sleep`` binding (not ``Vuer.uplink``),
    because params_proto duplicates the class in the MRO.
    """
    if getattr(vuer_server, "_maniparena_idle_sleep_patched", False):
        return
    orig = vuer_server.sleep

    async def _sleep(delay=0, result=None):
        if delay == 0 or delay == 0.0:
            delay = idle_s
        return await orig(delay, result=result)

    vuer_server.sleep = _sleep
    vuer_server._maniparena_idle_sleep_patched = True
