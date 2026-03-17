"""Centralized arena registry bootstrap for maniparena_sim."""

from __future__ import annotations

import importlib

_BOOTSTRAPPED = False


def bootstrap_arena_registry(extra_asset_modules: list[str] | None = None) -> None:
    """Register only modules required by maniparena_sim runtime."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    # Keep imports minimal to avoid side-effects from unused arena assets/devices.
    import isaaclab_arena.assets.background_library  # noqa: F401
    import isaaclab_arena.embodiments  # noqa: F401

    # Register maniparena_sim assets via decorators.
    import maniparena_sim.assets.environments.green_booth  # noqa: F401
    import maniparena_sim.assets.objects.tabletop_objects  # noqa: F401

    if extra_asset_modules:
        for module_name in extra_asset_modules:
            importlib.import_module(module_name)

    _BOOTSTRAPPED = True
