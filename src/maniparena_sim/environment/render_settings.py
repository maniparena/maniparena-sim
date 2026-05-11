"""Load and apply render settings from .settings.usda files.

Two-phase application (mirrors opensdk pattern):

1. RenderCfg(carb_settings=...) at env config time.
2. carb.settings.set() at runtime after gym.make().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


_ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets"
_RENDER_SETTINGS_DIR = _ASSETS_DIR / "rendering_settings"


def get_render_settings_path(name: str) -> Path | None:
    """Return path for ``<name>.settings.usda``, or None."""
    path = _RENDER_SETTINGS_DIR / f"{name}.settings.usda"
    return path if path.exists() else None


def _usda_value_to_python(value: Any) -> Any:
    """Convert USD layer value to a Python native type."""
    if hasattr(value, "Get"):
        value = value.Get()
    if isinstance(value, (bool, int, float, str)):
        return value
    for cast in (float, bool, int):
        try:
            return cast(value)
        except (TypeError, ValueError):
            continue
    return value


def load_render_settings(
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load render settings for *name* (e.g. "green_booth").

    Returns ``(render_cfg_dict, carb_dict)``:

    - render_cfg_dict: {"rtx.xxx.yyy": val} for RenderCfg.
    - carb_dict: {"/rtx/xxx/yyy": val} for carb.settings.

    Returns ({}, {}) when missing or on failure.
    """
    usda_path = get_render_settings_path(name)
    if usda_path is None:
        return {}, {}
    try:
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(str(usda_path))
        if not layer:
            return {}, {}
        raw = layer.customLayerData.get("renderSettings")
        render_dict = raw or {}
        if not isinstance(render_dict, dict) or not render_dict:
            return {}, {}
        render_cfg_dict: dict[str, Any] = {}
        carb_dict: dict[str, Any] = {}
        for key, value in render_dict.items():
            py_val = _usda_value_to_python(value)
            dotted = key.replace("rtx:", "rtx.")
            dotted = dotted.replace(":", ".")
            render_cfg_dict[dotted] = py_val
            carb_dict["/" + key.replace(":", "/")] = py_val
        return render_cfg_dict, carb_dict
    except Exception as exc:
        print(
            f"[RenderSettings] WARNING: "
            f"failed to load '{usda_path}': {exc}"
        )
        return {}, {}


# patch env_cfg before gym.make


def patch_env_cfg_render(
    env_cfg: Any,
    render_cfg_dict: dict[str, Any],
    sim_fps: int = 120,
    render_interval: int = 2,
) -> None:
    """Patch sim timing and render settings on env_cfg.

    Args:
        env_cfg: Environment configuration.
        render_cfg_dict: Loaded render settings dict.
        sim_fps: Simulation frequency (Hz).
        render_interval: Render every N sim steps.
    """
    env_cfg.sim.dt = 1.0 / float(sim_fps)
    env_cfg.sim.render_interval = render_interval
    if not render_cfg_dict:
        return
    import isaaclab.sim as sim_utils

    env_cfg.sim.render_cfg = sim_utils.RenderCfg(
        carb_settings=render_cfg_dict,
    )


#  apply carb settings after gym.make()

def apply_carb_settings(carb_dict: dict[str, Any]) -> None:
    """Write render settings to carb.settings."""
    if not carb_dict:
        return
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        for key, value in carb_dict.items():
            settings.set(key, value)
        print(
            f"[RenderSettings] Applied "
            f"{len(carb_dict)} carb settings"
        )
    except Exception as exc:
        print(
            f"[RenderSettings] WARNING: "
            f"carb.settings failed: {exc}"
        )
