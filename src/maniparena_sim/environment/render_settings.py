"""Load and apply render settings from .settings.usda files.

Two-phase application (mirrors opensdk pattern):

1. ``rtx_global_settings.carb_settings`` at env config time.
2. ``carb.settings.set()`` at runtime after gym.make().
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

    # Isaac Lab removed ``SimulationCfg.render_cfg``; put carb overrides on
    # the env-level Isaac RTX global settings instead.
    rtx = getattr(env_cfg, "rtx_global_settings", None)
    if rtx is not None and hasattr(rtx, "carb_settings"):
        existing = dict(getattr(rtx, "carb_settings", None) or {})
        existing.update(render_cfg_dict)
        rtx.carb_settings = existing
        return

    try:
        from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg

        env_cfg.rtx_global_settings = IsaacRtxRendererGlobalSettingsCfg(
            carb_settings=render_cfg_dict,
        )
    except Exception as exc:
        print(
            f"[RenderSettings] WARNING: "
            f"unable to attach render settings: {exc}"
        )


#  apply carb settings after gym.make()

def ensure_kit_viewport_color_render() -> None:
    """Keep Kit viewport lit when depth-only cameras are present (Sim 6).

    Isaac Lab's RTX renderer sets ``/rtx/sdg/force/disableColorRender`` when a
    camera requests depth without rgb/rgba. It only clears that flag when
    ``/isaaclab/has_gui`` is true. ``--viz kit`` enables KitVisualizer without
    setting ``has_gui``, so a chassis depth camera blacks the main viewport
    even though sensor buffers are fine. Same workaround as Lab's
    ``scripts/demos/sensors/tacsl_sensor.py``.
    """
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        viz = settings.get("/isaaclab/visualizer/types") or ""
        viz_tokens = {t for t in str(viz).replace(",", " ").split() if t}
        if "kit" not in viz_tokens:
            return
        settings.set_bool("/rtx/sdg/force/disableColorRender", False)
        print(
            "[RenderSettings] Cleared /rtx/sdg/force/disableColorRender "
            "for Kit viewport"
        )
    except Exception as exc:
        print(
            f"[RenderSettings] WARNING: "
            f"unable to keep Kit color render: {exc}"
        )


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
