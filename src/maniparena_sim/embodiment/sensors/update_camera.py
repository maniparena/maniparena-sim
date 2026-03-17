"""Camera update configs used by Bimanual wrist cameras."""

from __future__ import annotations

import re
from typing import Callable

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass
from pxr import Gf, Usd, UsdGeom


def _find_matching_paths(stage, pattern: str) -> list[str]:
    regex = re.compile(f"^{pattern.replace('.*', '[^/]+').replace('*', '[^/]*')}$")
    return [str(prim.GetPath()) for prim in stage.Traverse() if regex.match(str(prim.GetPath()))]


def _update_camera_base(prim: Usd.Prim, cfg) -> UsdGeom.Camera:
    camera = UsdGeom.Camera(prim)
    if cfg.clipping_range:
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*cfg.clipping_range))
    if cfg.focus_distance:
        camera.GetFocusDistanceAttr().Set(cfg.focus_distance)
    if cfg.f_stop:
        camera.GetFStopAttr().Set(cfg.f_stop)
    return camera


def _set_fisheye_via_api(prim_path: str, cfg: "OpenCVFisheyeCameraCfg", stage) -> None:
    if not all([cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.distortion_coefficients, cfg.width, cfg.height]):
        return
    from isaacsim.sensors.camera import Camera

    camera = Camera(prim_path=prim_path, resolution=(cfg.width, cfg.height))
    camera.initialize()
    camera.set_opencv_fisheye_properties(
        cx=cfg.cx,
        cy=cfg.cy,
        fx=cfg.fx,
        fy=cfg.fy,
        fisheye=cfg.distortion_coefficients,
    )


def _update_cameras(prim_path: str, cfg, api_fn) -> Usd.Prim | None:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    paths = _find_matching_paths(stage, prim_path) if "*" in prim_path else [prim_path]
    updated = None
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        _update_camera_base(prim, cfg)
        api_fn(path, cfg, stage)
        updated = prim
    return updated


def update_opencv_fisheye_camera(prim_path: str, cfg: "OpenCVFisheyeCameraCfg", translation=None, orientation=None):
    return _update_cameras(prim_path, cfg, _set_fisheye_via_api)


@configclass
class OpenCVFisheyeCameraCfg(PinholeCameraCfg):
    """Update-only config for an existing OpenCV fisheye camera prim."""

    func: Callable = update_opencv_fisheye_camera
    copy_from_source: bool = False
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_coefficients: list[float] | None = None
    clipping_range: tuple[float, float] | None = None
    focus_distance: float | None = None
    f_stop: float | None = None
