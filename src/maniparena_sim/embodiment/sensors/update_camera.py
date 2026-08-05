"""Camera update configs used by Bimanual wrist cameras."""

from __future__ import annotations

import re
from typing import Callable

from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.configclass import configclass
from pxr import Gf, Usd, UsdGeom


class _OpenCVCameraUpdater:
    """Apply OpenCV intrinsics to existing USD camera prims."""

    @staticmethod
    def find_matching_paths(stage, pattern: str) -> list[str]:
        regex = re.compile(f"^{pattern.replace('.*', '[^/]+').replace('*', '[^/]*')}$")
        return [str(prim.GetPath()) for prim in stage.Traverse() if regex.match(str(prim.GetPath()))]

    @staticmethod
    def update_camera_base(prim: Usd.Prim, cfg) -> UsdGeom.Camera:
        camera = UsdGeom.Camera(prim)
        if cfg.clipping_range:
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(*cfg.clipping_range))
        if cfg.focus_distance:
            camera.GetFocusDistanceAttr().Set(cfg.focus_distance)
        if cfg.f_stop:
            camera.GetFStopAttr().Set(cfg.f_stop)
        if cfg.focal_length is not None:
            camera.GetFocalLengthAttr().Set(float(cfg.focal_length))
        return camera

    @staticmethod
    def set_fisheye_via_api(prim_path: str, cfg: "OpenCVFisheyeCameraCfg", stage) -> None:
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

    @staticmethod
    def set_pinhole_via_api(prim_path: str, cfg: "OpenCVPinholeCameraCfg", stage) -> None:
        if not all([cfg.fx, cfg.fy, cfg.cx, cfg.cy, cfg.distortion_coefficients, cfg.width, cfg.height]):
            return
        from isaacsim.sensors.camera import Camera

        camera = Camera(prim_path=prim_path, resolution=(cfg.width, cfg.height))
        camera.initialize()
        camera.set_opencv_pinhole_properties(
            cx=cfg.cx,
            cy=cfg.cy,
            fx=cfg.fx,
            fy=cfg.fy,
            pinhole=cfg.distortion_coefficients,
        )

    @classmethod
    def update_cameras(cls, prim_path: str, cfg, api_fn) -> Usd.Prim | None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        paths = cls.find_matching_paths(stage, prim_path) if "*" in prim_path else [prim_path]
        updated = None
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            cls.update_camera_base(prim, cfg)
            api_fn(path, cfg, stage)
            updated = prim
        return updated

    @classmethod
    def update_fisheye(cls, prim_path: str, cfg: "OpenCVFisheyeCameraCfg", translation=None, orientation=None):
        return cls.update_cameras(prim_path, cfg, cls.set_fisheye_via_api)

    @classmethod
    def update_pinhole(cls, prim_path: str, cfg: "OpenCVPinholeCameraCfg", translation=None, orientation=None):
        return cls.update_cameras(prim_path, cfg, cls.set_pinhole_via_api)


# Isaac Lab ``SpawnerCfg.func`` must be a plain module-level callable; classmethods
# are not preserved by ``@configclass`` and fall back to ``spawn_camera``.
def update_opencv_fisheye_camera(
    prim_path: str, cfg: "OpenCVFisheyeCameraCfg", translation=None, orientation=None,
):
    return _OpenCVCameraUpdater.update_fisheye(prim_path, cfg, translation, orientation)


def update_opencv_pinhole_camera(
    prim_path: str, cfg: "OpenCVPinholeCameraCfg", translation=None, orientation=None,
):
    return _OpenCVCameraUpdater.update_pinhole(prim_path, cfg, translation, orientation)


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
    focal_length: float | None = None
    focus_distance: float | None = None
    f_stop: float | None = None


@configclass
class OpenCVPinholeCameraCfg(PinholeCameraCfg):
    """Update-only config for an existing OpenCV pinhole camera prim.

    OpenCV pinhole layout: ``[k1, k2, p1, p2, k3, k4, k5, k6]``.
    Intrinsics are applied only when fx, fy, cx, cy, width, height, and
    distortion_coefficients are all set; otherwise only base attrs (e.g.
    clipping_range) are updated on the existing prim.
    """

    func: Callable = update_opencv_pinhole_camera
    copy_from_source: bool = False
    width: int | None = None
    height: int | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    distortion_coefficients: list[float] | None = None
    clipping_range: tuple[float, float] | None = None
    focal_length: float | None = None
    focus_distance: float | None = None
    f_stop: float | None = None
