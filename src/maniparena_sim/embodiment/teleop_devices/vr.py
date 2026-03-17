"""OpenXR teleop helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3RelRetargeterCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device


@dataclass
class VRTeleopDeviceCfg:
    sim_device: str | None = None
    delta_pos_scale_factor: float = 15.0
    delta_rot_scale_factor: float = 10.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5
    zero_out_xy_rotation: bool = False
    use_wrist_rotation: bool = True
    use_wrist_position: bool = True
    enable_visualization: bool = False


class VRTeleopDevice:
    """Thin wrapper that builds a bimanual OpenXR teleop controller."""

    device_name = "vr"

    def __init__(self, cfg: VRTeleopDeviceCfg):
        self.cfg = cfg

    def _make_se3_cfg(self, bound_hand: OpenXRDevice.TrackingTarget) -> Se3RelRetargeterCfg:
        return Se3RelRetargeterCfg(
            bound_hand=bound_hand,
            zero_out_xy_rotation=self.cfg.zero_out_xy_rotation,
            use_wrist_rotation=self.cfg.use_wrist_rotation,
            use_wrist_position=self.cfg.use_wrist_position,
            delta_pos_scale_factor=self.cfg.delta_pos_scale_factor,
            delta_rot_scale_factor=self.cfg.delta_rot_scale_factor,
            alpha_pos=self.cfg.alpha_pos,
            alpha_rot=self.cfg.alpha_rot,
            enable_visualization=self.cfg.enable_visualization,
            sim_device=self.cfg.sim_device or "cpu",
        )

    def _make_gripper_cfg(self, bound_hand: OpenXRDevice.TrackingTarget) -> GripperRetargeterCfg:
        return GripperRetargeterCfg(bound_hand=bound_hand, sim_device=self.cfg.sim_device or "cpu")

    def get_devices_cfg(self, embodiment: object | None = None) -> DevicesCfg:
        xr_cfg = embodiment.get_xr_cfg() if embodiment is not None and hasattr(embodiment, "get_xr_cfg") else None
        return DevicesCfg(
            devices={
                self.device_name: OpenXRDeviceCfg(
                    retargeters=[
                        self._make_se3_cfg(OpenXRDevice.TrackingTarget.HAND_LEFT),
                        self._make_gripper_cfg(OpenXRDevice.TrackingTarget.HAND_LEFT),
                        self._make_se3_cfg(OpenXRDevice.TrackingTarget.HAND_RIGHT),
                        self._make_gripper_cfg(OpenXRDevice.TrackingTarget.HAND_RIGHT),
                    ],
                    sim_device=self.cfg.sim_device or "cpu",
                    xr_cfg=xr_cfg,
                )
            }
        )

    def create_controller(self, embodiment: Any = None, callbacks: dict | None = None) -> DeviceBase:
        devices_cfg = self.get_devices_cfg(embodiment=embodiment)
        return create_teleop_device(self.device_name, devices_cfg.devices, callbacks=callbacks)
