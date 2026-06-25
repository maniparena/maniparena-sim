"""CloudXR/OpenXR bimanual VR teleop device (machine-agnostic).

Use this path for Isaac/OpenXR immersive VR control (IsaacLab official CloudXR
runtime). The Vuer/WebXR browser controller-only path lives in
``maniparena_sim.embodiment.teleop_devices.vuer_controller``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import GripperRetargeterCfg, Se3AbsRetargeterCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device

from maniparena_sim.embodiment.teleop_devices.openxr_controller_retargeters import (
    ControllerButtonsRetargeterCfg,
    ControllerGripperRetargeterCfg,
    ControllerJoystickRetargeterCfg,
    ControllerSe3AbsRetargeterCfg,
)


def manaprint(*args, **kwargs):
    print(*args, **kwargs)


@dataclass
class VRTeleopDeviceCfg:
    """Configuration for a generic bimanual OpenXR teleop device.

    Slimmed in the 2026-04 Pink IK migration: motion-controller quaternion
    offset and gripper thresholds (previously duplicated against
    ``VRTeleopSettings`` and the underlying retargeter cfgs) are no longer
    stored here. The underlying ``ControllerSe3*RetargeterCfg`` /
    ``ControllerGripperRetargeterCfg`` defaults apply unless overridden in
    code by callers that build the retargeter cfgs directly.
    """

    sim_device: str | None = None
    zero_out_xy_rotation: bool = False
    use_wrist_rotation: bool = True
    use_wrist_position: bool = True
    enable_visualization: bool = False
    use_motion_controller_pose: bool = False
    use_motion_controller_gripper: bool = False
    use_motion_controller_buttons: bool = False
    use_motion_controller_joystick: bool = False
    motion_controller_linear_joystick_side: str = "left"
    motion_controller_angular_joystick_side: str = "right"
    xr_profile: str = "vr"


class VRTeleopDevice:
    """Thin wrapper that creates an OpenXR teleop controller."""

    device_name = "vr"

    def __init__(self, cfg: VRTeleopDeviceCfg):
        """Store configuration for later controller creation."""
        self.cfg = cfg

    def _apply_xr_profile_override(self) -> None:
        profile_name = str(getattr(self.cfg, "xr_profile", "") or "").strip().lower()
        if not profile_name:
            return
        try:
            from omni.kit.xr.core import XRCore
        except ModuleNotFoundError:
            return

        xr_core = XRCore.get_singleton()
        if xr_core is None:
            return
        try:
            xr_core.request_enable_profile(profile_name)
        except (AttributeError, RuntimeError) as exc:
            manaprint(f"DEBUG: Failed to enable XR profile '{profile_name}': {exc}")
            return

    def _make_se3_cfg(
        self,
        side: str,
    ) -> Se3AbsRetargeterCfg | ControllerSe3AbsRetargeterCfg:
        if self.cfg.use_motion_controller_pose:
            controller_target = self._controller_target(side)
            # ``motion_controller_quat_offset_wxyz`` defaults to (0.5, -0.5,
            # 0.5, 0.5) on the retargeter cfg itself; callers needing a
            # different offset should construct the retargeter cfg directly.
            return ControllerSe3AbsRetargeterCfg(
                bound_controller=controller_target,
                zero_out_xy_rotation=self.cfg.zero_out_xy_rotation,
                sim_device=self.cfg.sim_device or "cpu",
            )
        bound_hand = self._hand_target(side)
        return Se3AbsRetargeterCfg(
            bound_hand=bound_hand,
            zero_out_xy_rotation=self.cfg.zero_out_xy_rotation,
            use_wrist_rotation=self.cfg.use_wrist_rotation,
            use_wrist_position=self.cfg.use_wrist_position,
            enable_visualization=self.cfg.enable_visualization,
            sim_device=self.cfg.sim_device or "cpu",
        )

    def _make_gripper_cfg(
        self,
        side: str,
    ) -> GripperRetargeterCfg | ControllerGripperRetargeterCfg:
        if self.cfg.use_motion_controller_gripper:
            # ``input_name`` defaults to "trigger" on the retargeter cfg; the
            # retargeter emits a continuous [0, 1] value (no thresholding).
            return ControllerGripperRetargeterCfg(
                bound_controller=self._controller_target(side),
                sim_device=self.cfg.sim_device or "cpu",
            )
        bound_hand = self._hand_target(side)
        return GripperRetargeterCfg(
            bound_hand=bound_hand,
            sim_device=self.cfg.sim_device or "cpu",
        )

    def _make_button_cfg(self, side: str) -> ControllerButtonsRetargeterCfg:
        return ControllerButtonsRetargeterCfg(
            bound_controller=self._controller_target(side),
            sim_device=self.cfg.sim_device or "cpu",
        )

    def _make_joystick_cfg(self, side: str) -> ControllerJoystickRetargeterCfg:
        return ControllerJoystickRetargeterCfg(
            bound_controller=self._controller_target(side),
            sim_device=self.cfg.sim_device or "cpu",
        )

    def _base_joystick_sides(self) -> list[str]:
        sides = [
            str(self.cfg.motion_controller_linear_joystick_side),
            str(self.cfg.motion_controller_angular_joystick_side),
        ]
        unique_sides: list[str] = []
        for side in sides:
            if side not in unique_sides:
                unique_sides.append(side)
        return unique_sides

    @staticmethod
    def _hand_target(side: str) -> OpenXRDevice.TrackingTarget:
        if side == "left":
            return OpenXRDevice.TrackingTarget.HAND_LEFT
        return OpenXRDevice.TrackingTarget.HAND_RIGHT

    @staticmethod
    def _controller_target(side: str) -> OpenXRDevice.TrackingTarget:
        if side == "left":
            return OpenXRDevice.TrackingTarget.CONTROLLER_LEFT
        return OpenXRDevice.TrackingTarget.CONTROLLER_RIGHT

    def get_devices_cfg(self, embodiment: object | None = None) -> DevicesCfg:
        """Build the IsaacLab teleop device config."""
        xr_cfg = None
        if embodiment is not None and hasattr(embodiment, "get_xr_cfg"):
            xr_cfg = embodiment.get_xr_cfg()
        retargeters = [
            self._make_se3_cfg("left"),
            self._make_gripper_cfg("left"),
            self._make_se3_cfg("right"),
            self._make_gripper_cfg("right"),
        ]
        if self.cfg.use_motion_controller_buttons:
            retargeters.extend(
                [
                    self._make_button_cfg("left"),
                    self._make_button_cfg("right"),
                ]
            )
        if self.cfg.use_motion_controller_joystick:
            retargeters.extend(self._make_joystick_cfg(side) for side in self._base_joystick_sides())
        return DevicesCfg(
            devices={
                self.device_name: OpenXRDeviceCfg(
                    retargeters=retargeters,
                    sim_device=self.cfg.sim_device or "cpu",
                    xr_cfg=xr_cfg,
                ),
            }
        )

    def create_controller(
        self,
        embodiment: Any = None,
        callbacks: dict | None = None,
    ) -> DeviceBase:
        """Create the actual OpenXR controller instance."""
        devices_cfg = self.get_devices_cfg(embodiment=embodiment)
        controller = create_teleop_device(
            self.device_name,
            devices_cfg.devices,
            callbacks=callbacks,
        )
        self._apply_xr_profile_override()
        return controller
