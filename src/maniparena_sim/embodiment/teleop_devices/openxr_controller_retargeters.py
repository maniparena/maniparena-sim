"""OpenXR motion-controller retargeters for VR teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg
from scipy.spatial.transform import Rotation

from maniparena_sim.utils.math_utils import quat_wxyz_normalize, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz


def _quat_wxyz_to_rotation(quat_wxyz: tuple[float, float, float, float] | list[float]) -> Rotation:
    quat = quat_wxyz_normalize(np.asarray(quat_wxyz, dtype=np.float32))
    return Rotation.from_quat(quat_wxyz_to_xyzw(quat))


def _get_controller_inputs(data: dict, bound_controller: DeviceBase.TrackingTarget) -> np.ndarray:
    ctrl = data.get(bound_controller)
    if ctrl is None or np.size(ctrl) == 0:
        return np.zeros(7, dtype=np.float32)
    return np.asarray(
        ctrl[DeviceBase.MotionControllerDataRowIndex.INPUTS.value],
        dtype=np.float32,
    )


class ControllerSe3AbsRetargeter(RetargeterBase):
    """Map motion-controller pose directly to an absolute SE(3) command."""

    def __init__(self, cfg: "ControllerSe3AbsRetargeterCfg"):
        super().__init__(cfg)
        self.bound_controller = cfg.bound_controller
        self._zero_out_xy_rotation = cfg.zero_out_xy_rotation
        self._quat_offset = _quat_wxyz_to_rotation(cfg.motion_controller_quat_offset_wxyz)
        self._previous_pose: np.ndarray | None = None

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        pose = self._get_pose(data)
        position = pose[:3]
        quat_wxyz = pose[3:7]
        rotation = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz_normalize(quat_wxyz)))
        rotation = rotation * self._quat_offset
        if self._zero_out_xy_rotation:
            z, y, x = rotation.as_euler("ZYX")
            rotation = Rotation.from_euler("ZYX", [z, 0.0, 0.0])
        quat_wxyz = quat_xyzw_to_wxyz(rotation.as_quat())
        command = np.concatenate(
            [
                position,
                quat_wxyz,
            ]
        )
        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)

    def _get_pose(self, data: dict) -> np.ndarray:
        ctrl = data.get(self.bound_controller)
        if ctrl is None or np.size(ctrl) == 0:
            if self._previous_pose is not None:
                return self._previous_pose.copy()
            return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        pose = np.asarray(
            ctrl[DeviceBase.MotionControllerDataRowIndex.POSE.value],
            dtype=np.float32,
        )
        self._previous_pose = pose.copy()
        return pose


class ControllerGripperRetargeter(RetargeterBase):
    """Map a motion-controller analog trigger/squeeze to a continuous [0, 1] value.

    0 = released (open), 1 = fully pressed (closed). The planner maps this onto
    the embodiment's gripper joint range; no binary thresholding here.
    """

    def __init__(self, cfg: "ControllerGripperRetargeterCfg"):
        super().__init__(cfg)
        self.bound_controller = cfg.bound_controller
        self.input_name = cfg.input_name

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        inputs = _get_controller_inputs(data, self.bound_controller)
        input_idx = self._resolve_input_idx()
        value = float(inputs[input_idx])
        value = min(1.0, max(0.0, value))
        return torch.tensor([value], dtype=torch.float32, device=self._sim_device)

    def _resolve_input_idx(self) -> int:
        if self.input_name == "squeeze":
            return DeviceBase.MotionControllerInputIndex.SQUEEZE.value
        return DeviceBase.MotionControllerInputIndex.TRIGGER.value


@dataclass
class ControllerSe3AbsRetargeterCfg(RetargeterCfg):
    """Configuration for absolute motion-controller pose retargeting."""

    zero_out_xy_rotation: bool = False
    # Unit WXYZ offset that maps OpenXR controller orientation into the robot EE frame.
    motion_controller_quat_offset_wxyz: tuple[float, float, float, float] = (
        0.5,
        -0.5,
        0.5,
        0.5,
    )
    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = ControllerSe3AbsRetargeter


@dataclass
class ControllerGripperRetargeterCfg(RetargeterCfg):
    """Configuration for motion-controller gripper retargeting (analog [0, 1])."""

    input_name: str = "trigger"
    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = ControllerGripperRetargeter


class ControllerButtonsRetargeter(RetargeterBase):
    """Expose motion-controller button_0/button_1 states as a 2D tensor."""

    def __init__(self, cfg: "ControllerButtonsRetargeterCfg"):
        super().__init__(cfg)
        self.bound_controller = cfg.bound_controller

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        inputs = _get_controller_inputs(data, self.bound_controller)
        button_0 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value])
        button_1 = float(inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value])
        return torch.tensor([button_0, button_1], dtype=torch.float32, device=self._sim_device)


@dataclass
class ControllerButtonsRetargeterCfg(RetargeterCfg):
    """Configuration for motion-controller button retargeting."""

    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_RIGHT
    retargeter_type: type[RetargeterBase] = ControllerButtonsRetargeter


class ControllerJoystickRetargeter(RetargeterBase):
    """Expose motion-controller thumbstick axes as a 2D tensor."""

    def __init__(self, cfg: "ControllerJoystickRetargeterCfg"):
        super().__init__(cfg)
        self.bound_controller = cfg.bound_controller

    def get_requirements(self) -> list[RetargeterBase.Requirement]:
        return [RetargeterBase.Requirement.MOTION_CONTROLLER]

    def retarget(self, data: dict) -> torch.Tensor:
        inputs = _get_controller_inputs(data, self.bound_controller)
        thumbstick_x = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value])
        thumbstick_y = float(inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value])
        return torch.tensor([thumbstick_x, thumbstick_y], dtype=torch.float32, device=self._sim_device)


@dataclass
class ControllerJoystickRetargeterCfg(RetargeterCfg):
    """Configuration for motion-controller thumbstick retargeting."""

    bound_controller: DeviceBase.TrackingTarget = DeviceBase.TrackingTarget.CONTROLLER_LEFT
    retargeter_type: type[RetargeterBase] = ControllerJoystickRetargeter
