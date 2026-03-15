"""VR teleop planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from opencvpr.embodiment.teleop_devices.keyboard import _get_expected_action_dim, _map_action_dim
from opencvpr.embodiment.teleop_devices.vr import VRTeleopDevice, VRTeleopDeviceCfg
from opencvpr.planners.teleop_base import TeleopPlanner, TeleopSettings


@dataclass
class VRTeleopSettings(TeleopSettings):
    delta_pos_scale_factor: float = 15.0
    delta_rot_scale_factor: float = 10.0
    alpha_pos: float = 0.5
    alpha_rot: float = 0.5
    zero_out_xy_rotation: bool = False
    use_wrist_rotation: bool = True
    use_wrist_position: bool = True
    enable_visualization: bool = False
    disable_all_cameras_in_vr: bool = True


class VRTeleopPlanner(TeleopPlanner):
    Settings = VRTeleopSettings
    device_name = "vr"

    def __init__(self):
        super().__init__()
        self._vr_controller = None
        self._expected_action_dim = None
        self._teleop_active = False

    def _init_device(self, env: Any) -> None:
        embodiment = env.embodiment
        if self.settings.disable_all_cameras_in_vr:
            embodiment.enable_cameras = False
            embodiment.camera_config = None
        embodiment.action_config = embodiment.ActionsCfg()
        cfg = VRTeleopDeviceCfg(
            sim_device=str(env.device),
            delta_pos_scale_factor=self.settings.delta_pos_scale_factor,
            delta_rot_scale_factor=self.settings.delta_rot_scale_factor,
            alpha_pos=self.settings.alpha_pos,
            alpha_rot=self.settings.alpha_rot,
            zero_out_xy_rotation=self.settings.zero_out_xy_rotation,
            use_wrist_rotation=self.settings.use_wrist_rotation,
            use_wrist_position=self.settings.use_wrist_position,
            enable_visualization=self.settings.enable_visualization,
        )
        callbacks = {"START": self._start_teleop, "STOP": self._stop_teleop, "RESET": self._reset_episode}
        self._vr_controller = VRTeleopDevice(cfg).create_controller(embodiment=embodiment, callbacks=callbacks)
        self._expected_action_dim = _get_expected_action_dim(env)

    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw = self._vr_controller.advance()
        action = _map_action_dim(raw, self._expected_action_dim) if self._expected_action_dim is not None else raw
        action = action.clone()
        action[0:3] *= float(self.settings.position_scale)
        action[3:6] *= float(self.settings.rotation_scale)
        action[7:10] *= float(self.settings.position_scale)
        action[10:13] *= float(self.settings.rotation_scale)
        if not self._teleop_active:
            action.zero_()
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        self._teleop_active = False
        if self._vr_controller is not None:
            self._vr_controller.reset()

    def _start_teleop(self) -> None:
        self._teleop_active = True

    def _stop_teleop(self) -> None:
        self._teleop_active = False

    def _reset_episode(self) -> None:
        self._teleop_active = False
        self.signal_done(success=False)
