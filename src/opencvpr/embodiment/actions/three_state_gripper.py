"""Three-state gripper action term used by Bimanual."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_STATE_OPEN = 0
_STATE_CLOSE = 1
_STATE_GRASP = 2


class GripperSensor:
    """Wrap a contact sensor and detect simultaneous finger contact."""

    def __init__(
        self,
        env: "ManagerBasedEnv",
        sensor_name: str | None,
        left_regex: str,
        right_regex: str,
        force_threshold: float,
    ):
        self._env = env
        self._sensor_name = sensor_name
        self._force_threshold = force_threshold
        self._left_regex = left_regex
        self._right_regex = right_regex
        self._sensor = None
        self._left_ids = None
        self._right_ids = None

    def _ensure_sensor(self):
        if self._sensor is None and self._sensor_name:
            try:
                self._sensor = self._env.scene[self._sensor_name]
            except KeyError:
                omni.log.warn(f"Contact sensor '{self._sensor_name}' not found.")
        return self._sensor

    def _ensure_body_ids(self, sensor: ContactSensor) -> None:
        if self._left_ids is None:
            self._left_ids, _ = sensor.find_bodies(self._left_regex, preserve_order=True)
        if self._right_ids is None:
            self._right_ids, _ = sensor.find_bodies(self._right_regex, preserve_order=True)

    def both_fingers_in_contact(self, num_envs: int, device: torch.device) -> torch.Tensor:
        sensor = self._ensure_sensor()
        if sensor is None:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._ensure_body_ids(sensor)
        if not self._left_ids or not self._right_ids:
            return torch.zeros(num_envs, dtype=torch.bool, device=device)
        force_mag = torch.norm(sensor.data.net_forces_w, dim=-1)
        left_max = force_mag[:, self._left_ids].max(dim=1).values
        right_max = force_mag[:, self._right_ids].max(dim=1).values
        return (left_max > self._force_threshold) & (right_max > self._force_threshold)


class GripperStateMachine:
    """Simple OPEN/CLOSE/GRASP state machine."""

    def __init__(
        self,
        gripper_sensor: GripperSensor,
        *,
        num_envs: int,
        num_joints: int,
        device: torch.device,
        open_pos: torch.Tensor,
        close_pos: torch.Tensor,
        grasp_hold_ratio: float,
        absolute_input: bool,
        open_threshold: float,
        close_threshold: float,
    ):
        self._sensor = gripper_sensor
        self._num_envs = num_envs
        self._device = device
        self._open_pos = open_pos
        self._close_pos = close_pos
        self._grasp_hold_ratio = grasp_hold_ratio
        self._absolute_input = absolute_input
        self._open_threshold = open_threshold
        self._close_threshold = close_threshold
        self._state = torch.full((num_envs,), _STATE_OPEN, dtype=torch.long, device=device)
        self._grasp_hold_pos = torch.zeros(num_envs, num_joints, device=device)

    def step(self, value: torch.Tensor, joint_pos_getter) -> torch.Tensor:
        if self._absolute_input:
            cmd_open = value >= self._open_threshold
            cmd_close = value <= self._close_threshold
        else:
            cmd_open = value > 0
            cmd_close = value < 0

        is_open = self._state == _STATE_OPEN
        is_close = self._state == _STATE_CLOSE
        is_grasp = self._state == _STATE_GRASP
        self._state[is_open & cmd_close] = _STATE_CLOSE
        self._state[is_close & cmd_open] = _STATE_OPEN

        is_close = self._state == _STATE_CLOSE
        entering_grasp = is_close & self._sensor.both_fingers_in_contact(self._num_envs, self._device)
        if entering_grasp.any():
            self._grasp_hold_pos[entering_grasp] = joint_pos_getter()[entering_grasp] * self._grasp_hold_ratio
        self._state[entering_grasp] = _STATE_GRASP
        self._state[is_grasp & cmd_open] = _STATE_OPEN

        open_target = self._open_pos.unsqueeze(0).expand(self._num_envs, -1)
        close_target = self._close_pos.unsqueeze(0).expand(self._num_envs, -1)
        target = close_target.clone()
        target[self._state == _STATE_GRASP] = self._grasp_hold_pos[self._state == _STATE_GRASP]
        target[self._state == _STATE_OPEN] = open_target[self._state == _STATE_OPEN]
        return target

    def reset(self, env_ids: torch.Tensor) -> None:
        self._state[env_ids] = _STATE_OPEN
        self._grasp_hold_pos[env_ids] = 0.0


class ThreeStateGripperAction(ActionTerm):
    """ActionTerm wrapper around the three-state gripper machine."""

    cfg: "ThreeStateGripperActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ThreeStateGripperActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        sensor = GripperSensor(
            env,
            sensor_name=self.cfg.contact_sensor_name,
            left_regex=self.cfg.left_finger_body_regex,
            right_regex=self.cfg.right_finger_body_regex,
            force_threshold=self.cfg.force_threshold,
        )
        self._sm = GripperStateMachine(
            sensor,
            num_envs=self.num_envs,
            num_joints=self._num_joints,
            device=self.device,
            open_pos=self._parse_command(self.cfg.open_command_expr),
            close_pos=self._parse_command(self.cfg.close_command_expr),
            grasp_hold_ratio=self.cfg.grasp_hold_ratio,
            absolute_input=self.cfg.absolute_input,
            open_threshold=self.cfg.open_threshold,
            close_threshold=self.cfg.close_threshold,
        )

    def _parse_command(self, expr: dict[str, float]) -> torch.Tensor:
        cmd = torch.zeros(self._num_joints, device=self.device)
        indices, _, values = string_utils.resolve_matching_names_values(expr, self._joint_names)
        cmd[indices] = torch.tensor(values, device=self.device)
        return cmd

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._sm.step(actions.squeeze(-1), lambda: self._asset.data.joint_pos[:, self._joint_ids])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._raw_actions[env_ids] = 0.0
        self._sm.reset(env_ids)


@configclass
class ThreeStateGripperActionCfg(ActionTermCfg):
    """Configuration for the three-state gripper."""

    class_type: type[ActionTerm] = ThreeStateGripperAction
    joint_names: list[str] = MISSING
    open_command_expr: dict[str, float] = MISSING
    close_command_expr: dict[str, float] = MISSING
    absolute_input: bool = False
    open_threshold: float = 3.3
    close_threshold: float = 1.0
    contact_sensor_name: str | None = None
    force_threshold: float = 5.0
    grasp_hold_ratio: float = 0.9
    left_finger_body_regex: str = ".*_gripper_left_link"
    right_finger_body_regex: str = ".*_gripper_right_link"
