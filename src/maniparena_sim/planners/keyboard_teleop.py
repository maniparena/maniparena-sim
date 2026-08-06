"""Shared keyboard teleop planner for both robots (bimanual + ex001).

Keyboard emits per-arm delta pose commands. The planner integrates those deltas
into an absolute DiffIK hold target so that when no keys are pressed the last
commanded pose is re-sent every step (prevents arms from sagging under gravity).

Base/chassis wheel velocities are passed through live (zero when idle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from maniparena_sim.embodiment.teleop_devices.keyboard import (
    BimanualSe3Keyboard,
    BimanualSe3KeyboardCfg,
    _get_expected_action_dim,
    _map_action_dim,
)
from maniparena_sim.planners.teleop_base import TeleopPlanner, TeleopSettings
from maniparena_sim.utils.motion_utils import world_to_base_frame

_MOTION_EPS = 1e-8
_BASE_DIM = 2
_LIFT_DIM = 1
_ABS_IK_DIM = 19  # arms+grips+wheels+lift


@dataclass
class KeyboardTeleopSettings(TeleopSettings):
    pass


class KeyboardTeleopPlanner(TeleopPlanner):
    Settings = KeyboardTeleopSettings
    device_name = "keyboard"

    def __init__(self):
        super().__init__()
        self._keyboard_controller = None
        self._expected_action_dim: int | None = None
        self._gripper_targets: dict[int, tuple[float, float]] = {}
        self._gym_env = None
        self._hold_action: torch.Tensor | None = None
        self._ee_frame_names: tuple[str, str] = ("left_ee_frame", "right_ee_frame")
        self._gripper_joint_names: tuple[str, str] = ("left_arm_gripper", "right_arm_gripper")
        self._lift_joint_name: str = "lift_joint"
        self._lift_enabled: bool = False

    def prepare_episode(self, gym_env: Any, obs: Dict[str, Any]) -> None:
        super().prepare_episode(gym_env, obs)
        self._gym_env = gym_env
        self._capture_hold_action()

    def _init_device(self, env: Any) -> None:
        embodiment = getattr(env, "embodiment", None)
        differential_cfg = getattr(embodiment, "diff_drive_keyboard_controller_cfg", None)
        cfg = BimanualSe3KeyboardCfg(
            pos_sensitivity=self.settings.position_scale,
            rot_sensitivity=self.settings.rotation_scale,
            sim_device=str(env.device),
            differential_cfg=differential_cfg,
        )
        self._keyboard_controller = BimanualSe3Keyboard(cfg)
        self._expected_action_dim = _get_expected_action_dim(env)
        self._gripper_targets = self._resolve_gripper_targets(embodiment)
        self._ee_frame_names = self._resolve_ee_frame_names(embodiment)
        self._gripper_joint_names = self._resolve_gripper_joint_names(embodiment)
        self._lift_joint_name = self._resolve_lift_joint_name(embodiment)
        self._lift_enabled = (
            self._expected_action_dim is not None
            and self._expected_action_dim >= _ABS_IK_DIM
            and callable(getattr(embodiment, "get_vr_lift_joint_name", None))
        )
        self._keyboard_controller.add_callback("R", lambda: self.signal_done(success=False))
        self._keyboard_controller.add_callback("H", lambda: self.signal_done(success=True))

    def _resolve_lift_joint_name(self, embodiment: Any) -> str:
        getter = getattr(embodiment, "get_vr_lift_joint_name", None)
        if callable(getter):
            return getter()
        return "lift_joint"

    def _resolve_ee_frame_names(self, embodiment: Any) -> tuple[str, str]:
        getter = getattr(embodiment, "get_vr_ee_frame_names", None)
        if callable(getter):
            return getter()
        return ("left_ee_frame", "right_ee_frame")

    def _resolve_gripper_joint_names(self, embodiment: Any) -> tuple[str, str]:
        getter = getattr(embodiment, "get_vr_gripper_joint_names", None)
        if callable(getter):
            return getter()
        return ("left_arm_gripper", "right_arm_gripper")

    def _resolve_gripper_targets(self, embodiment: Any) -> dict[int, tuple[float, float]]:
        """Map keyboard gripper signs to raw joint targets when the embodiment clamps."""
        clamp = getattr(embodiment, "get_vr_gripper_clamp", None)
        if not callable(clamp):
            return {}
        bounds = clamp() or {}
        names = self._resolve_gripper_joint_names(embodiment)
        out: dict[int, tuple[float, float]] = {}
        if names[0] in bounds:
            out[6] = tuple(float(v) for v in bounds[names[0]])
        if names[1] in bounds:
            out[13] = tuple(float(v) for v in bounds[names[1]])
        return out

    def _apply_gripper_targets(self, action: torch.Tensor) -> torch.Tensor:
        if not self._gripper_targets:
            return action
        out = action.clone()
        for idx, (close_val, open_val) in self._gripper_targets.items():
            if out.shape[-1] > idx:
                out[idx] = open_val if float(out[idx]) > 0.0 else close_val
        return out

    def _normalize_quat(self, quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.linalg.norm(quat).clamp_min(1e-8)

    def _rotvec_to_quat_xyzw(self, rotvec: torch.Tensor) -> torch.Tensor:
        import isaaclab.utils.math as math_utils

        angle = torch.linalg.norm(rotvec)
        if float(angle) < _MOTION_EPS:
            device = rotvec.device
            return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        axis = rotvec / angle
        return math_utils.quat_from_angle_axis(angle, axis).reshape(4)

    def _ee_world_poses(self):
        left = self._gym_env.scene[self._ee_frame_names[0]]
        right = self._gym_env.scene[self._ee_frame_names[1]]
        return (
            left.data.target_pos_w[0, 0, :].clone(),
            left.data.target_quat_w[0, 0, :].clone(),
            right.data.target_pos_w[0, 0, :].clone(),
            right.data.target_quat_w[0, 0, :].clone(),
        )

    def _read_gripper(self, robot, name: str) -> float:
        ids, _ = robot.find_joints(name)
        return float(robot.data.joint_pos[0, ids[0]].item())

    def _read_lift(self, robot) -> float:
        ids, _ = robot.find_joints(self._lift_joint_name)
        return float(robot.data.joint_pos[0, ids[0]].item())

    def _capture_hold_action(self) -> None:
        if self._gym_env is None:
            return
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].clone()
        root_quat = robot.data.root_quat_w[0].clone()
        lpw, lqw, rpw, rqw = self._ee_world_poses()
        lpb, lqb = world_to_base_frame(lpw, lqw, root_pos, root_quat)
        rpb, rqb = world_to_base_frame(rpw, rqw, root_pos, root_quat)
        device = root_pos.device
        lg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[0])], device=device)
        rg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[1])], device=device)
        parts = [
            lpb.reshape(-1),
            self._normalize_quat(lqb.reshape(-1)),
            lg,
            rpb.reshape(-1),
            self._normalize_quat(rqb.reshape(-1)),
            rg,
        ]
        if self._expected_action_dim is not None and self._expected_action_dim > 16:
            parts.append(torch.zeros(_BASE_DIM, dtype=torch.float32, device=device))
        if self._lift_enabled or (
            self._expected_action_dim is not None and self._expected_action_dim >= _ABS_IK_DIM
        ):
            parts.append(
                torch.tensor([self._read_lift(robot)], dtype=torch.float32, device=device)
            )
        self._hold_action = torch.cat(parts)

    def _integrate_arm_delta(
        self,
        hold: torch.Tensor,
        pos_slice: slice,
        quat_slice: slice,
        dpos: torch.Tensor,
        drot: torch.Tensor,
    ) -> None:
        import isaaclab.utils.math as math_utils

        if float(dpos.abs().max()) > _MOTION_EPS:
            hold[pos_slice] = hold[pos_slice] + dpos
        if float(drot.abs().max()) > _MOTION_EPS:
            delta_q = self._rotvec_to_quat_xyzw(drot)
            current_q = hold[quat_slice]
            hold[quat_slice] = self._normalize_quat(
                math_utils.quat_mul(delta_q.unsqueeze(0), current_q.unsqueeze(0)).squeeze(0)
            )

    def _integrate_keyboard_into_hold(self, raw: torch.Tensor) -> torch.Tensor:
        if self._hold_action is None:
            self._capture_hold_action()
        if self._hold_action is None:
            return raw
        hold = self._hold_action.clone()
        self._integrate_arm_delta(hold, slice(0, 3), slice(3, 7), raw[0:3], raw[3:6])
        hold[7:8] = raw[6:7]
        self._integrate_arm_delta(hold, slice(8, 11), slice(11, 15), raw[7:10], raw[10:13])
        hold[15:16] = raw[13:14]
        if hold.shape[-1] > 16 and raw.shape[-1] >= 16:
            hold[16:18] = raw[14:16]
        elif hold.shape[-1] > 16:
            hold[16:18] = 0.0
        # Lift (index 18) stays at the held absolute target; keyboard has no lift axis yet.
        self._hold_action = hold
        return hold

    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw = self._keyboard_controller.advance()
        raw = self._apply_gripper_targets(raw)
        action = self._integrate_keyboard_into_hold(raw)
        if self._expected_action_dim is not None:
            action = _map_action_dim(action, self._expected_action_dim)
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        if self._keyboard_controller is not None:
            self._keyboard_controller.reset()
        self._hold_action = None
