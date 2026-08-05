"""ex001 VR/vuer teleop planner: DiffIK absolute (gen_method=0) + diff-drive base.

Two backends, selected by ``input_backend``:
  * "vuer"   -> vuer web/Pico WebXR (VuerControllerDevice).
  * "openxr" -> IsaacLab official CloudXR/OpenXR (VRTeleopDevice).

Coordinate handling for the ex001_6r-style arm teleoperation:
  * openxr: the controller-pose quaternion offset is applied inside the
    OpenXR retargeter (see ``openxr_controller_retargeters.py``); the planner
    converts world-frame controller poses into base-frame absolute IK targets.
  * vuer: ``vuer_use_delta_targets`` captures a reference controller+EE pose at
    teleop-start, then drives the arms by axis-remapped (``-z, -x, y``)
    controller deltas added to the reference EE pose in base frame. Rotation is
    a controller-rotation delta mapped through the same axis matrix.

Button scheme (vuer / motion controller):
  * Left-X  -> start teleop (captures the vuer delta reference)
  * Left-Y  -> manually export the current episode as success
  * Right-A -> reset / skip the current episode

Env action layout (18D abs-IK + base, DiffIK diffik slots):
  [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
   L_wheel(1), R_wheel(1)]
Raw device layout (per arm pose+grip interleaved, then buttons, then joystick):
  [L_pos(3), L_quat(4), L_grip(1), R_pos(3), R_quat(4), R_grip(1),
   buttons(4)=[L_A, L_B, R_A, R_B], joystick(4)=[L_x, L_y, R_x, R_y]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import isaaclab.utils.math as math_utils
import torch

from maniparena_sim.planners.teleop_base import TeleopPlanner, TeleopSettings
from maniparena_sim.utils.motion_utils import world_to_base_frame


@dataclass
class Ex001VRTeleopSettings(TeleopSettings):
    input_backend: str = "vuer"  # "vuer" | "openxr"

    # ── joystick mapping: LEFT stick drives the base, RIGHT stick drives lift ──
    use_motion_controller_base: bool = True
    motion_controller_base_deadband: float = 0.15
    motion_controller_base_linear_scale: float = 1.0
    motion_controller_base_angular_scale: float = 1.0
    # Lift is driven by the right-stick Y axis and integrated into an absolute
    # prismatic target. Only the vuer backend wires this up.
    motion_controller_lift_deadband: float = 0.15
    motion_controller_lift_step: float = 0.01  # meters per step at full deflection

    # ── gripper (analog trigger -> continuous joint target) ──
    motion_controller_gripper_input: str = "trigger"

    # ── button mapping (indices into the 4-button raw block) ──
    motion_controller_button_press_threshold: float = 0.5
    motion_controller_left_commit_button_index: int = 0   # Left-X
    motion_controller_left_toggle_button_index: int = 1   # Left-Y
    motion_controller_right_commit_button_index: int = 2  # Right-A
    motion_controller_right_toggle_button_index: int = 3  # Right-B
    motion_controller_left_x_starts_teleop: bool = True
    motion_controller_right_a_resets_episode: bool = True
    motion_controller_left_y_marks_success: bool = True

    # ── absolute-target quat offset (identity unless embodiment overrides) ──
    motion_controller_quat_offset_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    xr_target_base_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ── vuer backend ──
    vuer_host: str = "0.0.0.0"
    vuer_port: int | None = None
    vuer_display_fps: float = 30.0
    vuer_apply_xr_anchor: bool = False
    vuer_use_delta_targets: bool = True
    vuer_delta_position_scale: float = 1.0
    vuer_delta_position_axes: tuple[str, str, str] = ("-z", "-x", "y")
    vuer_delta_rotation_mode: str = "controller"  # "controller" | "hold"

    zero_out_xy_rotation: bool = False


# Raw device slot layout (interleaved per-arm pose+grip, quats are XYZW).
_RAW_L_POS = slice(0, 3)
_RAW_L_QUAT = slice(3, 7)
_RAW_L_GRIP = slice(7, 8)
_RAW_R_POS = slice(8, 11)
_RAW_R_QUAT = slice(11, 15)
_RAW_R_GRIP = slice(15, 16)
_RAW_ARM_DIM = 16
_BUTTON_DIM = 4
# Joystick block is always [L_x, L_y, R_x, R_y].
_JOY_DIM = 4
_JOY_LEFT_X = 0
_JOY_LEFT_Y = 1
_JOY_RIGHT_Y = 3
_BASE_DIM = 2
_LIFT_DIM = 1


class Ex001VRTeleopPlanner(TeleopPlanner):
    Settings = Ex001VRTeleopSettings
    device_name = "ex001_vr"

    def __init__(self):
        super().__init__()
        self._controller = None
        self._gym_env = None
        self._embodiment = None
        self._teleop_active = False
        self._hold_action: torch.Tensor | None = None
        self._gripper_targets: dict[int, tuple[float, float]] = {}
        self._gripper_joint_names: tuple[str, str] = ("left_arm_gripper", "right_arm_gripper")
        self._ee_frame_names: tuple[str, str] = ("left_ee_frame", "right_ee_frame")
        self._base_control_enabled = False
        self._lift_enabled = False
        self._lift_joystick_driven = False
        self._lift_joint_name = "lift_joint"
        self._lift_limits = (0.0, 0.78)
        self._lift_target: float | None = None
        # Lift position captured alongside the arm reference/hold pose. Used to
        # keep the arm IK target riding with the lift (see _lift_compensation).
        self._lift_ref: float | None = None
        self._prev_buttons: torch.Tensor | None = None
        self._abs_quat_offset: torch.Tensor | None = None
        self._vuer_delta_reference: dict[str, dict[str, torch.Tensor]] = {}
        self._vuer_axis_cache: dict[tuple, torch.Tensor] = {}

    # ── setup / device ────────────────────────────────────────────────────
    def _init_device(self, env: Any) -> None:
        embodiment = env.embodiment
        self._embodiment = embodiment
        backend = str(self.settings.input_backend).strip().lower()
        sim_device = str(env.device)
        self._ee_frame_names = embodiment.get_vr_ee_frame_names()
        self._gripper_joint_names = embodiment.get_vr_gripper_joint_names()
        self._gripper_targets = self._resolve_gripper_targets(embodiment)
        self._base_control_enabled = bool(self.settings.use_motion_controller_base) and (
            getattr(embodiment, "diff_drive_keyboard_controller_cfg", None) is not None
        )
        # The lift term is always present in the AbsIK action vector (both vuer
        # and openxr use the 19D config), so it must always be emitted. Only the
        # vuer right joystick *drives* it; under openxr lift simply holds.
        lift_getter = getattr(embodiment, "get_vr_lift_joint_name", None)
        self._lift_enabled = callable(lift_getter)
        self._lift_joystick_driven = self._lift_enabled and backend == "vuer"
        if self._lift_enabled:
            self._lift_joint_name = lift_getter()
            limits_getter = getattr(embodiment, "get_vr_lift_limits", None)
            if callable(limits_getter):
                lo, hi = limits_getter()
                self._lift_limits = (float(lo), float(hi))
        self._abs_quat_offset = self._resolve_abs_quat_offset(env, embodiment)
        callbacks = {"START": self._start, "STOP": self._stop, "RESET": self._reset_episode}
        # Joysticks emit a fixed [L_x, L_y, R_x, R_y] block: left stick drives the
        # base, right stick drives lift.
        need_joysticks = self._base_control_enabled or self._lift_enabled
        if backend == "vuer":
            from maniparena_sim.embodiment.teleop_devices.vuer_controller import (
                VuerControllerDevice, VuerControllerDeviceCfg,
            )
            cfg = VuerControllerDeviceCfg(
                sim_device=sim_device,
                host=self.settings.vuer_host,
                port=self.settings.vuer_port,
                display_fps=self.settings.vuer_display_fps,
                apply_xr_anchor=bool(self.settings.vuer_apply_xr_anchor),
                anchor_pose_provider=self._anchor_pose,
                gripper_input=self.settings.motion_controller_gripper_input,
                include_buttons=True,
                include_joysticks=need_joysticks,
            )
            self._controller = VuerControllerDevice(cfg)
        else:
            from maniparena_sim.embodiment.teleop_devices.cloudxr_device import (
                VRTeleopDevice, VRTeleopDeviceCfg,
            )
            cfg = VRTeleopDeviceCfg(
                sim_device=sim_device,
                zero_out_xy_rotation=self.settings.zero_out_xy_rotation,
                use_motion_controller_pose=True,
                use_motion_controller_gripper=True,
                use_motion_controller_buttons=True,
                use_motion_controller_joystick=need_joysticks,
                # Request both sticks so the fixed [L_x, L_y, R_x, R_y] layout holds.
                motion_controller_linear_joystick_side="left",
                motion_controller_angular_joystick_side="right",
            )
            self._controller = VRTeleopDevice(cfg).create_controller(embodiment=embodiment, callbacks=callbacks)

    def prepare_episode(self, gym_env: Any, obs: Dict[str, Any]) -> None:
        super().prepare_episode(gym_env, obs)
        self._gym_env = gym_env
        self._vuer_delta_reference = {}
        self._lift_target = self._read_lift_position()
        self._capture_hold_action()

    # ── helpers: config resolution ────────────────────────────────────────
    def _resolve_gripper_targets(self, embodiment: Any) -> dict[int, tuple[float, float]]:
        clamp = getattr(embodiment, "get_vr_gripper_clamp", None)
        if not callable(clamp):
            return {}
        bounds = clamp() or {}
        names = embodiment.get_vr_gripper_joint_names()
        out: dict[int, tuple[float, float]] = {}
        if names[0] in bounds:
            out[7] = tuple(float(v) for v in bounds[names[0]])   # diffik L grip slot
        if names[1] in bounds:
            out[15] = tuple(float(v) for v in bounds[names[1]])  # diffik R grip slot
        return out

    def _resolve_abs_quat_offset(self, env: Any, embodiment: Any) -> torch.Tensor:
        device = str(env.device)
        getter = getattr(embodiment, "get_vr_controller_abs_target_quat_offset", None) or getattr(
            embodiment, "get_vr_abs_target_quat_offset", None
        )
        if callable(getter):
            quat = getter()
        else:
            quat = self.settings.motion_controller_quat_offset_xyzw
        return self._normalize_quat(torch.tensor(quat, dtype=torch.float32, device=device))

    def _anchor_pose(self):
        import numpy as np
        if self._gym_env is None:
            return np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)
        robot = self._gym_env.scene["robot"]
        pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype("float32")
        quat = robot.data.root_quat_w[0].detach().cpu().numpy().astype("float32")
        return pos, quat

    # ── math helpers ──────────────────────────────────────────────────────
    def _normalize_quat(self, q: torch.Tensor) -> torch.Tensor:
        return q / torch.linalg.norm(q).clamp_min(1e-8)

    def _apply_abs_quat_offset(self, quat: torch.Tensor) -> torch.Tensor:
        offset = self._abs_quat_offset
        if offset is None:
            return self._normalize_quat(quat)
        return self._normalize_quat(math_utils.quat_mul(quat.unsqueeze(0), offset.unsqueeze(0))[0])

    def _ee_world_poses(self):
        left = self._gym_env.scene[self._ee_frame_names[0]]
        right = self._gym_env.scene[self._ee_frame_names[1]]
        return (
            left.data.target_pos_w[0, 0, :].clone(), left.data.target_quat_w[0, 0, :].clone(),
            right.data.target_pos_w[0, 0, :].clone(), right.data.target_quat_w[0, 0, :].clone(),
        )

    def _read_gripper(self, robot, name):
        ids, _ = robot.find_joints(name)
        return float(robot.data.joint_pos[0, ids[0]].item())

    # ── raw-pose accessors (XYZW controller quats in world frame) ─────────
    def _raw_side_position(self, raw: torch.Tensor, *, side: str) -> torch.Tensor:
        return raw[_RAW_L_POS] if side == "left" else raw[_RAW_R_POS]

    def _raw_side_quat(self, raw: torch.Tensor, *, side: str) -> torch.Tensor:
        return self._normalize_quat(raw[_RAW_L_QUAT] if side == "left" else raw[_RAW_R_QUAT])

    def _raw_side_pose_base(self, raw: torch.Tensor, *, side: str, root_pos, root_quat):
        pos_w = self._raw_side_position(raw, side=side)
        quat_w = raw[_RAW_L_QUAT] if side == "left" else raw[_RAW_R_QUAT]
        pos_base, quat_base = world_to_base_frame(pos_w, quat_w, root_pos, root_quat)
        quat_base = self._apply_abs_quat_offset(quat_base.reshape(-1))
        return pos_base.reshape(-1), self._normalize_quat(quat_base)

    # ── vuer delta-target coordinate path ─────────────────────────
    def _vuer_delta_enabled(self) -> bool:
        return (
            str(self.settings.input_backend).strip().lower() == "vuer"
            and bool(self.settings.vuer_use_delta_targets)
        )

    def _vuer_axis_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        cached = self._vuer_axis_cache.get(key)
        if cached is not None:
            return cached
        rows = []
        for axis in self.settings.vuer_delta_position_axes:
            a = str(axis).strip().lower()
            sign = -1.0 if a.startswith("-") else 1.0
            a = a[1:] if a.startswith(("-", "+")) else a
            if a not in ("x", "y", "z"):
                raise ValueError(f"Invalid vuer delta axis mapping: {axis!r}")
            row = torch.zeros(3, dtype=dtype, device=device)
            row[{"x": 0, "y": 1, "z": 2}[a]] = sign
            rows.append(row)
        mat = torch.stack(rows, dim=0)
        self._vuer_axis_cache[key] = mat
        return mat

    def _map_vuer_position_delta(self, delta: torch.Tensor) -> torch.Tensor:
        mat = self._vuer_axis_matrix(delta.device, delta.dtype)
        return mat @ delta

    def _map_vuer_rotation_delta(self, quat_delta: torch.Tensor) -> torch.Tensor:
        mat = self._vuer_axis_matrix(quat_delta.device, quat_delta.dtype)
        rot_vuer = math_utils.matrix_from_quat(self._normalize_quat(quat_delta).unsqueeze(0))[0]
        rot_robot = mat @ rot_vuer @ mat.transpose(0, 1)
        return self._normalize_quat(math_utils.quat_from_matrix(rot_robot.unsqueeze(0))[0])

    def _capture_vuer_delta_reference(self, raw: torch.Tensor) -> None:
        if not self._vuer_delta_enabled() or self._gym_env is None:
            return
        # Snapshot the lift height so arm targets can ride with the lift
        # carriage instead of staying fixed in the chassis-root frame.
        self._lift_ref = self._read_lift_position()
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].clone()
        root_quat = robot.data.root_quat_w[0].clone()
        lpw, lqw, rpw, rqw = self._ee_world_poses()
        l_tp, l_tq = world_to_base_frame(lpw, lqw, root_pos, root_quat)
        r_tp, r_tq = world_to_base_frame(rpw, rqw, root_pos, root_quat)
        self._vuer_delta_reference = {
            "left": {
                "controller_pos": self._raw_side_position(raw, side="left").detach().clone(),
                "controller_quat": self._raw_side_quat(raw, side="left").detach().clone(),
                "target_pos": l_tp.reshape(-1).detach().clone(),
                "target_quat": self._normalize_quat(l_tq.reshape(-1).detach().clone()),
            },
            "right": {
                "controller_pos": self._raw_side_position(raw, side="right").detach().clone(),
                "controller_quat": self._raw_side_quat(raw, side="right").detach().clone(),
                "target_pos": r_tp.reshape(-1).detach().clone(),
                "target_quat": self._normalize_quat(r_tq.reshape(-1).detach().clone()),
            },
        }

    def _vuer_delta_targets_base(self, raw: torch.Tensor):
        if not self._vuer_delta_enabled():
            return None
        if not self._vuer_delta_reference:
            self._capture_vuer_delta_reference(raw)
        if "left" not in self._vuer_delta_reference or "right" not in self._vuer_delta_reference:
            return None
        scale = float(self.settings.vuer_delta_position_scale)
        rot_mode = str(self.settings.vuer_delta_rotation_mode).strip().lower()
        # Lift travel since the reference was captured. The arms mount on the
        # lift carriage, so the base-frame EE rises with the lift; add the same
        # offset to the target Z to keep the arm command independent of lift.
        lift_offset = 0.0
        if self._lift_enabled and self._lift_ref is not None:
            lift_offset = self._read_lift_position() - self._lift_ref
        out: list[torch.Tensor] = []
        for side in ("left", "right"):
            ref = self._vuer_delta_reference[side]
            ctrl_quat = self._raw_side_quat(raw, side=side)
            raw_pos = self._raw_side_position(raw, side=side)
            raw_delta = raw_pos - ref["controller_pos"].to(raw_pos.device)
            target_pos = ref["target_pos"].to(raw_pos.device) + self._map_vuer_position_delta(raw_delta) * scale
            if lift_offset != 0.0:
                target_pos = target_pos.clone()
                target_pos[2] = target_pos[2] + lift_offset
            ref_quat = ref["target_quat"].to(ctrl_quat.device)
            if rot_mode == "controller":
                ref_ctrl = ref["controller_quat"].to(ctrl_quat.device)
                qd = math_utils.quat_mul(ctrl_quat.unsqueeze(0), math_utils.quat_inv(ref_ctrl.unsqueeze(0)))[0]
                qd = self._map_vuer_rotation_delta(qd)
                target_quat = math_utils.quat_mul(qd.unsqueeze(0), ref_quat.unsqueeze(0))[0]
            else:
                target_quat = ref_quat
            out.extend([target_pos, self._normalize_quat(target_quat)])
        return out[0], out[1], out[2], out[3]

    # ── base wheels (LEFT joystick) ───────────────────────────────────────
    def _wheel_action(self, joystick: torch.Tensor) -> torch.Tensor:
        """Differential-drive wheel velocities from the LEFT joystick.

        Left-stick Y -> linear (forward/back), left-stick X -> angular (yaw).
        """
        cfg = self._embodiment.diff_drive_keyboard_controller_cfg
        if joystick.numel() < _JOY_DIM:
            return joystick.new_zeros(_BASE_DIM)
        thumb_x = joystick[_JOY_LEFT_X]
        thumb_y = joystick[_JOY_LEFT_Y]
        db = float(self.settings.motion_controller_base_deadband)
        if db > 0.0:
            z = joystick.new_tensor(0.0)
            thumb_x = torch.where(torch.abs(thumb_x) < db, z, thumb_x)
            thumb_y = torch.where(torch.abs(thumb_y) < db, z, thumb_y)
        lin = -thumb_y * float(cfg.linear_velocity) * float(self.settings.motion_controller_base_linear_scale)
        ang = -thumb_x * float(cfg.angular_velocity) * float(self.settings.motion_controller_base_angular_scale)
        lw = (lin - 0.5 * ang * float(cfg.wheel_track_width)) / float(cfg.wheel_radius)
        rw = (lin + 0.5 * ang * float(cfg.wheel_track_width)) / float(cfg.wheel_radius)
        return torch.stack([lw, rw])

    # ── lift (RIGHT joystick) ─────────────────────────────────────────────
    def _read_lift_position(self) -> float:
        if not self._lift_enabled or self._gym_env is None:
            return 0.0
        robot = self._gym_env.scene["robot"]
        ids, _ = robot.find_joints(self._lift_joint_name)
        return float(robot.data.joint_pos[0, ids[0]].item())

    def _lift_action(self, joystick: torch.Tensor) -> torch.Tensor:
        """Absolute prismatic lift target.

        Driven by the RIGHT joystick Y under vuer; under openxr it simply holds
        the captured target so the 19D action vector stays consistent.
        """
        if self._lift_target is None:
            self._lift_target = self._read_lift_position()
        if self._lift_joystick_driven and joystick.numel() >= _JOY_DIM:
            axis = float(joystick[_JOY_RIGHT_Y])
            db = float(self.settings.motion_controller_lift_deadband)
            if abs(axis) >= db:
                # Push the stick forward (axis < 0 in the WebXR/OpenXR convention)
                # to raise the lift, pull back to lower it.
                self._lift_target = self._lift_target - axis * float(self.settings.motion_controller_lift_step)
        lo, hi = self._lift_limits
        self._lift_target = max(lo, min(hi, self._lift_target))
        return joystick.new_tensor([self._lift_target])

    # ── gripper / hold ────────────────────────────────────────────────────
    def _apply_gripper_targets(self, action: torch.Tensor) -> torch.Tensor:
        """Map the analog trigger value in [0, 1] to a continuous joint target.

        The raw grip slot carries the trigger amount (0 = released, 1 = fully
        pressed). The gripper defaults to CLOSED when the trigger is released and
        opens as the trigger is pressed: amount 0 -> close_val, amount 1 ->
        open_val, interpolated linearly so it tracks the trigger continuously.
        """
        if not self._gripper_targets:
            return action
        out = action.clone()
        for idx, (close_val, open_val) in self._gripper_targets.items():
            if out.shape[-1] > idx:
                amount = float(out[idx])
                amount = 0.0 if amount < 0.0 else (1.0 if amount > 1.0 else amount)
                out[idx] = close_val + amount * (open_val - close_val)
        return out

    def _capture_hold_action(self) -> None:
        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].clone()
        root_quat = robot.data.root_quat_w[0].clone()
        lpw, lqw, rpw, rqw = self._ee_world_poses()
        lpb, lqb = world_to_base_frame(lpw, lqw, root_pos, root_quat)
        rpb, rqb = world_to_base_frame(rpw, rqw, root_pos, root_quat)
        device = root_pos.device
        lg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[0])], device=device)
        rg = torch.tensor([self._read_gripper(robot, self._gripper_joint_names[1])], device=device)
        parts = [lpb.reshape(-1), self._normalize_quat(lqb.reshape(-1)), lg,
                 rpb.reshape(-1), self._normalize_quat(rqb.reshape(-1)), rg]
        if self._base_control_enabled:
            parts.append(torch.zeros(_BASE_DIM, dtype=torch.float32, device=device))
        if self._lift_enabled:
            parts.append(torch.tensor([self._read_lift_position()], dtype=torch.float32, device=device))
        self._hold_action = torch.cat(parts)

    # ── input parsing ─────────────────────────────────────────────────────
    def _split_raw(self, raw: torch.Tensor):
        pose = raw[:_RAW_ARM_DIM]
        cursor = _RAW_ARM_DIM
        if raw.numel() >= cursor + _BUTTON_DIM:
            buttons = raw[cursor:cursor + _BUTTON_DIM]
            cursor += _BUTTON_DIM
        else:
            buttons = raw.new_zeros(_BUTTON_DIM)
        joystick = raw[cursor:cursor + _JOY_DIM] if raw.numel() >= cursor + _JOY_DIM else raw.new_zeros(_JOY_DIM)
        return pose, buttons, joystick

    def _button_edges(self, buttons: torch.Tensor) -> dict[str, bool]:
        thr = float(self.settings.motion_controller_button_press_threshold)
        pressed = (buttons.detach().cpu() > thr)
        prev = self._prev_buttons if (self._prev_buttons is not None and self._prev_buttons.shape == pressed.shape) else torch.zeros_like(pressed)
        edges = pressed & (~prev)
        self._prev_buttons = pressed.clone()

        def at(i):
            return bool(edges[i]) if 0 <= i < edges.numel() else False

        return {
            "left_commit": at(int(self.settings.motion_controller_left_commit_button_index)),
            "left_toggle": at(int(self.settings.motion_controller_left_toggle_button_index)),
            "right_commit": at(int(self.settings.motion_controller_right_commit_button_index)),
            "right_toggle": at(int(self.settings.motion_controller_right_toggle_button_index)),
        }

    def _handle_buttons(self, edges: dict[str, bool], raw_pose: torch.Tensor) -> None:
        # Left-X starts teleop (captures the vuer delta reference).
        if self.settings.motion_controller_left_x_starts_teleop and edges["left_commit"]:
            self._teleop_active = True
            self._capture_hold_action()
            self._capture_vuer_delta_reference(raw_pose)
            return
        # Right-A resets / skips the episode.
        if self.settings.motion_controller_right_a_resets_episode and edges["right_commit"]:
            self._reset_episode()
            return
        # Left-Y exports the current episode as success.
        if self.settings.motion_controller_left_y_marks_success and edges["left_toggle"]:
            self.signal_done(success=True)
            return
        # Fallback toggle (right-B / left-Y when not bound above).
        if edges["right_toggle"] or edges["left_toggle"]:
            self._teleop_active = not self._teleop_active
            self._capture_hold_action()
            if self._teleop_active:
                self._capture_vuer_delta_reference(raw_pose)

    # ── main step ─────────────────────────────────────────────────────────
    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw = self._controller.advance().clone()
        pose, buttons, joystick = self._split_raw(raw)
        edges = self._button_edges(buttons)
        self._handle_buttons(edges, pose)

        if not self._teleop_active:
            action = self._hold_action.clone()
            action[7:8] = pose[_RAW_L_GRIP]
            action[15:16] = pose[_RAW_R_GRIP]
            if self._base_control_enabled and action.numel() >= 18:
                action[16:18] = self._wheel_action(joystick)
            if self._lift_enabled and action.numel() >= 19:
                action[18:19] = self._lift_action(joystick)
            return self._apply_gripper_targets(action), {"device": self.device_name}

        robot = self._gym_env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].clone()
        root_quat = robot.data.root_quat_w[0].clone()
        delta = self._vuer_delta_targets_base(pose)
        if delta is None:
            lpb, lqb = self._raw_side_pose_base(pose, side="left", root_pos=root_pos, root_quat=root_quat)
            rpb, rqb = self._raw_side_pose_base(pose, side="right", root_pos=root_pos, root_quat=root_quat)
        else:
            lpb, lqb, rpb, rqb = delta
        pos_offset = root_pos.new_tensor(self.settings.xr_target_base_pos_offset)
        parts = [lpb.reshape(-1) + pos_offset, self._normalize_quat(lqb.reshape(-1)), pose[_RAW_L_GRIP],
                 rpb.reshape(-1) + pos_offset, self._normalize_quat(rqb.reshape(-1)), pose[_RAW_R_GRIP]]
        if self._base_control_enabled:
            parts.append(self._wheel_action(joystick))
        if self._lift_enabled:
            parts.append(self._lift_action(joystick))
        return self._apply_gripper_targets(torch.cat(parts)), {"device": self.device_name}

    # ── lifecycle ─────────────────────────────────────────────────────────
    def _start(self) -> None:
        self._teleop_active = False
        self._vuer_delta_reference = {}
        if self._gym_env is not None:
            self._lift_target = self._read_lift_position()
            self._capture_hold_action()

    def _stop(self) -> None:
        self._teleop_active = False
        if self._gym_env is not None:
            self._capture_hold_action()

    def _reset_episode(self) -> None:
        self._teleop_active = False
        self._vuer_delta_reference = {}
        self.signal_done(success=False)

    def _reset_device(self) -> None:
        self._teleop_active = False
        self._prev_buttons = None
        self._vuer_delta_reference = {}
        self._lift_target = None
        self._lift_ref = None
        if self._controller is not None and hasattr(self._controller, "reset"):
            self._controller.reset()
