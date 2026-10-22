"""QUANTA_X1 control callbacks for the SDK ROS2 bridge.

Joint mode writes absolute joint targets via *slot_map*. EE mode writes
``lift_link`` PoseStamped commands into :class:`EePoseCommandBuffer`; the SDK
loop solves those to arm joint targets each step (the env itself only ever
sees joint positions). Gripper commands use the simulation-native ``0–1.89``
range directly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from maniparena_sim.kinematics.se3 import invert, matrix_to_pose, pose_to_matrix
from maniparena_sim.ros.math_utils import (
    compute_relative_pose,
    quat_multiply,
    to_numpy,
)
from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control

_LEFT_ARM_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 7))
_RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 7))


@dataclass
class CmdVelCommandBuffer:
    """Latest ``/chassis/cmd_vel`` command captured from ROS callbacks."""

    linear_x: float = 0.0
    linear_y: float = 0.0
    angular_z: float = 0.0
    sequence: int = 0
    _lock: object = field(default_factory=threading.Lock, repr=False, compare=False)

    def update(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        with self._lock:
            self.linear_x = float(linear_x)
            self.linear_y = float(linear_y)
            self.angular_z = float(angular_z)
            self.sequence += 1

    def as_tuple(self) -> tuple[float, float, float]:
        with self._lock:
            return (self.linear_x, self.linear_y, self.angular_z)

    def read_if_new(self, last_sequence: int) -> tuple[int, tuple[float, float, float]] | None:
        """Return an atomic snapshot only when sequence advances."""
        with self._lock:
            if self.sequence == int(last_sequence):
                return None
            return self.sequence, (self.linear_x, self.linear_y, self.angular_z)


@dataclass
class HomeEeRef:
    """Home gripper pose in the arm-base frame."""

    left_pos: list[float] | None = None
    left_quat: list[float] | None = None
    right_pos: list[float] | None = None
    right_quat: list[float] | None = None

    def clear(self) -> None:
        self.left_pos = self.left_quat = None
        self.right_pos = self.right_quat = None

    def set_side(self, side: str, pos, quat) -> None:
        xyz = [float(v) for v in np.asarray(pos, dtype=np.float32).reshape(-1)[:3]]
        q = [float(v) for v in np.asarray(quat, dtype=np.float32).reshape(-1)[:4]]
        if side == "right":
            self.right_pos, self.right_quat = xyz, q
        else:
            self.left_pos, self.left_quat = xyz, q

    def get(self, side: str) -> tuple[list[float], list[float]] | None:
        if side == "right":
            if self.right_pos is None or self.right_quat is None:
                return None
            return list(self.right_pos), list(self.right_quat)
        if self.left_pos is None or self.left_quat is None:
            return None
        return list(self.left_pos), list(self.left_quat)

    def compose(self, side: str, rel_pos, rel_quat) -> tuple[list[float], list[float]] | None:
        """Compose a home-relative pose onto the cached home."""
        home = self.get(side)
        if home is None:
            return None
        home_pos, home_quat = home
        pos = (np.asarray(home_pos, dtype=np.float32) + np.asarray(rel_pos, dtype=np.float32)).reshape(-1)
        quat = quat_multiply(rel_quat, home_quat)
        return [float(pos[0]), float(pos[1]), float(pos[2])], [float(v) for v in quat]

    def capture_from_robot(self, robot, lift_body_idx: int) -> None:
        lift_pos = to_numpy(robot.data.body_pos_w[0, int(lift_body_idx)])
        lift_quat = to_numpy(robot.data.body_quat_w[0, int(lift_body_idx)])
        body_names = list(robot.data.body_names)
        for side, link in (("left", "left_arm_gripper_base_link"), ("right", "right_arm_gripper_base_link")):
            idx = body_names.index(link)
            pos_b, quat_b = compute_relative_pose(
                to_numpy(robot.data.body_pos_w[0, idx]),
                to_numpy(robot.data.body_quat_w[0, idx]),
                lift_pos,
                lift_quat,
            )
            self.set_side(side, pos_b, quat_b)


@dataclass
class EePoseCommandBuffer:
    """Latest left/right EE pose commands."""

    left: list[float] | None = None
    right: list[float] | None = None
    _lock: object = field(default_factory=threading.Lock, repr=False, compare=False)

    def set_left(self, command: list[float]) -> None:
        with self._lock:
            self.left = [float(v) for v in command]

    def set_right(self, command: list[float]) -> None:
        with self._lock:
            self.right = [float(v) for v in command]

    def snapshot(self) -> tuple[list[float] | None, list[float] | None]:
        with self._lock:
            left = None if self.left is None else list(self.left)
            right = None if self.right is None else list(self.right)
            return left, right

    def clear(self) -> None:
        with self._lock:
            self.left = None
            self.right = None


def _required_slots(slot_map: dict[str, int], names: tuple[str, ...] | list[str]) -> list[int]:
    """Resolve required action slots once while registering callbacks."""
    missing = [name for name in names if name not in slot_map]
    if missing:
        raise ValueError(f"Missing QUANTA_X1 action slots: {missing}")
    return [int(slot_map[name]) for name in names]


def _write_joint_targets(action_buffer, indices, values) -> None:
    if action_buffer is None:
        return
    for joint_idx, value in zip(indices, values):
        action_buffer[0, int(joint_idx)] = float(value)


def _pose_stamped_xyz_xyzw(msg) -> tuple[list[float], list[float]] | None:
    try:
        pos = msg.pose.position
        quat = msg.pose.orientation
        return (
            [float(pos.x), float(pos.y), float(pos.z)],
            [float(quat.x), float(quat.y), float(quat.z), float(quat.w)],
        )
    except Exception:
        return None


def pose_stamped_to_diff_ik_command(msg, robot=None) -> list[float] | None:
    """Map ROS ``PoseStamped`` (XYZW) to ``[x, y, z, qx, qy, qz, qw]``."""
    del robot
    if getattr(getattr(msg, "header", None), "frame_id", "") not in ("", "neutral"):
        return None
    parsed = _pose_stamped_xyz_xyzw(msg)
    if parsed is None:
        return None
    pos_xyz, quat_xyzw = parsed
    pos = np.asarray(pos_xyz, dtype=np.float32).reshape(-1)
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


def control_arm_pose_command(msg, ee_pose_buffer, side: str = "left"):
    """Store a Cartesian pose command."""
    command = pose_stamped_to_diff_ik_command(msg)
    if command is None:
        return
    if side == "right":
        ee_pose_buffer.set_right(command)
    else:
        ee_pose_buffer.set_left(command)


def _scalar_joint(robot, joint_idx: int) -> float:
    value = to_numpy(robot.data.joint_pos[0, int(joint_idx)])
    return float(np.asarray(value).reshape(-1)[0])


def _body_matrix(robot, body_idx: int) -> np.ndarray:
    return pose_to_matrix(
        np.asarray(to_numpy(robot.data.body_pos_w[0, int(body_idx)]), dtype=np.float64),
        np.asarray(to_numpy(robot.data.body_quat_w[0, int(body_idx)]), dtype=np.float64),
    )


def project_ee_pose_commands(
    ee_pose_buffer,
    action_buffer,
    arm_slots,
    robot,
    arm_joint_ids,
    flange_body_ids,
    base_body_id,
    home_ref,
    ik,
) -> list:
    """Solve received EE commands (home-relative, ``lift_link`` frame) to arm joint targets.

    ``arm_slots`` / ``arm_joint_ids`` / ``flange_body_ids`` are indexed
    ``[left, right]``. Sides without a command keep their current joint targets.
    """
    results = [None, None]
    if ee_pose_buffer is None or action_buffer is None or robot is None or home_ref is None:
        return results
    commands = ee_pose_buffer.snapshot()
    to_base = invert(_body_matrix(robot, base_body_id))
    for arm, side in enumerate(("left", "right")):
        command = commands[arm]
        if command is None or len(command) < 7:
            continue
        composed = home_ref.compose(side, command[:3], command[3:7])
        if composed is None:
            continue
        pos_cmd, quat_cmd = composed
        pos_meas, quat_meas = matrix_to_pose(to_base @ _body_matrix(robot, flange_body_ids[arm]))
        q_meas = [_scalar_joint(robot, j) for j in arm_joint_ids[arm]]
        result = ik.solve(
            arm,
            np.asarray(pos_cmd, dtype=np.float64),
            np.asarray(quat_cmd, dtype=np.float64),
            pos_meas,
            quat_meas,
            q_meas,
        )
        results[arm] = result
        _write_joint_targets(action_buffer, arm_slots[arm], result.q)
    return results


def control_arm_joint_commands(msg, joint_indices, action_buffer):
    data = list(msg.data)
    indices = list(joint_indices)
    if len(data) != len(indices):
        raise ValueError(f"Expected {len(indices)} arm commands, got {len(data)}")
    _write_joint_targets(action_buffer, indices, data)


def control_head_position_commands(msg, pitch_slot, yaw_slot, action_buffer):
    """Handle head position controller commands ([head_pitch, head_yaw])."""
    data = list(msg.data)
    if len(data) != 2:
        raise ValueError(f"Expected 2 head commands, got {len(data)}")
    action_buffer[0, pitch_slot] = float(data[0])
    action_buffer[0, yaw_slot] = float(data[1])


def control_lift_position_commands(msg, lift_slot, action_buffer):
    data = list(msg.data)
    if len(data) != 1:
        raise ValueError(f"Expected 1 lift command, got {len(data)}")
    action_buffer[0, lift_slot] = float(data[0])


def control_gripper_commands(msg, joint_slot, action_buffer):
    data = list(msg.data)
    if len(data) != 1:
        raise ValueError(f"Expected 1 gripper command, got {len(data)}")
    action_buffer[0, int(joint_slot)] = float(data[0])


def control_chassis_cmd_vel(msg, cmd_vel_buffer=None, verbose=False):
    """Capture chassis velocity command from the navigation stack."""
    linear_x = float(msg.linear.x)
    linear_y = float(msg.linear.y)
    angular_z = float(msg.angular.z)
    if verbose:
        print(f"[ROS] cmd_vel: vx={linear_x:.3f}, vy={linear_y:.3f}, wz={angular_z:.3f}")
    cmd_vel_buffer.update(linear_x, linear_y, angular_z)


def fill_control_callbacks(
    control_callbacks,
    slot_map,
    action_buffer,
    cmd_vel_buffer=None,
    verbose=False,
    arm_control: str = "ee",
    ee_pose_buffer=None,
):
    """Register SDK control callbacks into *control_callbacks* dict."""
    if cmd_vel_buffer is None:
        raise ValueError("QUANTA_X1 cmd_vel_buffer is required")

    pitch_slot, yaw_slot, lift_slot, left_gripper, right_gripper = _required_slots(
        slot_map,
        [
            "head_pitch_joint",
            "head_yaw_joint",
            "lift_joint",
            "left_arm_gripper",
            "right_arm_gripper",
        ],
    )
    mode = normalize_arm_control(arm_control)
    if mode == "ee":
        if ee_pose_buffer is None:
            raise ValueError("ee arm control needs an EePoseCommandBuffer")
        control_callbacks["/left_arm_cartesian_controller/pose_cmd"] = partial(
            control_arm_pose_command, ee_pose_buffer=ee_pose_buffer, side="left"
        )
        control_callbacks["/right_arm_cartesian_controller/pose_cmd"] = partial(
            control_arm_pose_command, ee_pose_buffer=ee_pose_buffer, side="right"
        )
    else:
        if action_buffer is None:
            raise ValueError("QUANTA_X1 action_buffer is required")
        left_arm = _required_slots(slot_map, _LEFT_ARM_JOINTS)
        right_arm = _required_slots(slot_map, _RIGHT_ARM_JOINTS)
        control_callbacks["/left_arm_joint_controller/commands"] = partial(
            control_arm_joint_commands,
            joint_indices=left_arm,
            action_buffer=action_buffer,
        )
        control_callbacks["/right_arm_joint_controller/commands"] = partial(
            control_arm_joint_commands,
            joint_indices=right_arm,
            action_buffer=action_buffer,
        )
    control_callbacks["/head_position_controller/commands"] = partial(
        control_head_position_commands,
        pitch_slot=pitch_slot,
        yaw_slot=yaw_slot,
        action_buffer=action_buffer,
    )
    control_callbacks["/lift_position_controller/commands"] = partial(
        control_lift_position_commands,
        lift_slot=lift_slot,
        action_buffer=action_buffer,
    )
    control_callbacks["/left_gripper_controller/commands"] = partial(
        control_gripper_commands,
        joint_slot=left_gripper,
        action_buffer=action_buffer,
    )
    control_callbacks["/right_gripper_controller/commands"] = partial(
        control_gripper_commands,
        joint_slot=right_gripper,
        action_buffer=action_buffer,
    )
    control_callbacks["/chassis/cmd_vel"] = partial(
        control_chassis_cmd_vel,
        cmd_vel_buffer=cmd_vel_buffer,
        verbose=verbose,
    )
