"""EX001 control callbacks for the SDK ROS2 bridge.

Nav uses ``ActionsCfgNav`` (one ``joint_pos`` term + wheel velocity). Callbacks
write absolute joint targets into the shared action buffer via *slot_map*
(``joint_name -> action slot``). Gripper commands stay on the SDK ``0–4.5``
range and are converted to sim joint units here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import partial

from maniparena_sim.ros.ex001_sdk_topics import sdk_gripper_to_sim

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


def _slots_for_names(slot_map: dict[str, int], names: tuple[str, ...] | list[str]) -> list[int]:
    slots: list[int] = []
    for name in names:
        slot = slot_map.get(name)
        if slot is not None:
            slots.append(int(slot))
    return slots


def _write_joint_targets(action_buffer, indices, values) -> None:
    if action_buffer is None:
        return
    for joint_idx, value in zip(indices, values):
        if 0 <= int(joint_idx) < action_buffer.shape[1]:
            action_buffer[0, int(joint_idx)] = float(value)


def control_arm_joint_commands(msg, joint_indices, action_buffer):
    try:
        data = list(msg.data)
    except Exception:
        return
    indices = list(joint_indices)
    if not indices or len(data) < len(indices):
        return
    _write_joint_targets(action_buffer, indices, data[: len(indices)])


def control_head_position_commands(msg, slot_map, action_buffer):
    """Handle head position controller commands ([head_pitch, head_yaw])."""
    try:
        data = list(msg.data)
    except Exception:
        return
    if len(data) < 2 or action_buffer is None:
        return
    pitch_slot = slot_map.get("head_pitch_joint")
    yaw_slot = slot_map.get("head_yaw_joint")
    if pitch_slot is not None:
        action_buffer[0, pitch_slot] = float(data[0])
    if yaw_slot is not None:
        action_buffer[0, yaw_slot] = float(data[1])


def control_lift_position_commands(msg, slot_map, action_buffer):
    try:
        data = list(msg.data)
    except Exception:
        return
    if not data or action_buffer is None:
        return
    lift_slot = slot_map.get("lift_joint")
    if lift_slot is not None:
        action_buffer[0, lift_slot] = float(data[0])


def control_gripper_commands(msg, joint_slot, action_buffer):
    try:
        data = list(msg.data)
    except Exception:
        return
    if not data or joint_slot is None or action_buffer is None:
        return
    action_buffer[0, int(joint_slot)] = float(sdk_gripper_to_sim(data[0]))


def control_chassis_cmd_vel(msg, cmd_vel_buffer=None, verbose=False):
    """Capture chassis velocity command from the navigation stack."""
    linear_x = float(msg.linear.x)
    linear_y = float(msg.linear.y)
    angular_z = float(msg.angular.z)
    if verbose:
        print(f"[ROS] cmd_vel: vx={linear_x:.3f}, vy={linear_y:.3f}, wz={angular_z:.3f}")
    if cmd_vel_buffer is not None:
        cmd_vel_buffer.update(linear_x, linear_y, angular_z)


def fill_control_callbacks(
    control_callbacks,
    slot_map,
    action_buffer,
    cmd_vel_buffer=None,
    verbose=False,
):
    """Register SDK control callbacks into *control_callbacks* dict."""
    left_arm = _slots_for_names(slot_map, _LEFT_ARM_JOINTS)
    right_arm = _slots_for_names(slot_map, _RIGHT_ARM_JOINTS)
    left_gripper = slot_map.get("left_arm_gripper")
    right_gripper = slot_map.get("right_arm_gripper")

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
        slot_map=slot_map,
        action_buffer=action_buffer,
    )
    control_callbacks["/lift_position_controller/commands"] = partial(
        control_lift_position_commands,
        slot_map=slot_map,
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
