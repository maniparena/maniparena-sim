"""EX001 control callback functions for the ROS2 navigation bridge.

Each callback function processes an incoming ROS message and writes into
the shared action buffer or movement controller.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import partial


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


def control_mock_robot_command(msg, slot_map, action_buffer, verbose=False):
    """Handle unified joint command (/mock_robot_interface/command).

    Writes absolute joint position targets into the env action vector using the
    ActionManager slot layout (``slot_map`` maps joint name -> action slot).
    """
    if action_buffer is None:
        return
    try:
        n = len(msg.position)
        for i, name in enumerate(msg.name):
            slot = slot_map.get(name)
            if slot is None or i >= n:
                continue
            action_buffer[0, slot] = float(msg.position[i])
    except Exception as e:
        if verbose:
            print(f"[ROS] Error processing mock robot command: {e}")


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


def control_chassis_cmd_vel(msg, cmd_vel_buffer=None, verbose=False):
    """Capture chassis velocity command from the navigation stack."""
    linear_x = float(msg.linear.x)
    linear_y = float(msg.linear.y)
    angular_z = float(msg.angular.z)
    if verbose:
        print(f"[ROS] cmd_vel: vx={linear_x:.3f}, vy={linear_y:.3f}, wz={angular_z:.3f}")
    if cmd_vel_buffer is not None:
        cmd_vel_buffer.update(linear_x, linear_y, angular_z)


def noop_control(_msg):
    return None


# ── Registration ─────────────────────────────────────────────────────────────


def fill_control_callbacks(
    control_callbacks,
    slot_map,
    action_buffer,
    cmd_vel_buffer=None,
    verbose=False,
):
    """Register all control callbacks into *control_callbacks* dict."""
    control_callbacks["/mock_robot_interface/command"] = partial(
        control_mock_robot_command,
        slot_map=slot_map,
        action_buffer=action_buffer,
        verbose=verbose,
    )
    control_callbacks["/head_position_controller/commands"] = partial(
        control_head_position_commands,
        slot_map=slot_map,
        action_buffer=action_buffer,
    )
    control_callbacks["/chassis/cmd_vel"] = partial(
        control_chassis_cmd_vel,
        cmd_vel_buffer=cmd_vel_buffer,
        verbose=verbose,
    )
