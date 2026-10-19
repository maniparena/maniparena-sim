"""Validated SDK commands and action mapping, independent of ROS and Isaac Sim."""

from __future__ import annotations

from functools import partial

import numpy as np

from maniparena_sim.ros.math_utils import (
    compose_home_relative_pose,
    compose_pose,
    compute_relative_pose,
    to_numpy,
)
from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control
from maniparena_sim.ros.sdk_profiles import (
    HEAD_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
    SdkJointMapping,
    get_sdk_profile,
)


def finite_command(values, size: int) -> np.ndarray | None:
    """Reject malformed commands atomically, preserving the last valid target."""
    try:
        data = np.asarray(list(values), dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if data.shape != (size,) or not np.isfinite(data).all():
        return None
    return data


class SdkControl:
    """Own command state and convert external EE poses into IK targets."""

    def __init__(self, env, robot, actions, profile, arm_control="ee", cmd_vel_timeout_s=0.25):
        self.profile = get_sdk_profile(profile)
        self.mode = normalize_arm_control(arm_control)
        self.mapping = SdkJointMapping(robot, self.profile)
        self.actions = actions
        self.robot = robot
        self.timeout = float(cmd_vel_timeout_s)
        if not np.isfinite(self.timeout) or self.timeout < 0:
            raise ValueError("cmd_vel_timeout_s must be finite and nonnegative")
        if tuple(actions.shape) != (1, env.action_manager.total_action_dim):
            raise ValueError("SDK ROS commands require exactly one environment and a complete action buffer")
        self.joint_slots = {}
        self.pose_slots = []
        offset = 0
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            size = int(term.action_dim)
            if size == 0:
                continue
            slots = list(range(offset, offset + size))
            if name in self.profile.arm_action_terms and self.mode == "ee":
                if size != 7:
                    raise ValueError(f"{name} must have seven pose slots")
                self.pose_slots.append((name, slots))
            else:
                names = list(getattr(term, "_joint_names", ()))
                if len(names) != size:
                    raise ValueError(f"{name} must expose one joint name per action slot")
                self.joint_slots.update(zip(names, slots))
            offset += size
        self.pose_slots = dict(self.pose_slots)
        if self.mode == "ee" and set(self.pose_slots) != {
            *self.profile.arm_action_terms,
        }:
            raise ValueError("EE mode requires arm_action and right_arm_action")
        required = list(self.profile.gripper_joint_names)
        if self.mode == "joint":
            for names in self.profile.arm_joint_names:
                required += list(names)
        if self.profile.mobile:
            required += list(WAIST_JOINT_NAMES + HEAD_JOINT_NAMES + WHEEL_JOINT_NAMES)
        missing = [name for name in required if name not in self.joint_slots]
        if missing:
            raise ValueError(f"SDK action manager is missing joint slots: {missing}")
        self.ee_commands = [None] * len(self.profile.arm_joint_names)
        self.ee_home = [None] * len(self.profile.arm_joint_names)
        self._velocity = (0.0, 0.0, 0.0)
        self._velocity_pending = False
        self._velocity_stamp = None
        self.callbacks = {}
        for side in range(len(self.profile.arm_joint_names)):
            if self.mode == "ee":
                self.callbacks[self.profile.arm_pose_commands[side]] = partial(self.pose_command, side=side)
            else:
                self.callbacks[self.profile.arm_joint_commands[side]] = partial(
                    self.joint_command,
                    names=self.profile.arm_joint_names[side],
                )
            self.callbacks[self.profile.gripper_commands[side]] = partial(
                self.joint_command,
                names=(self.profile.gripper_joint_names[side],),
            )
        if self.profile.mobile:
            self.callbacks["/head_position_controller/commands"] = partial(self.joint_command, names=HEAD_JOINT_NAMES)
            self.callbacks["/waist_forward_position_controller/commands"] = partial(
                self.joint_command, names=WAIST_JOINT_NAMES
            )
            self.callbacks["/chassis/cmd_vel"] = self.velocity_command
        self.reset(robot)

    def joint_command(self, msg, names):
        values = finite_command(getattr(msg, "data", None), len(names))
        if values is not None:
            for name, value in zip(names, values):
                self.actions[0, self.joint_slots[name]] = float(value)

    def pose_command(self, msg, side):
        """Accept the profile's EE pose contract with ROS XYZW quaternions."""
        try:
            frame_id = getattr(getattr(msg, "header", None), "frame_id", "")
            expected_frame = "neutral" if self.profile.ee_pose_init_relative else "base_link"
            if frame_id not in ("", expected_frame):
                return
            p, q = msg.pose.position, msg.pose.orientation
            values = finite_command((p.x, p.y, p.z, q.x, q.y, q.z, q.w), 7)
        except (AttributeError, TypeError):
            return
        if values is None:
            return
        norm = np.linalg.norm(values[3:])
        if norm < 1e-8:
            return
        values[3:] /= norm
        self.ee_commands[side] = values

    def velocity_command(self, msg):
        try:
            values = finite_command((msg.linear.x, msg.linear.y, msg.angular.z), 3)
        except AttributeError:
            return
        if values is not None:
            self._velocity = tuple(float(value) for value in values)
            self._velocity_pending = True

    def latest_cmd_vel(self, sim_time: float) -> tuple[float, float, float]:
        if self._velocity_pending:
            self._velocity_stamp = float(sim_time)
            self._velocity_pending = False
        if self._velocity_stamp is None:
            return (0.0, 0.0, 0.0)
        age = float(sim_time) - self._velocity_stamp
        if age < 0 or (self.timeout > 0 and age > self.timeout):
            self._velocity = (0.0, 0.0, 0.0)
        return self._velocity

    def reset(self, robot=None):
        """Discard old commands and seed every position term from current joints."""
        robot = self.robot if robot is None else robot
        self.actions[:] = 0.0
        indices = {name: i for i, name in enumerate(robot.data.joint_names)}
        for name, slot in self.joint_slots.items():
            if name not in WHEEL_JOINT_NAMES:
                self.actions[0, slot] = float(robot.data.joint_pos[0, indices[name]])
        self._velocity = (0.0, 0.0, 0.0)
        self._velocity_pending = False
        self._velocity_stamp = None
        self.ee_commands = [None] * len(self.profile.arm_joint_names)
        if self.mode == "ee":
            base = self.mapping.base
            for side, flange in enumerate(self.mapping.flanges):
                pos, quat = compute_relative_pose(
                    to_numpy(robot.data.body_pos_w[0, flange]),
                    to_numpy(robot.data.body_quat_w[0, flange]),
                    to_numpy(robot.data.body_pos_w[0, base]),
                    to_numpy(robot.data.body_quat_w[0, base]),
                )
                home = np.concatenate((pos, quat))
                self.ee_home[side] = home
                if self.profile.ee_pose_init_relative:
                    # A reset defines the public zero pose.  The internal
                    # action is still seeded with the current absolute pose.
                    self.ee_commands[side] = np.array(
                        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                        dtype=np.float64,
                    )
                else:
                    self.ee_commands[side] = home.copy()
            self.project_ee_pose_commands(robot)

    def project_ee_pose_commands(self, robot=None):
        if self.mode != "ee":
            return
        robot = self.robot if robot is None else robot
        base = self.mapping.base
        base_pos, base_quat = compute_relative_pose(
            to_numpy(robot.data.body_pos_w[0, base]),
            to_numpy(robot.data.body_quat_w[0, base]),
            to_numpy(robot.data.root_pos_w[0]),
            to_numpy(robot.data.root_quat_w[0]),
        )
        for side, term_name in enumerate(self.profile.arm_action_terms):
            command = self.ee_commands[side]
            if command is None:
                continue
            if self.profile.ee_pose_init_relative:
                home = self.ee_home[side]
                if home is None:
                    continue
                flange_pos, flange_quat = compose_home_relative_pose(
                    home[:3], home[3:], command[:3], command[3:]
                )
            else:
                flange_pos, flange_quat = command[:3], command[3:]
            pos, quat = compose_pose(base_pos, base_quat, flange_pos, flange_quat)
            for slot, value in zip(self.pose_slots[term_name], np.concatenate((pos, quat))):
                self.actions[0, slot] = float(value)
