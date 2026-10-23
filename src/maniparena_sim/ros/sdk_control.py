"""Validated SDK commands and action mapping, independent of ROS and Isaac Sim.

The simulator is always driven with joint-position targets. In ``joint`` mode the
arm commands are written straight into the joint action slots; in ``ee`` mode a
flange pose command is solved to joint targets by :mod:`maniparena_sim.kinematics`.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from maniparena_sim.kinematics.arm_ik import ArmIkController, ArmSpec
from maniparena_sim.kinematics.se3 import invert, matrix_to_pose, pose_to_matrix
from maniparena_sim.ros.math_utils import (
    compose_home_relative_pose,
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


def _body_pose_matrix(robot, body: int) -> np.ndarray:
    return pose_to_matrix(
        np.asarray(to_numpy(robot.data.body_pos_w[0, body]), dtype=np.float64),
        np.asarray(to_numpy(robot.data.body_quat_w[0, body]), dtype=np.float64),
    )


class SdkControl:
    """Own command state and convert external EE poses into joint targets."""

    def __init__(
        self,
        env,
        robot,
        actions,
        profile,
        arm_control="ee",
        cmd_vel_timeout_s=0.25,
        ik: ArmIkController | None = None,
    ):
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
        offset = 0
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            size = int(term.action_dim)
            if size == 0:
                continue
            names = list(getattr(term, "_joint_names", ()))
            if len(names) != size:
                raise ValueError(f"{name} must expose one joint name per action slot")
            self.joint_slots.update(zip(names, range(offset, offset + size)))
            offset += size
        required = list(self.profile.gripper_joint_names)
        for names in self.profile.arm_joint_names:
            required += list(names)
        if self.profile.mobile:
            required += list(WAIST_JOINT_NAMES + HEAD_JOINT_NAMES + WHEEL_JOINT_NAMES)
        missing = [name for name in required if name not in self.joint_slots]
        if missing:
            raise ValueError(f"SDK action manager is missing joint slots: {missing}")
        self.arm_slots = [[self.joint_slots[name] for name in names] for names in self.profile.arm_joint_names]
        if ik is None:
            usd_path = str(robot.cfg.spawn.usd_path)
            ik = ArmIkController.from_usd(
                usd_path,
                [
                    ArmSpec(tuple(names), flange)
                    for names, flange in zip(self.profile.arm_joint_names, self.profile.flange_names)
                ],
            )
        self.ik = ik
        self.ik_results = [None] * len(self.profile.arm_joint_names)
        self.ee_commands = [None] * len(self.profile.arm_joint_names)
        self.ee_home = [None] * len(self.profile.arm_joint_names)
        self._velocity = (0.0, 0.0, 0.0)
        self._velocity_pending = False
        self._velocity_stamp = None
        self.callbacks = self.callbacks_for(self.mode)
        self.reset(robot)

    # -- command callbacks -------------------------------------------------

    def callbacks_for(self, mode: str) -> dict:
        """Topic -> callback for ``mode``; matches ``profile.subscribe_topics(mode)``."""
        mode = normalize_arm_control(mode)
        callbacks = {}
        for side in range(len(self.profile.arm_joint_names)):
            if mode == "ee":
                callbacks[self.profile.arm_pose_commands[side]] = partial(self.pose_command, side=side)
            else:
                callbacks[self.profile.arm_joint_commands[side]] = partial(
                    self.joint_command,
                    names=self.profile.arm_joint_names[side],
                )
            callbacks[self.profile.gripper_commands[side]] = partial(
                self.joint_command,
                names=(self.profile.gripper_joint_names[side],),
            )
        if self.profile.mobile:
            callbacks["/head_position_controller/commands"] = partial(self.joint_command, names=HEAD_JOINT_NAMES)
            callbacks["/waist_forward_position_controller/commands"] = partial(
                self.joint_command, names=WAIST_JOINT_NAMES
            )
            callbacks["/chassis/cmd_vel"] = self.velocity_command
        return callbacks

    def set_mode(self, mode: str) -> dict:
        """Switch arm control mode, holding the current joint targets. Returns the new callbacks."""
        mode = normalize_arm_control(mode)
        if mode != self.mode:
            self.mode = mode
            # Pending poses must not leak across modes; the arms keep the
            # joint targets already in the action buffer until a command arrives.
            self.ee_commands = [None] * len(self.ee_commands)
            self.ik.reset()
            self.callbacks = self.callbacks_for(mode)
        return self.callbacks

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

    # -- state ---------------------------------------------------------------

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
        self.ee_commands = [None] * len(self.ee_commands)
        self.ik_results = [None] * len(self.ik_results)
        self.ik.reset()
        # The reset pose is the public zero pose of home-relative profiles; it
        # is captured in both modes so end poses stay available after a switch.
        base = self.mapping.base
        for side, flange in enumerate(self.mapping.flanges):
            pos, quat = compute_relative_pose(
                to_numpy(robot.data.body_pos_w[0, flange]),
                to_numpy(robot.data.body_quat_w[0, flange]),
                to_numpy(robot.data.body_pos_w[0, base]),
                to_numpy(robot.data.body_quat_w[0, base]),
            )
            self.ee_home[side] = np.concatenate((pos, quat))

    def _arm_joint_values(self, robot, side: int) -> np.ndarray:
        return np.array(
            [float(np.asarray(to_numpy(robot.data.joint_pos[0, i])).reshape(-1)[0]) for i in self.mapping.arms[side]],
            dtype=np.float64,
        )

    def project_ee_pose_commands(self, robot=None):
        """Solve pending EE commands to joint targets (no-op in joint mode)."""
        if self.mode != "ee":
            return
        robot = self.robot if robot is None else robot
        to_base = invert(_body_pose_matrix(robot, self.mapping.base))
        for side, flange in enumerate(self.mapping.flanges):
            command = self.ee_commands[side]
            if command is None:
                continue
            if self.profile.ee_pose_init_relative:
                home = self.ee_home[side]
                flange_pos, flange_quat = compose_home_relative_pose(home[:3], home[3:], command[:3], command[3:])
            else:
                flange_pos, flange_quat = command[:3], command[3:]
            measured_pos, measured_quat = matrix_to_pose(to_base @ _body_pose_matrix(robot, flange))
            result = self.ik.solve(
                side,
                np.asarray(flange_pos, dtype=np.float64),
                np.asarray(flange_quat, dtype=np.float64),
                measured_pos,
                measured_quat,
                self._arm_joint_values(robot, side),
            )
            self.ik_results[side] = result
            for slot, value in zip(self.arm_slots[side], result.q):
                self.actions[0, slot] = float(value)


def make_sdk_control(*args, ik_backend="arm", **kwargs):
    """Select a controller explicitly; importing the legacy backend needs no QP runtime."""
    if ik_backend == "arm":
        return SdkControl(*args, **kwargs)
    if ik_backend == "whole_body_qp":
        from maniparena_sim.ros.whole_body_control import WholeBodySdkControl

        return WholeBodySdkControl(*args, **kwargs)
    raise ValueError(f"Unknown IK backend: {ik_backend!r}; expected arm or whole_body_qp")
