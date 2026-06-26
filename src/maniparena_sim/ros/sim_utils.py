"""Simulation ↔ ROS utility helpers.

Consolidates camera intrinsics cache, ROS time conversion, lightweight
robot-state snapshots, and pose extraction helpers.
"""

from __future__ import annotations

import numpy as np
import torch

from maniparena_sim.utils.debug_print import manaprint
from maniparena_sim.ros.math_utils import to_numpy

# ── Camera Intrinsics Cache ──────────────────────────────────────────────────

camera_cache: dict = {}
_warned_applied_torque_robot_ids: set[int] = set()


def init_camera_cache(env, camera_config: dict) -> None:
    """Populate ``camera_cache`` with intrinsics for each camera in the scene."""
    camera_cache.clear()
    for name, cfg in camera_config.items():
        entity = cfg.get("scene_entity", name)
        if entity not in env.scene.keys():
            continue
        sensor = env.scene[entity]
        m = sensor.data.intrinsic_matrices[0].cpu().numpy()
        clip = sensor.cfg.spawn.clipping_range
        camera_cache[name] = {
            "fx": m[0, 0],
            "fy": m[1, 1],
            "cx": m[0, 2],
            "cy": m[1, 2],
            "width": sensor.cfg.width,
            "height": sensor.cfg.height,
            "near_clip": clip[0],
            "far_clip": clip[1],
        }
    if camera_cache:
        manaprint(f"INFO: Camera cache initialized: {list(camera_cache.keys())}")


# ── ROS Time ─────────────────────────────────────────────────────────────────


def get_ros_time(sim_time: float | None = None):
    """Return a ``builtin_interfaces/Time`` message from *sim_time* or wall clock."""
    from builtin_interfaces.msg import Time as RosTime

    if sim_time is None:
        import time

        sim_time = time.time()

    msg = RosTime()
    msg.sec = int(sim_time)
    msg.nanosec = int((sim_time - int(sim_time)) * 1e9)
    return msg


def build_robot_state_snapshot(env, robot, imu_sensor=None) -> dict:
    """Build a lightweight observation-like dict from live sim tensors.

    This avoids calling ``observation_manager.compute()`` for high-rate ROS
    topics that only need robot state and optional IMU data.
    """

    policy: dict[str, torch.Tensor] = {}
    root_state_w = robot.data.root_state_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    body_pose_w = robot.data.body_pose_w

    env_origins = env.scene.env_origins
    policy["root_pos_w"] = root_state_w[:, 0:3] - env_origins
    policy["root_quat_w"] = root_state_w[:, 3:7]
    policy["root_lin_vel_w"] = root_state_w[:, 7:10]
    policy["root_ang_vel_w"] = root_state_w[:, 10:13]
    policy["joint_positions"] = joint_pos
    policy["joint_velocities"] = joint_vel
    # We don't have measured efforts cheaply here. Reuse currently applied torque.
    applied_torque = robot.data.applied_torque
    if applied_torque is None:
        robot_id = id(robot)
        if robot_id not in _warned_applied_torque_robot_ids:
            manaprint("WARNING: applied_torque unavailable, using zeros for joint_efforts")
            _warned_applied_torque_robot_ids.add(robot_id)
        applied_torque = torch.zeros_like(joint_pos)
    policy["joint_efforts"] = applied_torque
    policy["body_pose_w"] = body_pose_w.reshape(body_pose_w.shape[0], -1)

    if imu_sensor is not None:
        policy["imu_ang_vel"] = imu_sensor.data.ang_vel_b
        policy["imu_lin_acc"] = imu_sensor.data.lin_acc_b
        policy["imu_orientation"] = imu_sensor.data.quat_w

    return {"policy": policy}


# ── Pose Extraction from Observation Dicts ───────────────────────────────────


def get_root_pose(obs: dict, env_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Extract root (position, quaternion) numpy arrays from *obs*."""
    policy = obs.get("policy", obs)
    return to_numpy(policy["root_pos_w"][env_idx]), to_numpy(policy["root_quat_w"][env_idx])


def get_body_pose(obs: dict, body_idx: int, env_idx: int = 0) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract (position, orientation) for a specific body, or ``(None, None)``."""
    policy = obs.get("policy", obs)
    body_pose_raw = policy.get("body_pose_w")
    if body_pose_raw is None:
        return None, None
    body_pose = to_numpy(body_pose_raw[env_idx])
    start = body_idx * 7
    p = body_pose[start : start + 7]
    return p[:3], p[3:]
