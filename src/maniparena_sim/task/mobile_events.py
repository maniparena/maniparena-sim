"""Reset-event helpers shared by the quanta_x1 mobile-manipulation tasks."""

from __future__ import annotations

import random
from typing import Any

import torch
from isaaclab.envs import ManagerBasedEnv


def write_root_pose(
    asset: Any,
    env: ManagerBasedEnv,
    env_id: int,
    position_xyz: list[float],
    quat_xyzw: list[float],
    *,
    reset_joints_to_default: bool = False,
) -> None:
    """Teleport one env's asset to an env-local root pose and zero its velocity."""
    env_ids = torch.as_tensor([int(env_id)], dtype=torch.long, device=env.device)
    pose = torch.zeros(1, 7, dtype=torch.float32, device=env.device)
    pose[0, :3] = torch.as_tensor(position_xyz, dtype=torch.float32, device=env.device)
    pose[0, 3:7] = torch.as_tensor(quat_xyzw, dtype=torch.float32, device=env.device)
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    if hasattr(asset, "write_joint_state_to_sim") and hasattr(asset.data, "joint_pos"):
        if reset_joints_to_default and hasattr(asset.data, "default_joint_pos"):
            joint_pos = asset.data.default_joint_pos[env_ids].clone()
        else:
            joint_pos = asset.data.joint_pos[env_ids].clone()
        if joint_pos.numel() > 0:
            asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        _ = asset.data.body_link_pose_w[env_ids]
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def env_rng(env: ManagerBasedEnv, attr: str, seed: int | None) -> random.Random:
    """Per-env-object python RNG, created on first use (``seed=None`` -> OS entropy)."""
    rng = getattr(env, attr, None)
    if rng is None:
        rng = random.Random(None if seed is None else int(seed))
        setattr(env, attr, rng)
    return rng
