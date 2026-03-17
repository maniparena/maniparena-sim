"""Task-level event functions for object reset and randomization."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab_arena.utils.pose import Pose


def _sync_pose_to_usd(asset, env_ids: torch.Tensor, local_pose: torch.Tensor) -> None:
    try:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for index, env_idx in enumerate(env_ids):
            prim_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", f"envs/env_{int(env_idx)}").replace("env_.*", f"env_{int(env_idx)}")
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            pos = local_pose[index, :3].cpu().tolist()
            quat = local_pose[index, 3:7].cpu().tolist()
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            translate_set = False
            orient_set = False
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    op.Set(Gf.Vec3d(*pos))
                    translate_set = True
                elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    op.Set(Gf.Quatd(quat[0], quat[1], quat[2], quat[3]))
                    orient_set = True
            if not translate_set or not orient_set:
                xformable.ClearXformOpOrder()
                xformable.AddTranslateOp().Set(Gf.Vec3d(*pos))
                xformable.AddOrientOp().Set(Gf.Quatd(quat[0], quat[1], quat[2], quat[3]))
    except Exception:
        pass


def randomize_object_pose(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, initial_pose: Pose, position_range: dict[str, list[float]]) -> None:
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)
    pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("x", [0.0, 0.0]))
    pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("y", [0.0, 0.0]))
    pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("z", [0.0, 0.0]))
    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose_tensor, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def randomize_object_orientation(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, initial_pose: Pose, yaw_range: list[float] | None = None) -> None:
    if env_ids is None:
        return
    if yaw_range is None:
        yaw_range = [-3.14159, 3.14159]
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)
    half_yaw = torch.empty(num_envs, device=env.device).uniform_(yaw_range[0], yaw_range[1]) * 0.5
    pose_tensor[:, 3] = torch.cos(half_yaw)
    pose_tensor[:, 4] = 0.0
    pose_tensor[:, 5] = 0.0
    pose_tensor[:, 6] = torch.sin(half_yaw)
    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose_tensor, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def randomize_object_pose_full(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_pose: Pose,
    position_range: dict[str, list[float]] | None = None,
    yaw_range: list[float] | None = None,
) -> None:
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)
    if position_range is not None:
        pose_tensor[:, 0] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("x", [0.0, 0.0]))
        pose_tensor[:, 1] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("y", [0.0, 0.0]))
        pose_tensor[:, 2] += torch.empty(num_envs, device=env.device).uniform_(*position_range.get("z", [0.0, 0.0]))
    if yaw_range is not None:
        half_yaw = torch.empty(num_envs, device=env.device).uniform_(yaw_range[0], yaw_range[1]) * 0.5
        pose_tensor[:, 3] = torch.cos(half_yaw)
        pose_tensor[:, 4] = 0.0
        pose_tensor[:, 5] = 0.0
        pose_tensor[:, 6] = torch.sin(half_yaw)
    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose_tensor, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)


def randomize_object_pose_from_pose_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    initial_pose: Pose,
    pose_range: dict[str, list[float] | tuple[float, float]],
) -> None:
    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    pose_tensor = initial_pose.to_tensor(device=env.device).repeat(num_envs, 1)
    for coord_index, key in enumerate(("x", "y", "z")):
        if key in pose_range:
            pose_tensor[:, coord_index] = torch.empty(num_envs, device=env.device).uniform_(*pose_range[key])
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(pose_tensor[:, 3:7])
    if "roll" in pose_range:
        roll = torch.empty(num_envs, device=env.device).uniform_(*pose_range["roll"])
    if "pitch" in pose_range:
        pitch = torch.empty(num_envs, device=env.device).uniform_(*pose_range["pitch"])
    if "yaw" in pose_range:
        yaw = torch.empty(num_envs, device=env.device).uniform_(*pose_range["yaw"])
    pose_tensor[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    _sync_pose_to_usd(asset, env_ids, pose_tensor)
    pose_tensor[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose_tensor, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=env.device), env_ids=env_ids)
