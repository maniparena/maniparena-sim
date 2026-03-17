"""Termination helpers shared by the three supported tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def all_objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    destination_cfgs: list[SceneEntityCfg],
    x_threshold: float = 0.12,
    y_threshold: float = 0.16,
    z_threshold: float = 0.06,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for object_cfg, dest_cfg in zip(object_cfgs, destination_cfgs):
        obj: RigidObject = env.scene[object_cfg.name]
        dest: RigidObject = env.scene[dest_cfg.name]
        obj_pos = obj.data.root_pos_w - env.scene.env_origins
        dest_pos = dest.data.root_pos_w - env.scene.env_origins
        placed = (
            (torch.abs(obj_pos[:, 0] - dest_pos[:, 0]) < x_threshold)
            & (torch.abs(obj_pos[:, 1] - dest_pos[:, 1]) < y_threshold)
            & (torch.abs(obj_pos[:, 2] - dest_pos[:, 2]) < z_threshold)
            & (torch.norm(obj.data.root_lin_vel_w, dim=-1) < velocity_threshold)
        )
        all_placed = all_placed & placed
    return all_placed


def all_objects_in_basket(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    basket_cfg: SceneEntityCfg,
    x_threshold: float = 0.14,
    y_threshold: float = 0.14,
    z_threshold: float = 0.12,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    basket: RigidObject = env.scene[basket_cfg.name]
    basket_pos = basket.data.root_pos_w - env.scene.env_origins
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        obj_pos = obj.data.root_pos_w - env.scene.env_origins
        placed = (
            (torch.abs(obj_pos[:, 0] - basket_pos[:, 0]) < x_threshold)
            & (torch.abs(obj_pos[:, 1] - basket_pos[:, 1]) < y_threshold)
            & (torch.abs(obj_pos[:, 2] - basket_pos[:, 2]) < z_threshold)
            & (torch.norm(obj.data.root_lin_vel_w, dim=-1) < velocity_threshold)
        )
        all_placed = all_placed & placed
    return all_placed


def any_object_dropped(env: ManagerBasedRLEnv, object_cfgs: list[SceneEntityCfg], minimum_height: float) -> torch.Tensor:
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        pos = obj.data.root_pos_w - env.scene.env_origins
        any_dropped = any_dropped | (pos[:, 2] < minimum_height)
    return any_dropped


def contacts_all_buttons_once(
    env: ManagerBasedRLEnv,
    contact_sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    num_envs = env.num_envs
    num_buttons = len(contact_sensor_cfgs)
    history_attr = "_buttons_contact_history_state"
    if (not hasattr(env, history_attr)) or (getattr(env, history_attr).shape != (num_envs, num_buttons)):
        setattr(env, history_attr, torch.zeros((num_envs, num_buttons), dtype=torch.bool, device=env.device))
    contact_history: torch.Tensor = getattr(env, history_attr)
    reset_mask = torch.zeros((num_envs,), dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = reset_mask | (env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = reset_mask | (env.reset_buf > 0)
    if torch.any(reset_mask):
        contact_history[reset_mask] = False
    for index, sensor_cfg in enumerate(contact_sensor_cfgs):
        sensor = env.scene[sensor_cfg.name]
        force_norm = torch.norm(sensor.data.force_matrix_w, dim=-1)
        in_contact = torch.amax(force_norm, dim=(1, 2)) > force_threshold
        contact_history[:, index] = contact_history[:, index] | in_contact
    return torch.all(contact_history, dim=1)
