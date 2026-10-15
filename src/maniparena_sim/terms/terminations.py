"""Termination helpers shared by the supported tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from maniparena_sim.task.mobile_geometry import (
    in_footprint_zone,
    in_local_frame_box,
    upright_dot,
    world_aabb_from_local_bbox,
)


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


def _hold_counter(
    env: ManagerBasedRLEnv, key: str, inside: torch.Tensor, consecutive_steps: int, latch: bool
) -> torch.Tensor:
    """Per-env "inside for N consecutive steps" counter, optionally latched per episode."""
    states = getattr(env, "_mobile_hold_states", None)
    if states is None:
        states = {}
        setattr(env, "_mobile_hold_states", states)
    state = states.get(key)
    if state is None or state["count"].shape[0] != env.num_envs:
        state = {
            "count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "latched": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        states[key] = state
    if hasattr(env, "episode_length_buf"):
        fresh = env.episode_length_buf <= 1
        state["count"][fresh] = 0
        state["latched"][fresh] = False
    state["count"] = torch.where(inside, state["count"] + 1, torch.zeros_like(state["count"]))
    reached = state["count"] >= int(consecutive_steps)
    if latch:
        state["latched"] = state["latched"] | reached
        return state["latched"]
    return reached


def _local_bbox(asset, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Object-frame bbox (scale applied) of an Arena asset, broadcast over envs."""
    bbox = asset.get_bounding_box()
    lo = bbox.min_point.to(env.device)[:1].expand(env.num_envs, 3)
    hi = bbox.max_point.to(env.device)[:1].expand(env.num_envs, 3)
    return lo, hi


def object_in_reference_footprint(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg,
    reference_asset,
    xy_inset_ratio: float = 0.1,
    z_range: tuple[float, float] = (-0.03, 0.22),
    velocity_threshold: float | None = 0.03,
    consecutive_steps: int = 5,
    latch: bool = True,
    key_suffix: str = "",
) -> torch.Tensor:
    """Object centre inside a reference object's live footprint.

    The object's root must stay inside the reference's live world AABB (XY shrunk
    by ``xy_inset_ratio``), within ``z_range`` above the reference base, slower
    than ``velocity_threshold``, for ``consecutive_steps`` steps. Used for the
    bin / basket placement gates (bin: bottle, basket: fruit).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ref: RigidObject = env.scene[reference_cfg.name]
    origins = env.scene.env_origins
    lo, hi = _local_bbox(reference_asset, env)
    ref_min, ref_max = world_aabb_from_local_bbox(lo, hi, ref.data.root_pos_w - origins, ref.data.root_quat_w)
    inside = in_footprint_zone(
        obj.data.root_pos_w - origins, ref_min, ref_max, xy_inset_ratio=xy_inset_ratio, z_range=z_range
    )
    if velocity_threshold is not None:
        inside = inside & (torch.norm(obj.data.root_lin_vel_w, dim=-1) < velocity_threshold)
    return _hold_counter(
        env, f"footprint:{object_cfg.name}:{reference_cfg.name}{key_suffix}", inside, consecutive_steps, latch
    )


def object_in_reference_frame_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    velocity_threshold: float = 0.03,
    min_upright_dot: float | None = 0.95,
    consecutive_steps: int = 5,
    latch: bool = False,
    key_suffix: str = "",
) -> torch.Tensor:
    """Object root inside a box in the reference's own frame (shelf placement gate)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ref: RigidObject = env.scene[reference_cfg.name]
    origins = env.scene.env_origins
    inside = in_local_frame_box(
        obj.data.root_pos_w - origins,
        ref.data.root_pos_w - origins,
        ref.data.root_quat_w,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
    ) & (torch.norm(obj.data.root_lin_vel_w, dim=-1) < velocity_threshold)
    if min_upright_dot is not None:
        inside = inside & (upright_dot(obj.data.root_quat_w) >= float(min_upright_dot))
    return _hold_counter(
        env, f"frame_box:{object_cfg.name}:{reference_cfg.name}{key_suffix}", inside, consecutive_steps, latch
    )


def joints_near_default(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    atol: float,
    consecutive_steps: int = 1,
    key: str = "joints",
) -> torch.Tensor:
    """All named joints within ``atol`` of their default position for ``consecutive_steps`` steps."""
    robot = env.scene["robot"]
    indices, _ = robot.find_joints(list(joint_names), preserve_order=True)
    delta = (robot.data.joint_pos[:, indices] - robot.data.default_joint_pos[:, indices]).abs()
    return _hold_counter(env, f"near_default:{key}", torch.all(delta <= atol, dim=-1), consecutive_steps, False)


_GATES = {"footprint": object_in_reference_footprint, "frame_box": object_in_reference_frame_box}


def place_then_retract(
    env: ManagerBasedRLEnv,
    placements: list[dict],
    arm_joint_names: list[str],
    arm_atol: float,
    gripper_joint_names: list[str] | None = None,
    gripper_atol: float = 0.05,
    retract_steps: int = 10,
) -> torch.Tensor:
    """Episode success = every placement gate passed, then the arms are back home with the objects still placed.

    A run only counts as success after the final return / retract stage: all placement gates
    (each with its own hold / latch settings) have fired, both arms are within ``arm_atol`` of their default
    pose for ``retract_steps`` steps (grippers within ``gripper_atol`` when given), and every object is still
    inside its target footprint (instantaneous check, no velocity gate).

    Each placement is ``{"kind": "footprint" | "frame_box", **gate kwargs}``.
    """
    placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    still = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for spec in placements:
        gate = _GATES[spec["kind"]]
        kwargs = {k: v for k, v in spec.items() if k != "kind"}
        placed = placed & gate(env, **kwargs)
        relaxed = {**kwargs, "consecutive_steps": 1, "latch": False, "key_suffix": "/still"}
        if spec["kind"] == "footprint":
            relaxed["velocity_threshold"] = None
        still = still & gate(env, **relaxed)
    home = joints_near_default(env, arm_joint_names, arm_atol, retract_steps, key="arms")
    if gripper_joint_names:
        home = home & joints_near_default(env, gripper_joint_names, gripper_atol, 1, key="grippers")
    return placed & home & still
