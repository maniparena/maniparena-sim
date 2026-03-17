"""Task helpers for scene queries and reset-event construction."""

from __future__ import annotations

from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.configclass import make_configclass

from maniparena_sim.task.base import DomainRandomizationCfg
from maniparena_sim.task.events import (
    randomize_object_orientation,
    randomize_object_pose,
    randomize_object_pose_from_pose_range,
    randomize_object_pose_full,
)


def build_object_reset_event_term(
    asset_name: str,
    initial_pose: Any,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> EventTermCfg:
    position_cfg = domain_randomization_cfg.object_position if domain_randomization_cfg is not None else None
    orientation_cfg = domain_randomization_cfg.object_orientation if domain_randomization_cfg is not None else None
    if position_cfg is not None and orientation_cfg is not None:
        return EventTermCfg(
            func=randomize_object_pose_full,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "position_range": position_cfg.to_range_dict(),
                "yaw_range": list(orientation_cfg.yaw_range),
            },
        )
    if position_cfg is not None:
        return EventTermCfg(
            func=randomize_object_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "position_range": position_cfg.to_range_dict(),
            },
        )
    if orientation_cfg is not None:
        return EventTermCfg(
            func=randomize_object_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "yaw_range": list(orientation_cfg.yaw_range),
            },
        )
    if pose_range:
        return EventTermCfg(
            func=randomize_object_pose_from_pose_range,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(asset_name),
                "initial_pose": initial_pose,
                "pose_range": pose_range,
            },
        )
    return EventTermCfg(func=set_object_pose, mode="reset", params={"pose": initial_pose, "asset_cfg": SceneEntityCfg(asset_name)})


def find_assets_by_tag(scene, tag: str) -> list:
    assets = getattr(scene, "assets", {})
    assets_list = list(assets.values()) if isinstance(assets, dict) else list(assets)
    matches = []
    for asset in assets_list:
        semantic_tags = getattr(asset, "semantic_tags", None) or []
        if isinstance(semantic_tags, str):
            semantic_tags = [semantic_tags]
        if tag in semantic_tags:
            matches.append(asset)
    return matches


def find_background(scene):
    assets = getattr(scene, "assets", {})
    assets_list = list(assets.values()) if isinstance(assets, dict) else list(assets)
    for asset in assets_list:
        if getattr(asset, "_component_type", "") == "BackgroundComponent" or hasattr(asset, "object_min_z"):
            return asset
    return None


def build_object_reset_events(
    objects: list,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> list[tuple[str, type, Any]]:
    fields = []
    for index, obj in enumerate(objects):
        initial_pose = obj.get_initial_pose()
        if initial_pose is None:
            continue
        fields.append(
            (
                f"reset_object_{index}_pose",
                EventTermCfg,
                build_object_reset_event_term(
                    asset_name=obj.name,
                    initial_pose=initial_pose,
                    domain_randomization_cfg=domain_randomization_cfg,
                    pose_range=pose_range,
                ),
            )
        )
    return fields


def build_default_events_cfg(
    objects: list,
    domain_randomization_cfg: DomainRandomizationCfg | None = None,
    pose_range: dict | None = None,
) -> Any:
    fields = [
        (
            "reset_scene_to_default",
            EventTermCfg,
            EventTermCfg(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}),
        )
    ]
    fields.extend(build_object_reset_events(objects, domain_randomization_cfg=domain_randomization_cfg, pose_range=pose_range))
    return make_configclass("ObjectResetEventsCfg", fields)()
