"""quanta_x1 fixed-base tabletop builders (sort_blocks / fruits_to_basket).

They reuse the desktop task classes and success terms; only the reset events differ
(ranges come from ``task/quanta_tabletop_layouts.py``): every object of the task is
sampled jointly in one absolute region with a minimum centre separation.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab_arena.utils.configclass import make_configclass

from maniparena_sim.task.builders.fruits_to_basket_builder import FruitsToBasketBuilder
from maniparena_sim.task.builders.sort_blocks_builder import SortBlocksBuilder
from maniparena_sim.task.events import randomize_objects_in_region, randomize_objects_in_slots
from maniparena_sim.task.mobile_layouts import ARM_JOINT_NAMES
from maniparena_sim.task.quanta_tabletop_layouts import (
    FRUIT_PLACE_GATE,
    FRUIT_RESET,
    FRUIT_SLOTS,
    SORT_BLOCK_REGION,
    SORT_PLACE_GATE,
    TABLETOP_RETRACT,
)
from maniparena_sim.task.utils import find_assets_by_tag
from maniparena_sim.terms.terminations import place_then_retract


def _region_events_cfg(class_name: str, groups: list[tuple[list, dict]], seed: int | None):
    """reset_scene_to_default, then one joint-sampling event per (objects, region) group."""
    fields = [
        (
            "reset_scene_to_default",
            EventTermCfg,
            EventTermCfg(func=mdp_isaac_lab.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}),
        )
    ]
    for index, (objects, region) in enumerate(groups):
        fields.append(
            (
                f"randomize_group_{index}",
                EventTermCfg,
                EventTermCfg(
                    func=randomize_objects_in_region,
                    mode="reset",
                    params={
                        "asset_cfgs": [SceneEntityCfg(obj.name) for obj in objects],
                        "initial_poses": [obj.get_initial_pose() for obj in objects],
                        "seed": None if seed is None else int(seed) + index,
                        **region,
                    },
                ),
            )
        )
    return make_configclass(class_name, fields)()


def _success_cfg(placements: list[dict]) -> TerminationTermCfg:
    """Success: every placement gate fired, then both arms home with the objects still placed."""
    return TerminationTermCfg(
        func=place_then_retract,
        params={"placements": placements, "arm_joint_names": ARM_JOINT_NAMES, **TABLETOP_RETRACT},
    )


def _scene_seed(scene) -> int | None:
    return (getattr(scene, "scene_options", None) or {}).get("seed")


class QuantaSortBlocksBuilder(SortBlocksBuilder):
    def finalize(self, task_cfg, scene):
        task = super().finalize(task_cfg, scene)
        blocks = find_assets_by_tag(scene, "sort_target")
        papers = find_assets_by_tag(scene, "sort_dest")
        task.events_cfg = _region_events_cfg(
            "QuantaSortBlocksEventsCfg", [(blocks, SORT_BLOCK_REGION)], _scene_seed(scene)
        )
        task.termination_cfg.success = _success_cfg(
            [
                {
                    "kind": "footprint",
                    "object_cfg": SceneEntityCfg(block.name),
                    "reference_cfg": SceneEntityCfg(paper.name),
                    "reference_asset": paper,
                    **SORT_PLACE_GATE,
                }
                for block, paper in zip(blocks, papers)
            ]
        )
        return task


class QuantaFruitsToBasketBuilder(FruitsToBasketBuilder):
    def finalize(self, task_cfg, scene):
        task = super().finalize(task_cfg, scene)
        assets = list(scene.assets.values() if isinstance(scene.assets, dict) else scene.assets)
        by_name = {asset.name: asset for asset in assets}
        basket = find_assets_by_tag(scene, "container")[0]
        movers = [by_name[name] for name in ("apple", "banana", "pear", "bread")]
        task.events_cfg = make_configclass(
            "QuantaFruitsEventsCfg",
            [
                (
                    "reset_scene_to_default",
                    EventTermCfg,
                    EventTermCfg(
                        func=mdp_isaac_lab.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
                    ),
                ),
                (
                    "randomize_slots",
                    EventTermCfg,
                    EventTermCfg(
                        func=randomize_objects_in_slots,
                        mode="reset",
                        params={
                            "asset_cfgs": [SceneEntityCfg(obj.name) for obj in movers],
                            "initial_poses": [obj.get_initial_pose() for obj in movers],
                            "slots": FRUIT_SLOTS,
                            "seed": _scene_seed(scene),
                            **FRUIT_RESET,
                        },
                    ),
                ),
            ],
        )()
        # Bread is a distractor: only apple, banana and pear must end up in the basket.
        task.termination_cfg.success = _success_cfg(
            [
                {
                    "kind": "footprint",
                    "object_cfg": SceneEntityCfg(by_name[name].name),
                    "reference_cfg": SceneEntityCfg(basket.name),
                    "reference_asset": basket,
                    **FRUIT_PLACE_GATE,
                }
                for name in ("apple", "banana", "pear")
            ]
        )
        return task
