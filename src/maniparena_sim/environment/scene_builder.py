"""Arena-style scene assembly for supported maniparena_sim tasks."""

from __future__ import annotations

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.environment.registry import bootstrap_arena_registry


def _apply_semantic_tags(asset) -> None:
    tags = getattr(asset, "tags", None) or []
    asset.semantic_tags = list(tags)


def _build_sort_blocks_scene(registry: AssetRegistry) -> Scene:
    background = registry.get_asset_by_name("green_booth")()
    background.set_initial_pose(
        Pose(
            position_xyz=(-1.0, -0.562, 0.0),
            rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
        )
    )
    blocks = [
        registry.get_asset_by_name("yellow_brick")(),
        registry.get_asset_by_name("green_brick")(),
        registry.get_asset_by_name("red_brick")(),
    ]
    destinations = [
        registry.get_asset_by_name("yellow_paper")(),
        registry.get_asset_by_name("green_paper")(),
        registry.get_asset_by_name("pink_paper")(),
    ]

    block_positions = [
        (0.05, -0.15, -0.15),
        (0.05, 0.0, -0.15),
        (0.05, 0.15, -0.15),
    ]
    for index, block in enumerate(blocks):
        block.set_initial_pose(
            Pose(
                position_xyz=block_positions[index],
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        _apply_semantic_tags(block)
    dest_positions = [
        (0.236, 0.076, -0.211),
        (0.238, -0.272, -0.223),
        (0.240, -0.615, -0.226),
    ]
    for index, dest in enumerate(destinations):
        dest.set_initial_pose(
            Pose(
                position_xyz=dest_positions[index],
                rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711),
            )
        )
        _apply_semantic_tags(dest)
    return Scene(assets=[background, *blocks, *destinations])


def _build_fruits_to_basket_scene(registry: AssetRegistry) -> Scene:
    background = registry.get_asset_by_name("green_booth")()
    background.set_initial_pose(
        Pose(
            position_xyz=(-1.0, -0.562, 0.0),
            rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
        )
    )
    objects = [
        registry.get_asset_by_name("bread")(),
        registry.get_asset_by_name("apple")(),
        registry.get_asset_by_name("banana")(),
        registry.get_asset_by_name("pear")(),
    ]
    basket = registry.get_asset_by_name("platform_pink")()

    basket.set_initial_pose(
        Pose(
            position_xyz=(0.262, -0.194, -0.226),
            rotation_wxyz=(0.0, 0.70711, 0.70711, 0.0),
        )
    )
    _apply_semantic_tags(basket)
    object_positions = [
        (0.05, -0.30, -0.15),
        (0.05, -0.10, -0.15),
        (0.05, 0.10, -0.15),
        (0.05, 0.30, -0.15),
    ]
    for index, obj in enumerate(objects):
        obj.set_initial_pose(
            Pose(
                position_xyz=object_positions[index],
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        _apply_semantic_tags(obj)
    return Scene(assets=[background, basket, *objects])


def _build_buttons_contact_scene(registry: AssetRegistry) -> Scene:
    background = registry.get_asset_by_name("green_booth")()
    background.set_initial_pose(
        Pose(
            position_xyz=(-1.0, -0.562, 0.0),
            rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
        )
    )
    buttons = [
        registry.get_asset_by_name("button_green")(),
        registry.get_asset_by_name("button_pink")(),
        registry.get_asset_by_name("button_blue")(),
    ]
    button_positions = [
        (0.05, -0.15, -0.17),
        (0.05, 0.0, -0.17),
        (0.05, 0.15, -0.17),
    ]
    for index, button in enumerate(buttons):
        button.set_initial_pose(
            Pose(
                position_xyz=button_positions[index],
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        _apply_semantic_tags(button)
    return Scene(assets=[background, *buttons])


def build_scene(task_name: str):
    bootstrap_arena_registry()
    registry = AssetRegistry()
    if task_name == "sort_blocks":
        return _build_sort_blocks_scene(registry)
    if task_name == "fruits_to_basket":
        return _build_fruits_to_basket_scene(registry)
    if task_name == "buttons_contact":
        return _build_buttons_contact_scene(registry)
    raise ValueError(f"Unsupported task '{task_name}' for scene assembly.")
