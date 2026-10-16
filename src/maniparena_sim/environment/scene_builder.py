"""Arena-style scene assembly for supported maniparena_sim tasks."""

from __future__ import annotations

import random

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.environment.registry import bootstrap_arena_registry
from maniparena_sim.environment.render_settings import load_render_settings
from maniparena_sim.task.mobile_layouts import (
    BASKET_TABLE_POSE,
    BIN_BOTTLE_LOAD_POSE,
    BIN_POSE,
    BIN_TABLE_POSE,
    FRUIT_RESET,
    FRUIT_TABLE_POSE,
    SHELF_BOTTLE_LOAD_POSE,
    SHELF_POSE,
    SHELF_TABLE_POSE,
    TABLETOP_LOCAL_Z,
)
from maniparena_sim.task.quanta_tabletop_layouts import (
    FRUIT_NOMINAL_LOAD_POSES,
    QUANTA_BASKET_POSE,
    SORT_BRICK_LOAD_POSES,
    SORT_PAPER_POSES,
)

_BIMANUAL_BG_POSE = Pose(
    position_xyz=(-1.0, -0.562, 0.0),
    rotation_xyzw=(0.0, 0.0, -0.999, 0.04457),
)
_QUANTA_X1_BG_POSE = Pose(
    position_xyz=(-0.452, -0.271, -0.985),
    rotation_xyzw=(0.0, 0.0, -0.704, 0.709),
)
_IDENTITY_ROT = (0.0, 0.0, 0.0, 1.0)
_PAPER_ROT = (0.0, 0.0, 0.70711, 0.70711)

_SORT_BLOCK_POSES = [
    (0.05, -0.15, -0.15),
    (0.05, 0.0, -0.15),
    (0.05, 0.15, -0.15),
]
_SORT_DEST_POSES = [
    (0.236, 0.02, -0.221),
    (0.238, -0.25, -0.223),
    (0.240, -0.52, -0.226),
]
_FRUIT_POSES = [
    (0.05, -0.30, -0.15),
    (0.05, -0.10, -0.15),
    (0.05, 0.10, -0.15),
    (0.05, 0.30, -0.15),
]
_BASKET_POSE = (0.262, -0.224, -0.226)
_BUTTON_POSES = [
    (0.05, -0.15, -0.17),
    (0.05, 0.0, -0.17),
    (0.05, 0.15, -0.17),
]

# Arena Pose / Lab 3 ArticulationCfg.init_state.rot use XYZW.
# Woodshelf layout follows the collected demonstrations (task/mobile_layouts.py): table root z=0,
# tabletop plane z=0.7067, bottle dropped from root z=0.84 by the reset event.
_PUT_BOTTLE_TABLE_POSE = SHELF_TABLE_POSE
_PUT_BOTTLE_SHELF_POSE = SHELF_POSE
_PUT_BOTTLE_BOTTLE_POSE = SHELF_BOTTLE_LOAD_POSE
PUT_BOTTLE_QUANTA_X1_INIT_POS = (-0.31154, -0.30644, 0.0)
PUT_BOTTLE_QUANTA_X1_INIT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)  # yaw=0


def _apply_semantic_tags(asset) -> None:
    tags = getattr(asset, "tags", None) or []
    asset.semantic_tags = list(tags)


def _pose_at(position_xyz: tuple[float, float, float], rotation_xyzw=_IDENTITY_ROT) -> Pose:
    return Pose(position_xyz=position_xyz, rotation_xyzw=rotation_xyzw)


def _spawn_background(registry: AssetRegistry, robot: str):
    """Pick background USD + pose. Robot only affects the booth asset, not objects."""
    if robot == "quanta_x1":
        background = registry.get_asset_by_name("green_booth_mesh_ground")()
        background.set_initial_pose(_QUANTA_X1_BG_POSE)
        return background
    background = registry.get_asset_by_name("green_booth")()
    background.set_initial_pose(_BIMANUAL_BG_POSE)
    return background


def _place_assets(assets, positions, *, rotation_xyzw=_IDENTITY_ROT) -> None:
    for asset, position in zip(assets, positions):
        asset.set_initial_pose(_pose_at(position, rotation_xyzw))
        _apply_semantic_tags(asset)


def _build_sort_blocks_scene(registry: AssetRegistry, robot: str) -> Scene:
    """Desktop-arm sort_blocks (ManipArena bricks / papers)."""
    background = _spawn_background(registry, robot)
    blocks = [
        registry.get_asset_by_name("yellow_brick")(),
        registry.get_asset_by_name("green_brick")(),
        registry.get_asset_by_name("red_brick")(),
    ]
    destinations = [
        registry.get_asset_by_name(name)() for name in ("yellow_paper", "green_paper", "pink_paper")
    ]
    _place_assets(blocks, _SORT_BLOCK_POSES)
    _place_assets(destinations, _SORT_DEST_POSES, rotation_xyzw=_PAPER_ROT)
    return Scene(assets=[background, *blocks, *destinations])


def _build_quanta_x1_sort_blocks_scene(registry: AssetRegistry) -> Scene:
    """QUANTA_X1 sort_blocks: manaenv bricks / papers (``_2`` assets), same layout as data collection."""
    background = _spawn_background(registry, "quanta_x1")
    blocks = [
        registry.get_asset_by_name(name)() for name in ("yellow_brick_2", "green_brick_2", "red_brick_2")
    ]
    # Fixed (kinematic) papers; bricks rest on the table, papers 1.5 mm above it.
    destinations = [
        registry.get_asset_by_name(name)() for name in ("yellow_paper_2", "green_paper_2", "pink_paper_2")
    ]
    _place_assets(blocks, SORT_BRICK_LOAD_POSES)
    _place_assets(destinations, SORT_PAPER_POSES, rotation_xyzw=_PAPER_ROT)
    return Scene(assets=[background, *blocks, *destinations])


def _build_fruits_to_basket_scene(registry: AssetRegistry, robot: str) -> Scene:
    """Desktop-arm fruits_to_basket."""
    background = _spawn_background(registry, robot)
    objects = [
        registry.get_asset_by_name("bread")(),
        registry.get_asset_by_name("apple")(),
        registry.get_asset_by_name("banana")(),
        registry.get_asset_by_name("pear")(),
    ]
    basket = registry.get_asset_by_name("platform_pink")()
    basket.set_initial_pose(_pose_at(_BASKET_POSE, _PAPER_ROT))
    _apply_semantic_tags(basket)
    _place_assets(objects, _FRUIT_POSES)
    return Scene(assets=[background, basket, *objects])


def _build_quanta_x1_fruits_to_basket_scene(registry: AssetRegistry) -> Scene:
    """QUANTA_X1 fruits_to_basket: manaenv-sized basket (``platform_pink_2``), centred between the arms."""
    background = _spawn_background(registry, "quanta_x1")
    objects = [
        registry.get_asset_by_name("bread")(),
        registry.get_asset_by_name("apple")(),
        registry.get_asset_by_name("banana")(),
        registry.get_asset_by_name("pear")(),
    ]
    basket = registry.get_asset_by_name("platform_pink_2")()
    basket.set_initial_pose(_pose_at(QUANTA_BASKET_POSE, _PAPER_ROT))
    _apply_semantic_tags(basket)
    # objects = [bread, apple, banana, pear]
    fruit_poses = [FRUIT_NOMINAL_LOAD_POSES[name] for name in ("bread", "apple", "banana", "pear")]
    _place_assets(objects, fruit_poses)
    return Scene(assets=[background, basket, *objects])


def _build_buttons_contact_scene(registry: AssetRegistry, robot: str) -> Scene:
    background = _spawn_background(registry, robot)
    buttons = [
        registry.get_asset_by_name("button_green")(),
        registry.get_asset_by_name("button_pink")(),
        registry.get_asset_by_name("button_blue")(),
    ]
    _place_assets(buttons, _BUTTON_POSES)
    return Scene(assets=[background, *buttons])


def _build_dummy_task_scene(registry: AssetRegistry) -> Scene:
    background = registry.get_asset_by_name("nav_f16")()
    background.set_initial_pose(_pose_at((0.0, 0.0, 0.0)))
    return Scene(assets=[background])


def _build_put_bottle_on_woodshelf_scene(registry: AssetRegistry) -> Scene:
    """Assemble the woodshelf mobile-manip scene (static poses, no planner)."""
    background = registry.get_asset_by_name("put_bottle_on_woodshelf_green_booth")()
    background.set_initial_pose(_pose_at((0.0, 0.0, 0.0)))
    table = registry.get_asset_by_name("put_bottle_on_woodshelf_table")()
    shelf = registry.get_asset_by_name("put_bottle_on_woodshelf_shelf")()
    bottle = registry.get_asset_by_name("put_bottle_on_woodshelf_bottle_s")()
    table.set_initial_pose(_pose_at(*_PUT_BOTTLE_TABLE_POSE))
    shelf.set_initial_pose(_pose_at(*_PUT_BOTTLE_SHELF_POSE))
    # Smoke-load pose for construction only; episode resets sample from case_bank.
    bottle.set_initial_pose(_pose_at(*_PUT_BOTTLE_BOTTLE_POSE), create_reset_event=False)
    for asset in (table, shelf, bottle):
        _apply_semantic_tags(asset)
    return Scene(assets=[background, table, shelf, bottle])


def _choose_pool_member(kind: str, requested: str | None, rng: random.Random) -> str:
    """Registered asset name of one pool member (``requested`` = pool name, or None/"random")."""
    from maniparena_sim.assets.objects.mobile_manip import ASSET_PREFIX, POOLS_PATH, pool_asset_names

    names = pool_asset_names(kind)
    if not names:
        raise FileNotFoundError(
            f"No '{kind}' pool registered: {POOLS_PATH} is missing or empty. "
            "The mobile pick-and-place assets are expected under assets/mobile_manip/."
        )
    if requested in (None, "", "random"):
        return rng.choice(names)
    candidate = requested if requested in names else ASSET_PREFIX + requested
    if candidate not in names:
        raise ValueError(f"Unknown {kind} pool member '{requested}'. Available: {names}")
    return candidate


def _floor_booth(registry: AssetRegistry):
    background = registry.get_asset_by_name("put_bottle_on_woodshelf_green_booth")()
    background.set_initial_pose(_pose_at((0.0, 0.0, 0.0)))
    return background


def _table(registry: AssetRegistry, name: str, pose):
    table = registry.get_asset_by_name("put_bottle_on_woodshelf_table")(instance_name=name)
    table.set_initial_pose(_pose_at(*pose))
    return table


def _build_move_bottle_to_bin_scene(registry: AssetRegistry, options: dict) -> Scene:
    """Bottle on a table, bin on the floor (move_bottle_to_bin; episode resets re-sample)."""
    rng = random.Random(options.get("seed"))
    background = _floor_booth(registry)
    table = _table(registry, "table", BIN_TABLE_POSE)
    bin_asset = registry.get_asset_by_name("mobile_trash_can")(instance_name="bin")
    bin_asset.set_initial_pose(_pose_at(*BIN_POSE), create_reset_event=False)
    bottle_name = _choose_pool_member("bottles", options.get("bottle"), rng)
    bottle = registry.get_asset_by_name(bottle_name)(instance_name="bottle")
    bottle.set_initial_pose(_pose_at(*BIN_BOTTLE_LOAD_POSE), create_reset_event=False)
    assets = (table, bin_asset, bottle)
    for asset in assets:
        _apply_semantic_tags(asset)
    return Scene(assets=[background, *assets])


def _build_move_fruit_to_basket_scene(registry: AssetRegistry, options: dict) -> Scene:
    """Fruit on one table, basket on another (move_fruit_to_basket)."""
    rng = random.Random(options.get("seed"))
    background = _floor_booth(registry)
    fruit_table = _table(registry, "fruit_table", FRUIT_TABLE_POSE)
    basket_table = _table(registry, "basket_table", BASKET_TABLE_POSE)
    fruit = registry.get_asset_by_name(_choose_pool_member("fruits", options.get("fruit"), rng))(
        instance_name="fruit"
    )
    basket = registry.get_asset_by_name(_choose_pool_member("baskets", options.get("basket"), rng))(
        instance_name="basket"
    )
    for table_pose, asset in ((FRUIT_TABLE_POSE, fruit), (BASKET_TABLE_POSE, basket)):
        # Load pose = table-top centre; episode resets re-sample the XY.
        root_z = TABLETOP_LOCAL_Z - float(asset.get_bounding_box().min_point[0, 2]) + FRUIT_RESET["root_clearance_m"]
        asset.set_initial_pose(_pose_at((table_pose[0][0], table_pose[0][1], root_z)), create_reset_event=False)
    assets = (fruit_table, basket_table, fruit, basket)
    for asset in assets:
        _apply_semantic_tags(asset)
    return Scene(assets=[background, *assets])


def _attach_render_settings(
    scene: Scene,
    settings_name: str = "green_booth",
    sim_fps: int = 120,
    render_decremental: int = 2,
) -> Scene:
    """Load render settings and attach to scene."""
    render_cfg_dict, carb_dict = load_render_settings(settings_name)
    scene.render_cfg_dict = render_cfg_dict
    scene.render_carb_dict = carb_dict
    scene.sim_fps = sim_fps
    scene.render_decremental = render_decremental
    return scene


def build_scene(task_name: str, *, robot: str = "bimanual", scene_options: dict | None = None):
    """Assemble scene assets for *task_name*.

    Object poses depend only on the task. ``robot`` selects the background booth
    USD and its world pose (``green_booth`` for bimanual,
    ``green_booth_mesh_ground`` for quanta_x1). Robot base pose is set on the
    embodiment, not here. ``scene_options`` (quanta_x1 mobile tasks) selects pool
    members (``bottle`` / ``fruit`` / ``basket``: pool name or "random") and the
    sampling ``seed``; the chosen USD stays fixed for the process lifetime.
    """
    if robot not in ("bimanual", "quanta_x1"):
        raise ValueError(f"Unsupported robot '{robot}' for scene assembly.")
    bootstrap_arena_registry()
    registry = AssetRegistry()
    options = dict(scene_options or {})
    if robot == "quanta_x1" and task_name == "sort_blocks":
        scene = _build_quanta_x1_sort_blocks_scene(registry)
    elif robot == "quanta_x1" and task_name == "fruits_to_basket":
        scene = _build_quanta_x1_fruits_to_basket_scene(registry)
    elif task_name == "sort_blocks":
        scene = _build_sort_blocks_scene(registry, robot)
    elif task_name == "fruits_to_basket":
        scene = _build_fruits_to_basket_scene(registry, robot)
    elif task_name == "buttons_contact":
        scene = _build_buttons_contact_scene(registry, robot)
    elif task_name == "dummy_task":
        scene = _build_dummy_task_scene(registry)
    elif task_name == "put_bottle_on_woodshelf":
        scene = _build_put_bottle_on_woodshelf_scene(registry)
    elif task_name == "move_bottle_to_bin":
        scene = _build_move_bottle_to_bin_scene(registry, options)
    elif task_name == "move_fruit_to_basket":
        scene = _build_move_fruit_to_basket_scene(registry, options)
    else:
        raise ValueError(f"Unsupported task '{task_name}' for scene assembly.")
    settings_name = "nav_f16" if task_name == "dummy_task" else "green_booth"
    scene.scene_options = options
    return _attach_render_settings(scene, settings_name=settings_name)
