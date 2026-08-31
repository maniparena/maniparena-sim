"""Arena-style scene assembly for supported maniparena_sim tasks."""

from __future__ import annotations

from isaaclab_arena.assets.registries import AssetRegistry
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.environment.registry import bootstrap_arena_registry
from maniparena_sim.environment.render_settings import load_render_settings

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
#
# Raise table root so tabletop plane is world z=0.75 (was ~0.7067, Δz=0.0433).
_PUT_BOTTLE_TABLETOP_Z = 0.75
_PUT_BOTTLE_TABLE_ROOT_Z = 0.0433
_PUT_BOTTLE_BOTTLE_ROOT_Z = 0.8833  # keep prior clearance above the plane
_PUT_BOTTLE_TABLE_POSE = (
    (1.5, -0.95, _PUT_BOTTLE_TABLE_ROOT_Z),
    (0.0, 0.0, 0.707107, 0.707107),
)
_PUT_BOTTLE_SHELF_POSE = ((0.63, -1.7907, 0.0), (0.0, 0.0, 1.0, 0.0))  # z-180
_PUT_BOTTLE_BOTTLE_POSE = (
    (1.505, -0.95, _PUT_BOTTLE_BOTTLE_ROOT_Z),
    (-0.01837, 0.007908, 0.000433, 0.9998),
)
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
    background = _spawn_background(registry, robot)
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
    _place_assets(blocks, _SORT_BLOCK_POSES)
    _place_assets(destinations, _SORT_DEST_POSES, rotation_xyzw=_PAPER_ROT)
    return Scene(assets=[background, *blocks, *destinations])


def _build_fruits_to_basket_scene(registry: AssetRegistry, robot: str) -> Scene:
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


def build_scene(task_name: str, *, robot: str = "bimanual"):
    """Assemble scene assets for *task_name*.

    Object poses depend only on the task. ``robot`` selects the background booth
    USD and its world pose (``green_booth`` for bimanual,
    ``green_booth_mesh_ground`` for quanta_x1). Robot base pose is set on the
    embodiment, not here.
    """
    if robot not in ("bimanual", "quanta_x1"):
        raise ValueError(f"Unsupported robot '{robot}' for scene assembly.")
    bootstrap_arena_registry()
    registry = AssetRegistry()
    if task_name == "sort_blocks":
        scene = _build_sort_blocks_scene(registry, robot)
    elif task_name == "fruits_to_basket":
        scene = _build_fruits_to_basket_scene(registry, robot)
    elif task_name == "buttons_contact":
        scene = _build_buttons_contact_scene(registry, robot)
    elif task_name == "dummy_task":
        scene = _build_dummy_task_scene(registry)
    elif task_name == "put_bottle_on_woodshelf":
        scene = _build_put_bottle_on_woodshelf_scene(registry)
    else:
        raise ValueError(f"Unsupported task '{task_name}' for scene assembly.")
    settings_name = "nav_f16" if task_name == "dummy_task" else "green_booth"
    return _attach_render_settings(scene, settings_name=settings_name)
