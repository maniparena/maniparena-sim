"""Local put_bottle_on_woodshelf assets (vendored Nucleus USDs, no spawn patches)."""

from __future__ import annotations

import os

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.assets import ASSETS_DIR

_PUT_BOTTLE_DIR = os.path.join(ASSETS_DIR, "put_bottle_on_woodshelf")


@register_asset
class PutBottleOnWoodshelfGreenBooth(LibraryBackground):
    name = "put_bottle_on_woodshelf_green_booth"
    tags = ["background", "put_bottle_on_woodshelf"]
    usd_path = os.path.join(
        _PUT_BOTTLE_DIR, "scenes", "green_booth", "green_booth_mesh_none_3x3.usd"
    )
    initial_pose = Pose.identity()
    object_min_z = -0.2


@register_asset
class PutBottleOnWoodshelfTable(LibraryObject):
    name = "put_bottle_on_woodshelf_table"
    tags = ["object", "furniture", "table", "put_bottle_on_woodshelf", "navigation_target"]
    usd_path = os.path.join(_PUT_BOTTLE_DIR, "models", "props", "table", "table_011.usd")
    object_type = ObjectType.RIGID
    scale = (1.0, 1.0, 1.0)


@register_asset
class PutBottleOnWoodshelfShelf(LibraryObject):
    name = "put_bottle_on_woodshelf_shelf"
    tags = [
        "object",
        "furniture",
        "shelf",
        "place_target",
        "put_bottle_on_woodshelf",
        "navigation_target",
    ]
    usd_path = os.path.join(_PUT_BOTTLE_DIR, "models", "props", "shelf", "shelf_c.usd")
    object_type = ObjectType.RIGID
    scale = (1.0, 1.0, 1.0)


@register_asset
class PutBottleOnWoodshelfBottleS(LibraryObject):
    name = "put_bottle_on_woodshelf_bottle_s"
    tags = ["object", "pickable", "bottle", "put_bottle_on_woodshelf"]
    usd_path = os.path.join(
        _PUT_BOTTLE_DIR, "models", "objects", "bottle", "bottle_nongfuspring_s.usd"
    )
    object_type = ObjectType.RIGID
    scale = (1.0, 1.0, 1.0)
