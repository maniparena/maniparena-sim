"""Tabletop objects for opencvpr tasks."""

from __future__ import annotations

import os

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from opencvpr.assets import ASSETS_DIR

_OBJECTS_DIR = os.path.join(ASSETS_DIR, "objects")


def _object_usd(asset_dir: str, asset_file: str) -> str:
    return os.path.join(_OBJECTS_DIR, asset_dir, asset_file)


class _TabletopObject(Object):
    name: str
    tags: list[str]
    usd_path: str
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            name=self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            scale=self.scale,
            initial_pose=initial_pose,
        )


class _Brick(_TabletopObject):
    tags = ["object", "sort_target"]
    scale = (1.0, 1.0, 1.0)


class _Paper(_TabletopObject):
    tags = ["object", "sort_dest"]
    scale = (1.0, 1.0, 1.0)


class _Button(_TabletopObject):
    tags = ["object", "button"]
    scale = (1.0, 1.0, 1.0)


@register_asset
class YellowBrick(_Brick):
    name = "yellow_brick"
    usd_path = _object_usd("brick", "brick_yellow.usd")


@register_asset
class GreenBrick(_Brick):
    name = "green_brick"
    usd_path = _object_usd("brick", "brick_green.usd")


@register_asset
class RedBrick(_Brick):
    name = "red_brick"
    usd_path = _object_usd("brick", "brick_red.usd")


@register_asset
class YellowPaper(_Paper):
    name = "yellow_paper"
    usd_path = _object_usd("paper", "paper_yellow.usd")


@register_asset
class GreenPaper(_Paper):
    name = "green_paper"
    usd_path = _object_usd("paper", "paper_green.usd")


@register_asset
class PinkPaper(_Paper):
    name = "pink_paper"
    usd_path = _object_usd("paper", "paper_pink.usd")


@register_asset
class ButtonGreen(_Button):
    name = "button_green"
    usd_path = _object_usd("button_color", "button_green.usd")


@register_asset
class ButtonPink(_Button):
    name = "button_pink"
    usd_path = _object_usd("button_color", "button_pink.usd")


@register_asset
class ButtonBlue(_Button):
    name = "button_blue"
    usd_path = _object_usd("button_color", "button_blue.usd")


@register_asset
class Pear(_TabletopObject):
    name = "pear"
    tags = ["object", "collect_target"]
    usd_path = _object_usd("pear", "pear.usd")
    scale = (0.8, 0.8, 0.8)


@register_asset
class Bread(_TabletopObject):
    name = "bread"
    tags = ["object"]
    usd_path = _object_usd("bread", "bread.usd")
    scale = (0.8, 0.8, 0.8)


@register_asset
class PlatformPink(_TabletopObject):
    name = "platform_pink"
    tags = ["object", "container"]
    usd_path = os.path.join(
        _OBJECTS_DIR, "platform_pink", "platform_pink_physics.usd",
    )
    scale = (0.003, 0.003, 0.003)


@register_asset
class Apple(_TabletopObject):
    name = "apple"
    tags = ["object", "collect_target"]
    usd_path = _object_usd("apple", "apple.usd")
    scale = (0.8, 0.8, 0.8)


@register_asset
class Banana(_TabletopObject):
    name = "banana"
    tags = ["object", "collect_target"]
    usd_path = _object_usd("banana", "banana.usd")
    scale = (0.8, 0.8, 0.8)
