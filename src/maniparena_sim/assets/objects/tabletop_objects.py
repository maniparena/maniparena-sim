"""Tabletop objects for maniparena_sim tasks."""

from __future__ import annotations

import os

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.assets import ASSETS_DIR

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
    scale = (2.0, 2.0, 2.0)


@register_asset
class GreenBrick(_Brick):
    name = "green_brick"
    usd_path = _object_usd("brick", "brick_green.usd")
    scale = (2.0, 2.0, 2.0)


@register_asset
class RedBrick(_Brick):
    name = "red_brick"
    usd_path = _object_usd("brick", "brick_red.usd")
    scale = (2.0, 2.0, 2.0)

@register_asset
class YellowPaper(_Paper):
    name = "yellow_paper"
    usd_path = _object_usd("paper", "paper_yellow.usd")
    scale = (0.8, 0.8, 0.8)


@register_asset
class GreenPaper(_Paper):
    name = "green_paper"
    usd_path = _object_usd("paper", "paper_green.usd")
    scale = (0.8, 0.8, 0.8)

@register_asset
class PinkPaper(_Paper):
    name = "pink_paper"
    usd_path = _object_usd("paper", "paper_pink.usd")
    scale = (0.8, 0.8, 0.8)

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
class BasketB(_TabletopObject):
    name = "platform_pink"
    tags = ["object", "container"]
    usd_path = _object_usd("platform_pink", "platform_pink.usd")
    scale = (0.8, 0.8, 0.8)


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


class _KinematicMixin:
    """Spawn the object as a kinematic (fixed) body."""

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        import isaaclab.sim as sim_utils

        Object.__init__(
            self,
            name=self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            scale=self.scale,
            initial_pose=initial_pose,
            spawn_cfg_addon={"rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)},
        )


# --- quanta_x1 variants (suffix ``_2``) ----------------------------------------------
# The desktop-arm bricks / papers above differ from the ones used to collect the QUANTA_X1
# training data (manaenv library): bricks are 0.01 kg instead of 0.05 kg, papers are 1 mm
# sheets instead of ~19 mm slabs, and the paper colours differ. The ``_2`` assets below use
# the manaenv library USDs under ``assets/quanta_x1_objects``; the desktop assets stay as is.
_QUANTA_OBJECTS_DIR = os.path.join(ASSETS_DIR, "quanta_x1_objects")


def _quanta_object_usd(asset_dir: str, asset_file: str) -> str:
    return os.path.join(_QUANTA_OBJECTS_DIR, asset_dir, asset_file)


@register_asset
class YellowBrick2(YellowBrick):
    name = "yellow_brick_2"
    usd_path = _quanta_object_usd("brick", "brick_yellow.usd")


@register_asset
class GreenBrick2(GreenBrick):
    name = "green_brick_2"
    usd_path = _quanta_object_usd("brick", "brick_green.usd")


@register_asset
class RedBrick2(RedBrick):
    name = "red_brick_2"
    usd_path = _quanta_object_usd("brick", "brick_red.usd")


# Fixed (kinematic) papers. ``paper_pink.usd`` is paper_yellow with the tint estimated from
# the manaenv ``paper_pink`` thumbnail (the library copy is not available offline).
@register_asset
class YellowPaper2(_KinematicMixin, YellowPaper):
    name = "yellow_paper_2"
    usd_path = _quanta_object_usd("paper", "paper_yellow.usd")


@register_asset
class GreenPaper2(_KinematicMixin, GreenPaper):
    name = "green_paper_2"
    usd_path = _quanta_object_usd("paper", "paper_green.usd")


@register_asset
class PinkPaper2(_KinematicMixin, PinkPaper):
    name = "pink_paper_2"
    usd_path = _quanta_object_usd("paper", "paper_pink.usd")


# manaenv ``basket_b`` at scale 0.7 is the same mesh as ``platform_pink`` at 0.8 * 0.7 = 0.56.
@register_asset
class BasketB2(_KinematicMixin, BasketB):
    name = "platform_pink_2"
    scale = (0.56, 0.56, 0.56)
