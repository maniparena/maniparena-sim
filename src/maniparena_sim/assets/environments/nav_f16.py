"""nav_f16 open-scene background for ex001 teleop and navigation."""

from __future__ import annotations

import os

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.assets import ASSETS_DIR


@register_asset
class NavF16Background(LibraryBackground):
    name = "nav_f16"
    tags = ["background"]
    usd_path = os.path.join(ASSETS_DIR, "nav_f16", "nav_f16.usd")
    initial_pose = Pose(
        position_xyz=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )
    object_min_z = -10.0

    def __init__(self):
        super().__init__()
