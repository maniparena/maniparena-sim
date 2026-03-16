"""Green booth background registration for opencvpr tasks."""

from __future__ import annotations

import os

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from opencvpr.assets import ASSETS_DIR


@register_asset
class GreenBoothBackground(LibraryBackground):
    name = "green_booth"
    tags = ["background"]
    usd_path = os.path.join(ASSETS_DIR, "green_booth", "green_booth.usd")
    initial_pose = Pose(
        position_xyz=(0.0, 0.0, 0.0),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    object_min_z = -0.4

    def __init__(self):
        super().__init__()
