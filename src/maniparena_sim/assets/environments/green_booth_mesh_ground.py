"""Green booth mesh-ground background for ex001 tabletop tasks."""

from __future__ import annotations

import os

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from maniparena_sim.assets import ASSETS_DIR


@register_asset
class GreenBoothMeshGroundBackground(LibraryBackground):
    name = "green_booth_mesh_ground"
    tags = ["background"]
    usd_path = os.path.join(ASSETS_DIR, "green_booth", "green_booth_mesh_ground.usd")
    initial_pose = Pose(
        position_xyz=(-0.452, -0.271, -0.985),
        rotation_xyzw=(0.0, 0.0, -0.704, 0.709),
    )
    object_min_z = -0.4

    def __init__(self):
        super().__init__()
