"""quanta_x1 fixed-base tabletop tasks (sort_blocks, fruits_to_basket).

The desktop-arm versions of these tasks keep their own layout in ``scene_builder``; the numbers
below only apply when the robot is ``quanta_x1`` and match the reference data-collection setup.
The booth is the existing ManipArena booth (floor + table top at world z=-0.235).
"""

from __future__ import annotations

# Robot root (yaw 0, free root, never randomized), spawn lift and head pitch (positive = down).
TABLETOP_ROBOT = {
    "sort_blocks": {"pos": (-0.52, -0.26, -0.985), "lift": 0.455, "head_pitch": 0.35},
    "fruits_to_basket": {"pos": (-0.60, -0.26, -0.985), "lift": 0.379, "head_pitch": 0.35},
}

# --- sort_blocks -------------------------------------------------------------------
# Bricks rest on the table top (z fixed); resets draw all three jointly from one absolute
# region. Papers are fixed ManipArena poses (scale 0.8, yaw +90 deg) 1.5 mm above the table.
SORT_BRICK_Z = -0.234  # brick USD origin is at its bottom: rests on the table top (-0.235)
SORT_BRICK_LOAD_POSES = [  # yellow, green, red (inside the region, apart by >= 0.11)
    (0.05, -0.16, SORT_BRICK_Z),
    (0.05, -0.26, SORT_BRICK_Z),
    (0.05, -0.36, SORT_BRICK_Z),
]
SORT_BLOCK_REGION = {
    "x_range": (-0.04, 0.06),
    "y_range": (-0.46, -0.06),
    "yaw_range": (-0.3, 0.3),
    "min_separation": 0.11,
}
SORT_PAPER_POSES = [
    (0.236, 0.02, -0.2335),
    (0.238, -0.25, -0.2335),
    (0.240, -0.52, -0.2335),
]
# Place gates (per brick on its paper) and the final "return both arms home" stage.
SORT_PLACE_GATE = {
    "xy_inset_ratio": 0.05,
    "z_range": (-0.02, 0.08),
    "velocity_threshold": 0.05,
    "consecutive_steps": 5,
    "latch": True,
}

# --- fruits_to_basket --------------------------------------------------------------
# Four slots (two per side of the basket: near x=0.04, far x=0.16 / 0.20). Every reset gives apple, banana,
# pear and the bread distractor a random permutation of the slots, plus a small XY jitter and a yaw
# offset. Bread is never a success target.
FRUIT_Z = -0.2348
FRUIT_SLOTS = [
    (0.04, 0.10, FRUIT_Z),
    (0.16, 0.03, FRUIT_Z),
    (0.04, -0.62, FRUIT_Z),
    (0.20, -0.55, FRUIT_Z),
]
FRUIT_RESET = {
    "position_jitter": ((-0.01, 0.01), (-0.01, 0.01)),
    "yaw_range": (-0.4, 0.4),
}
# Keep the scene's nominal poses separate from the final reset slots. The remote
# vnext scene still names banana at x=0.20; v11 only moved its reset slot to x=0.16.
FRUIT_NOMINAL_LOAD_POSES = {
    "apple": (0.04, 0.10, FRUIT_Z),
    "banana": (0.20, 0.03, FRUIT_Z),
    "pear": (0.04, -0.62, FRUIT_Z),
    "bread": (0.20, -0.55, FRUIT_Z),
}
# QUANTA_X1 collection basket: scale 0.7 == platform_pink x 0.56 (asset ``platform_pink_2``), kinematic, yaw +90 deg.
QUANTA_BASKET_POSE = (0.11, -0.26, -0.232)
FRUIT_PLACE_GATE = {
    "xy_inset_ratio": 0.05,
    "z_range": (-0.03, 0.22),
    "velocity_threshold": None,
    "consecutive_steps": 5,
    "latch": True,
}

# --- shared "return both arms home" stage (articulation_joints_near_default) -------------
TABLETOP_RETRACT = {"arm_atol": 0.15, "retract_steps": 10}
