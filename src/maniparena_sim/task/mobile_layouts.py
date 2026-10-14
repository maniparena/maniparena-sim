"""Scene layouts and reset ranges of the quanta_x1 mobile tasks.

The numbers match the reference data-collection setup, so evaluation sees the same scenes as
the collected demonstrations. Plain Python (no Isaac imports). Poses are
``(position_xyz, quaternion_xyzw)`` in the env frame; yaw-only quaternions are written as
literals.
"""

from __future__ import annotations

import math

_YAW_P90 = (0.0, 0.0, 0.707107, 0.707107)
_YAW_M90 = (0.0, 0.0, -0.707107, 0.707107)
_YAW_180 = (0.0, 0.0, 1.0, 0.0)
YAW_M90_RAD = -0.5 * math.pi

# The twelve arm joints that the retract / return stage compares with their defaults.
ARM_JOINT_NAMES = [f"{side}_arm_joint{index}" for side in ("left", "right") for index in range(1, 7)]
GRIPPER_JOINT_NAMES = ["left_arm_gripper", "right_arm_gripper"]
# Mobile ``retract_arm`` stage: grippers closed (atol 0.05), arms home (atol 0.3, 10 steps).
MOBILE_RETRACT = {
    "arm_joint_names": ARM_JOINT_NAMES,
    "arm_atol": 0.3,
    "gripper_joint_names": GRIPPER_JOINT_NAMES,
    "gripper_atol": 0.05,
    "retract_steps": 10,
}

# Floor-standing quanta_x1 (mobile base) initial joints.
MOBILE_LIFT_INIT = 0.379
MOBILE_HEAD_PITCH_INIT = -0.2618

# ----------------------------------------------------------------------------
# move_bottle_to_bin
# ----------------------------------------------------------------------------
BIN_TABLE_POSE = ((1.5, -0.95, 0.0), _YAW_P90)  # table_011, top plane z=0.7067
BIN_POSE = ((1.3, 0.06, 0.0), _YAW_M90)  # trash_can_a, load pose (reset re-samples XY)
BIN_BOTTLE_LOAD_POSE = ((1.45, -0.95, 0.86), (0.0, 0.0, 0.0, 1.0))
BIN_RESET = {
    "robot_start_x": (-0.7, 0.18),
    "robot_start_y": (-1.1, 1.1),
    # Bottle XY in the table-top support frame (x_axis=+Y, y_axis=-X).
    "bottle_local_x": (-0.25, 0.25),
    "bottle_local_y": (-0.05, 0.15),
    "support_frame": {
        "center": (1.5, -0.95, 0.7067),
        "x_axis": (0.0, 1.0, 0.0),
        "y_axis": (-1.0, 0.0, 0.0),
        "object_root_z": 0.86,
    },
    "bin_x": (1.35, 1.5),
    "bin_y": (0.1, 0.7),
    "bin_z": 0.0,
    "bin_quat_xyzw": _YAW_M90,
}
# Placement gate ("bottle_in_active_bin"): live bin AABB footprint.
BIN_SUCCESS = {
    "xy_inset_ratio": 0.1,
    "z_range": (-0.03, 0.22),
    "velocity_threshold": 0.03,
    "consecutive_steps": 5,
    "latch": True,
}

# ----------------------------------------------------------------------------
# move_fruit_to_basket
# ----------------------------------------------------------------------------
FRUIT_TABLE_POSE = ((1.5, 1.0, 0.0), _YAW_M90)
BASKET_TABLE_POSE = ((1.5, -0.95, 0.0), _YAW_M90)
TABLETOP_LOCAL_Z = 0.7505  # local top-surface anchor on table_011
FRUIT_RESET = {
    "robot_start_x": (-0.7, 0.18),
    "robot_start_y": (-1.1, 1.1),
    # Table-local surface ranges; the tables are yawed -90 deg.
    "fruit_local_x": (-0.26, 0.26),
    "fruit_local_y": (-0.12, 0.12),
    "basket_local_x": (-0.08, 0.08),
    "basket_local_y": (-0.06, 0.06),
    "table_yaw_rad": YAW_M90_RAD,
    "root_clearance_m": 0.004,
}
FRUIT_SUCCESS = dict(BIN_SUCCESS)  # fruit in live basket footprint, same gate numbers

# ----------------------------------------------------------------------------
# put_bottle_on_woodshelf (reset anchors live in configs/tasks/put_bottle_on_woodshelf/case_bank.yaml)
# ----------------------------------------------------------------------------
SHELF_TABLE_POSE = ((1.5, -0.95, 0.0), _YAW_P90)  # top plane z=0.7067, no raise
SHELF_POSE = ((0.63, -1.7907, 0.0), _YAW_180)
SHELF_BOTTLE_LOAD_POSE = ((1.505, -0.95, 0.905), (-0.01837, 0.007908, 0.000433, 0.9998))
# Placement gate in the shelf frame (place_bottle.yaml): box + upright, not latched.
SHELF_SUCCESS = {
    "x_range": (-0.53, 0.54),
    "y_range": (-0.1007, 0.0893),
    "z_range": (0.16, 0.85),
    "velocity_threshold": 0.03,
    "min_upright_dot": 0.95,
    "consecutive_steps": 5,
    "latch": False,
}
