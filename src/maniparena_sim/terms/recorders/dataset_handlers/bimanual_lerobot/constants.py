"""Constants for Bimanual direct LeRobot export."""

from __future__ import annotations

import numpy as np

CAMERA_NAME_MAPPING = {"head_cam": "faceImg", "left_wrist_cam": "leftImg", "right_wrist_cam": "rightImg"}
LEROBOT_CAMERA_COLUMN_MAPPING = {value: f"observation.images.{value}" for value in CAMERA_NAME_MAPPING.values()}
ACTIVE_JOINT_INDICES = np.asarray([0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 15], dtype=np.int64)
GRIPPER_CLIP_MIN = 0.0
GRIPPER_CLIP_MAX = 4.5
VECTOR_SIZE = 14
ROBOT_TYPE = "bimanual"
JOINT_VECTOR_NAMES = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_gripper",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_gripper",
]
EE_VECTOR_NAMES = [
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "left_rot_x",
    "left_rot_y",
    "left_rot_z",
    "left_gripper",
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "right_rot_x",
    "right_rot_y",
    "right_rot_z",
    "right_gripper",
]
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CHUNK_SIZE = 1000
