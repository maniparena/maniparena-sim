"""EX001 SDK ROS 2 topic names and gripper scale.

Copied from ManaEnv ``manaenv/ros_utils/ex001_sdk_topics.py`` (SDK surface
commit). These names match ``ROS2_SDK_Mapping_EX001.md``. The navigation /
ROS bridge must not publish or subscribe ManaEnv-private names such as
``/manaenv/*`` or ``/mock_robot_interface/*``.

Meaning of remapped names (do not reuse a name for a different quantity):

- ``/tracked_pose``: robot chassis/root world pose (SDK ``get_pose_stream``).
  Not obstacle poses and not a spawn command.
- ``/odom``: chassis odometry from root motion. Not the old ``/chassis/odom``.
- ``/left_arm/end_pose`` / ``/right_arm/end_pose``: gripper base-link pose.
- ``/camera1/...`` / ``/camera3/...``: left / right wrist RGB.
- ``/camera_head_front/...``: head RGB / depth.
- ``/camera_chassis_front/depth/points``: chassis depth point cloud.
"""

from __future__ import annotations

EX001_SDK_GRIPPER_MIN = 0.0
EX001_SDK_GRIPPER_MAX = 4.5
EX001_SIM_GRIPPER_MIN = 0.0
EX001_SIM_GRIPPER_MAX = 1.89

EX001_SDK_PUBLISH_TOPICS: frozenset[str] = frozenset(
    {
        "/head/joint_states",
        "/odom",
        "/tracked_pose",
        "/joint_states",
        "/tf_static",
        "/hal/chassis/imu",
        "/scan",
        "/camera_chassis_front/depth/points",
        "/camera_head_front/color/image_raw/compressed",
        "/camera_head_front/depth/image_raw/compressedDepth",
        "/camera1/usb_cam1/image_raw/image_compressed",
        "/camera3/usb_cam3/image_raw/image_compressed",
        "/left_arm/joint_states",
        "/right_arm/joint_states",
        "/left_arm/end_pose",
        "/right_arm/end_pose",
        "/left_gripper/joint_states",
        "/right_gripper/joint_states",
        "/lift/joint_states",
    }
)

EX001_SDK_SUBSCRIBE_TOPICS: frozenset[str] = frozenset(
    {
        "/head_position_controller/commands",
        "/chassis/cmd_vel",
        "/left_arm_joint_controller/commands",
        "/right_arm_joint_controller/commands",
        "/left_gripper_controller/commands",
        "/right_gripper_controller/commands",
        "/lift_position_controller/commands",
    }
)

_BANNED_NAME_PREFIXES = ("/manaenv/", "/mock_robot_interface/", "/livox/")
_BANNED_EXACT = frozenset({"/chassis/odom"})


def iter_banned_sdk_topics(topics: set[str] | frozenset[str]) -> list[str]:
    """Return topic names that must not appear on the EX001 SDK surface."""
    banned: list[str] = []
    for topic in topics:
        if topic in _BANNED_EXACT or topic.startswith(_BANNED_NAME_PREFIXES):
            banned.append(topic)
    return sorted(banned)


def sdk_gripper_to_sim(value: float) -> float:
    """Map SDK gripper range ``0.0–4.5`` onto the EX001 sim joint ``0.0–1.89``."""
    span = EX001_SDK_GRIPPER_MAX - EX001_SDK_GRIPPER_MIN
    if span <= 0.0:
        return EX001_SIM_GRIPPER_MIN
    ratio = (float(value) - EX001_SDK_GRIPPER_MIN) / span
    ratio = min(max(ratio, 0.0), 1.0)
    return EX001_SIM_GRIPPER_MIN + ratio * (EX001_SIM_GRIPPER_MAX - EX001_SIM_GRIPPER_MIN)


def sim_gripper_to_sdk(value: float) -> float:
    """Map the EX001 sim gripper joint onto SDK range ``0.0–4.5``."""
    span = EX001_SIM_GRIPPER_MAX - EX001_SIM_GRIPPER_MIN
    if span <= 0.0:
        return EX001_SDK_GRIPPER_MIN
    ratio = (float(value) - EX001_SIM_GRIPPER_MIN) / span
    ratio = min(max(ratio, 0.0), 1.0)
    return EX001_SDK_GRIPPER_MIN + ratio * (EX001_SDK_GRIPPER_MAX - EX001_SDK_GRIPPER_MIN)
