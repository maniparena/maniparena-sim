"""QUANTA_X1 SDK ROS 2 topic names.

These names match ``ROS2_SDK_Mapping_QUANTA_X1.md``.

Meaning of remapped names (do not reuse a name for a different quantity):

- ``/tracked_pose``: robot chassis/root world pose (SDK ``get_pose_stream``).
  Not obstacle poses and not a spawn command.
- ``/odom``: chassis odometry from root motion.
- ``/left_arm/end_pose`` / ``/right_arm/end_pose``: gripper base-link pose.
- ``/camera1/...`` / ``/camera3/...``: left / right wrist RGB.
- ``/camera_head_front/...``: head RGB / depth.
- ``/camera_chassis_front/depth/points``: chassis depth point cloud.
"""

from __future__ import annotations

QUANTA_X1_SDK_PUBLISH_TOPICS: frozenset[str] = frozenset(
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

QUANTA_X1_SDK_SUBSCRIBE_TOPICS: frozenset[str] = frozenset(
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
