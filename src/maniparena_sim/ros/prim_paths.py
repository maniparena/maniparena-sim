"""Centralized USD prim-path configuration for EX001 robot sensors."""

from dataclasses import dataclass


@dataclass
class EX001PathConfig:
    """Path configuration for EX001 robot sensors."""

    # Base robot path
    robot_base: str = "{ENV_REGEX_NS}/Robot"

    # Camera paths
    chassis_front_color: str = (
        "{ENV_REGEX_NS}/Robot/camera_chassis_front_color_optical_frame/Chassis_Front_Color_Camera"
    )
    chassis_front_depth: str = (
        "{ENV_REGEX_NS}/Robot/camera_chassis_front_depth_optical_frame/Chassis_Front_Depth_Camera"
    )
    head_front_color: str = "{ENV_REGEX_NS}/Robot/camera_head_front_color_optical_frame/Head_Front_Color_Camera"
    left_gripper: str = "{ENV_REGEX_NS}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera"
    right_gripper: str = "{ENV_REGEX_NS}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera"

    # IMU path
    imu: str = "{ENV_REGEX_NS}/Robot/imu_link"

    # 2D RTX lidar prim (this USD uses the ``a_d_laser`` prim).
    lidar: str = "{ENV_REGEX_NS}/Robot/a_d_laser/Laser"


EX001_PATHS = EX001PathConfig()
