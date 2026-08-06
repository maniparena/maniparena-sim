"""ROS message-building configuration for the EX001 navigation robot.

Camera ``rgb_key`` / ``depth_key`` map to the embodiment's ``camera_obs`` keys
(``left_wrist_cam``, ``right_wrist_cam``, ``head_cam`` RGB, ``chassis_cam``
depth). The logical camera names below are keyed by the ROS topic they feed.
"""


class EX001RosConfig:
    """Config bundle for the EX001 robot."""

    CAMERA_CONFIG = {
        # /camera1/usb_cam1/image_raw/image_compressed  <- left wrist RGB
        "camera1": {
            "scene_entity": "left_wrist_camera",
            "rgb_key": "left_wrist_cam",
            "depth_key": None,
            "frame_id": "left_arm_gripper_camera_color_frame",
            "compress_fmt": "jpeg",
            "compress_quality": 80,
            "pointcloud_downsample": 1,
        },
        # /camera3/usb_cam3/image_raw/image_compressed  <- right wrist RGB
        "camera3": {
            "scene_entity": "right_wrist_camera",
            "rgb_key": "right_wrist_cam",
            "depth_key": None,
            "frame_id": "right_arm_gripper_camera_color_frame",
            "compress_fmt": "jpeg",
            "compress_quality": 80,
            "pointcloud_downsample": 1,
        },
        # /camera_head_front/color/image_raw/compressed  <- head RGB
        "head_front_color_camera": {
            "scene_entity": "head_camera",
            "rgb_key": "head_cam",
            "depth_key": None,
            "frame_id": "camera_head_front_color_optical_frame",
            "compress_fmt": "jpeg",
            "compress_quality": 80,
            "pointcloud_downsample": 1,
        },
        # /camera_chassis_front/depth/points  <- chassis depth
        "chassis_front_camera": {
            "scene_entity": "chassis_camera",
            "rgb_key": None,
            "depth_key": "chassis_cam",
            "frame_id": "camera_chassis_front_depth_optical_frame",
            "compress_fmt": "png",
            "compress_quality": None,
            "pointcloud_downsample": 1,
        },
    }

    IMU_CONFIG = {
        "chassis_imu": {
            "frame_id": "imu_link",
            "obs_ang_vel_key": "imu_ang_vel",
            "obs_lin_acc_key": "imu_lin_acc",
            "obs_orientation_key": "imu_orientation",
            "identity_orientation": (0.0, 0.0, 0.0, 1.0),
            "zero_covariance": [0.0] * 9,
        }
    }

    ODOM_CONFIG = {
        "chassis_odom": {
            "frame_id": "odom",
            "child_frame_id": "base_link",
            "obs_root_pos_key": "root_pos_w",
            "obs_root_quat_key": "root_quat_w",
            "obs_lin_vel_key": "root_lin_vel_w",
            "obs_ang_vel_key": "root_ang_vel_w",
        }
    }

    LIDAR_CONFIG = {
        "chassis_lidar": {
            "frame_id": "a_d_laser",
            "invalid_range_value": float("inf"),
        }
    }

    CHASSIS_CONTROL_CONFIG = {
        "control_mode": "differential",
        "wheel_joint_names": ("left_wheel_joint", "right_wheel_joint"),
        # Sim-effective EX001 geometry (older nominal 0.078 / 0.48 over-drives).
        "wheel_radius": 0.084,
        "wheel_track_width": 0.458,
    }

    KEYBOARD_MOVEMENT_CONFIG = {
        "twist_topic": "/chassis/cmd_vel",
        "qos_depth": 10,
    }


EX001RosConfig = EX001RosConfig()

ROS_QOS_CONFIG = {
    "default_publisher_depth": 10,
    "default_subscriber_depth": 10,
}
