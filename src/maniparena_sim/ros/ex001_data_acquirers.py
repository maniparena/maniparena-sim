"""EX001 data acquirer functions for the ROS2 navigation bridge.

Each acquirer has the signature ``(obs, extras, ...) -> ROS msg`` and is
registered by :func:`fill_data_acquirer` into the communicator's data-acquirer
dict.
"""

from typing import Any, Callable

from maniparena_sim.ros.message_builder import MessageBuilder
from maniparena_sim.ros.ros2_config import EX001RosConfig
from maniparena_sim.ros.sim_utils import camera_cache

_ODOM_CFG = EX001RosConfig.ODOM_CONFIG["chassis_odom"]
_IMU_CFG = EX001RosConfig.IMU_CONFIG["chassis_imu"]
_LIDAR_CFG = EX001RosConfig.LIDAR_CONFIG["chassis_lidar"]
_CAMERA_CONFIG = EX001RosConfig.CAMERA_CONFIG


def bind_with_dynamic_stamp(
    acquirer: Callable[..., Any],
    stamp_holder: dict,
    *bound_args: Any,
) -> Callable[[Any, Any], Any]:
    """Return a callback that fetches the stamp at call-time."""

    def callback(obs, extras):
        return acquirer(obs, extras, *bound_args, stamp_holder["stamp"])

    return callback


# ── Individual acquirer functions ────────────────────────────────────────────


def acquirer_mock_robot_state(obs, extras, joint_mapping, stamp):
    indices = joint_mapping.all_controlled_indices
    names = [joint_mapping.joint_names[i] for i in indices]
    return MessageBuilder.joint_states(obs, indices, names, stamp)


def acquirer_head_joint_states(obs, extras, joint_mapping, stamp):
    return MessageBuilder.joint_states(
        obs,
        [joint_mapping.head_pitch[0], joint_mapping.head_yaw[0]],
        ["head_pitch_joint", "head_yaw_joint"],
        stamp,
    )


def acquirer_chassis_odom(obs, extras, odom_origin, stamp):
    return MessageBuilder.odom(obs, stamp, odom_origin, _ODOM_CFG)


def acquirer_chassis_imu(obs, extras, stamp):
    return MessageBuilder.imu(obs, stamp, _IMU_CFG)


def acquirer_scan(obs, extras, lidar, stamp):
    return MessageBuilder.laserscan(lidar, stamp, _LIDAR_CFG)


def acquirer_tracked_pose(obs, extras, stamp):
    return MessageBuilder.pose_stamped_from_root(obs, stamp)


def acquirer_odom(obs, extras, odom_origin, stamp):
    return MessageBuilder.odom(obs, stamp, odom_origin, _ODOM_CFG)


def acquirer_camera_chassis_front_depth_points(obs, extras, stamp):
    return MessageBuilder.depth_pointcloud(obs, "chassis_front_camera", stamp, _CAMERA_CONFIG, camera_cache)


def acquirer_camera1_image_compressed(obs, extras, stamp):
    return MessageBuilder.compressed_rgb(obs, "camera1", stamp, _CAMERA_CONFIG)


def acquirer_camera3_image_compressed(obs, extras, stamp):
    return MessageBuilder.compressed_rgb(obs, "camera3", stamp, _CAMERA_CONFIG)


def acquirer_camera_head_front_color_compressed(obs, extras, stamp):
    return MessageBuilder.compressed_rgb(obs, "head_front_color_camera", stamp, _CAMERA_CONFIG)


# ── Registration ─────────────────────────────────────────────────────────────


def fill_data_acquirer(
    data_acquirer,
    joint_mapping,
    stamp_holder,
    odom_origin,
    lidar_2d,
    env,
):
    """Register all data acquirer callbacks into *data_acquirer* dict (2D nav)."""
    s = stamp_holder

    data_acquirer["/mock_robot_interface/state"] = bind_with_dynamic_stamp(acquirer_mock_robot_state, s, joint_mapping)
    data_acquirer["/head/joint_states"] = bind_with_dynamic_stamp(acquirer_head_joint_states, s, joint_mapping)
    data_acquirer["/chassis/odom"] = bind_with_dynamic_stamp(acquirer_chassis_odom, s, odom_origin)
    data_acquirer["/hal/chassis/imu"] = bind_with_dynamic_stamp(acquirer_chassis_imu, s)
    data_acquirer["/scan"] = bind_with_dynamic_stamp(acquirer_scan, s, lidar_2d)
    data_acquirer["/tracked_pose"] = bind_with_dynamic_stamp(acquirer_tracked_pose, s)
    data_acquirer["/odom"] = bind_with_dynamic_stamp(acquirer_odom, s, odom_origin)

    # Camera streams.
    data_acquirer["/camera_chassis_front/depth/points"] = bind_with_dynamic_stamp(
        acquirer_camera_chassis_front_depth_points, s
    )
    data_acquirer["/camera1/usb_cam1/image_raw/image_compressed"] = bind_with_dynamic_stamp(
        acquirer_camera1_image_compressed, s
    )
    data_acquirer["/camera3/usb_cam3/image_raw/image_compressed"] = bind_with_dynamic_stamp(
        acquirer_camera3_image_compressed, s
    )
    data_acquirer["/camera_head_front/color/image_raw/compressed"] = bind_with_dynamic_stamp(
        acquirer_camera_head_front_color_compressed, s
    )
