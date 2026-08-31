"""QUANTA_X1 data acquirer functions for the SDK ROS2 bridge.

Each acquirer has the signature ``(obs, extras, ...) -> ROS msg`` and is
registered by :func:`fill_data_acquirer` into the communicator's data-acquirer
dict. Camera *scene* keys stay ManipArena-local; SDK *topic* names are fixed.
"""

from typing import Any, Callable

from maniparena_sim.ros.message_builder import MessageBuilder
from maniparena_sim.ros.ros2_config import QuantaX1RosConfig
from maniparena_sim.ros.sim_utils import camera_cache

_ODOM_CFG = QuantaX1RosConfig.ODOM_CONFIG["chassis_odom"]
_IMU_CFG = QuantaX1RosConfig.IMU_CONFIG["chassis_imu"]
_CAMERA_CONFIG = QuantaX1RosConfig.CAMERA_CONFIG


def bind_with_dynamic_stamp(
    acquirer: Callable[..., Any],
    stamp_holder: dict,
    *bound_args: Any,
) -> Callable[[Any, Any], Any]:
    """Return a callback that fetches the stamp at call-time."""

    def callback(obs, extras):
        return acquirer(obs, extras, *bound_args, stamp_holder["stamp"])

    return callback


def acquirer_all_joint_states(obs, extras, joint_mapping, stamp):
    names = list(joint_mapping.joint_names)
    indices = list(range(len(names)))
    return MessageBuilder.joint_states(obs, indices, names, stamp)


def acquirer_arm_joint_states(obs, extras, joint_indices, joint_names, stamp):
    return MessageBuilder.joint_states(obs, list(joint_indices), list(joint_names), stamp)


def acquirer_head_joint_states(obs, extras, joint_mapping, stamp):
    return MessageBuilder.joint_states(
        obs,
        [joint_mapping.head_pitch[0], joint_mapping.head_yaw[0]],
        ["head_pitch_joint", "head_yaw_joint"],
        stamp,
    )


def acquirer_lift_joint_states(obs, extras, joint_mapping, stamp):
    return MessageBuilder.joint_states(obs, [joint_mapping.lift[0]], ["lift_joint"], stamp)


def acquirer_gripper_joint_states(obs, extras, joint_index, joint_name, stamp):
    return MessageBuilder.joint_states(obs, [joint_index], [joint_name], stamp)


def acquirer_arm_end_pose(obs, extras, body_index, stamp):
    return MessageBuilder.pose_stamped_from_body(obs, body_index, stamp)


def acquirer_chassis_imu(obs, extras, stamp):
    return MessageBuilder.imu(obs, stamp, _IMU_CFG)


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
    return MessageBuilder.compressed_rgb(obs, "head_camera", stamp, _CAMERA_CONFIG)


def acquirer_camera_head_front_depth_compressed(obs, extras, stamp):
    return MessageBuilder.compressed_depth(obs, "head_camera", stamp, _CAMERA_CONFIG, camera_cache)


def acquirer_chassis_scan(obs, extras, lidar_2d, stamp):
    """Build ``/scan`` from RTX GenericModelOutput using the bridge stamp."""
    del obs, extras
    if lidar_2d is None:
        return None
    return lidar_2d.build_laserscan(stamp)


def fill_data_acquirer(
    data_acquirer,
    joint_mapping,
    stamp_holder,
    odom_origin,
    lidar_2d=None,
):
    """Register SDK data acquirer callbacks into *data_acquirer* dict.

    ``/scan`` is published through the main communicator when ``lidar_2d`` is
    available, so it shares the same timestamp as ``/clock`` and TF.
    """
    s = stamp_holder

    data_acquirer["/joint_states"] = bind_with_dynamic_stamp(acquirer_all_joint_states, s, joint_mapping)
    data_acquirer["/left_arm/joint_states"] = bind_with_dynamic_stamp(
        acquirer_arm_joint_states,
        s,
        getattr(joint_mapping, "left_arm", ()),
        [f"left_arm_joint{i}" for i in range(1, 7)],
    )
    data_acquirer["/right_arm/joint_states"] = bind_with_dynamic_stamp(
        acquirer_arm_joint_states,
        s,
        getattr(joint_mapping, "right_arm", ()),
        [f"right_arm_joint{i}" for i in range(1, 7)],
    )
    data_acquirer["/head/joint_states"] = bind_with_dynamic_stamp(acquirer_head_joint_states, s, joint_mapping)
    data_acquirer["/lift/joint_states"] = bind_with_dynamic_stamp(acquirer_lift_joint_states, s, joint_mapping)
    left_gripper = getattr(joint_mapping, "left_gripper", [0])
    right_gripper = getattr(joint_mapping, "right_gripper", [0])
    data_acquirer["/left_gripper/joint_states"] = bind_with_dynamic_stamp(
        acquirer_gripper_joint_states, s, left_gripper[0], "left_arm_gripper"
    )
    data_acquirer["/right_gripper/joint_states"] = bind_with_dynamic_stamp(
        acquirer_gripper_joint_states, s, right_gripper[0], "right_arm_gripper"
    )
    left_ee = getattr(joint_mapping, "left_gripper_body", [0])
    right_ee = getattr(joint_mapping, "right_gripper_body", [0])
    data_acquirer["/left_arm/end_pose"] = bind_with_dynamic_stamp(acquirer_arm_end_pose, s, int(left_ee[0]))
    data_acquirer["/right_arm/end_pose"] = bind_with_dynamic_stamp(acquirer_arm_end_pose, s, int(right_ee[0]))
    data_acquirer["/odom"] = bind_with_dynamic_stamp(acquirer_odom, s, odom_origin)
    data_acquirer["/tracked_pose"] = bind_with_dynamic_stamp(acquirer_tracked_pose, s)
    data_acquirer["/hal/chassis/imu"] = bind_with_dynamic_stamp(acquirer_chassis_imu, s)
    if lidar_2d is not None:
        data_acquirer["/scan"] = bind_with_dynamic_stamp(acquirer_chassis_scan, s, lidar_2d)

    data_acquirer["/camera_chassis_front/depth/points"] = bind_with_dynamic_stamp(
        acquirer_camera_chassis_front_depth_points, s
    )
    data_acquirer["/camera_head_front/color/image_raw/compressed"] = bind_with_dynamic_stamp(
        acquirer_camera_head_front_color_compressed, s
    )
    data_acquirer["/camera_head_front/depth/image_raw/compressedDepth"] = bind_with_dynamic_stamp(
        acquirer_camera_head_front_depth_compressed, s
    )
    data_acquirer["/camera1/usb_cam1/image_raw/image_compressed"] = bind_with_dynamic_stamp(
        acquirer_camera1_image_compressed, s
    )
    data_acquirer["/camera3/usb_cam3/image_raw/image_compressed"] = bind_with_dynamic_stamp(
        acquirer_camera3_image_compressed, s
    )
