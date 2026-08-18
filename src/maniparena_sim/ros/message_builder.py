"""ROS2 message builders for converting simulation data to ROS messages.

All methods are static. Config dicts are passed as parameters so the builder
is not tied to a particular robot model.
"""

import numpy as np

from maniparena_sim.ros.constants import DEPTH_MAX_MM, FLOAT_EPS, RGB_CHANNELS, RGB_MAX_VALUE, STANDARD_GRAVITY
from maniparena_sim.ros.math_utils import (
    meters_to_mm,
    quat_inverse,
    quat_multiply,
    quat_rotate_inverse,
    to_numpy,
    to_numpy_2d,
)
from maniparena_sim.ros.stamp_utils import copy_stamp


class MessageBuilder:
    """Build ROS2 messages from scene observation data."""

    # ── Odometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def odom(obs, stamp, odom_origin, odom_cfg):
        """Build Odometry message relative to odom origin.

        Args:
            obs: Observation dict with ``policy`` group.
            stamp: ROS ``builtin_interfaces/Time`` stamp.
            odom_origin: ``OdomOrigin`` instance.
            odom_cfg: Single-entry config dict, e.g.
                ``EX001RosConfig.ODOM_CONFIG["chassis_odom"]``.
        """
        from geometry_msgs.msg import Point, Quaternion, Vector3
        from nav_msgs.msg import Odometry

        policy_obs = obs.get("policy", obs)

        root_pos = to_numpy(policy_obs[odom_cfg["obs_root_pos_key"]][0])
        root_quat = to_numpy(policy_obs[odom_cfg["obs_root_quat_key"]][0])
        lin_vel = to_numpy(policy_obs[odom_cfg["obs_lin_vel_key"]][0])
        ang_vel = to_numpy(policy_obs[odom_cfg["obs_ang_vel_key"]][0])

        pos_diff_world = root_pos - odom_origin.position
        position = quat_rotate_inverse(odom_origin.orientation, pos_diff_world)
        orientation = quat_multiply(quat_inverse(odom_origin.orientation), root_quat)

        msg = Odometry()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = odom_cfg["frame_id"]
        msg.child_frame_id = odom_cfg["child_frame_id"]

        msg.pose.pose.position = Point(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )
        msg.pose.pose.orientation = Quaternion(
            x=float(orientation[0]),
            y=float(orientation[1]),
            z=float(orientation[2]),
            w=float(orientation[3]),
        )
        msg.twist.twist.linear = Vector3(
            x=float(lin_vel[0]),
            y=float(lin_vel[1]),
            z=float(lin_vel[2]),
        )
        msg.twist.twist.angular = Vector3(
            x=float(ang_vel[0]),
            y=float(ang_vel[1]),
            z=float(ang_vel[2]),
        )
        return msg

    # ── IMU ───────────────────────────────────────────────────────────────────

    @staticmethod
    def imu(obs, stamp, imu_cfg):
        """Build IMU message from observation data.

        Note: linear_acceleration is published in g (gravitational acceleration), not m/s².

        Args:
            imu_cfg: Single-entry config dict, e.g.
                ``EX001RosConfig.IMU_CONFIG["chassis_imu"]``.
        """
        from sensor_msgs.msg import Imu as RosImu

        policy_obs = obs.get("policy", obs)
        ang_vel = to_numpy(policy_obs[imu_cfg["obs_ang_vel_key"]][0])
        lin_acc = to_numpy(policy_obs[imu_cfg["obs_lin_acc_key"]][0])
        orientation_key = imu_cfg.get("obs_orientation_key")
        quat_w = (
            to_numpy(policy_obs[orientation_key][0])
            if orientation_key and orientation_key in policy_obs
            else np.asarray(imu_cfg["identity_orientation"], dtype=np.float32)
        )

        msg = RosImu()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = imu_cfg["frame_id"]

        msg.angular_velocity.x = float(ang_vel[0])
        msg.angular_velocity.y = float(ang_vel[1])
        msg.angular_velocity.z = float(ang_vel[2])
        msg.linear_acceleration.x = float(lin_acc[0] / STANDARD_GRAVITY)
        msg.linear_acceleration.y = float(lin_acc[1] / STANDARD_GRAVITY)
        msg.linear_acceleration.z = float(lin_acc[2] / STANDARD_GRAVITY)

        msg.orientation.x = float(quat_w[0])
        msg.orientation.y = float(quat_w[1])
        msg.orientation.z = float(quat_w[2])
        msg.orientation.w = float(quat_w[3])
        msg.orientation_covariance = imu_cfg["zero_covariance"]
        msg.angular_velocity_covariance = imu_cfg["zero_covariance"]
        msg.linear_acceleration_covariance = imu_cfg["zero_covariance"]

        return msg

    # ── LaserScan ─────────────────────────────────────────────────────────────

    @staticmethod
    def laserscan(
        template,
        *,
        stamp_ns,
        frame_id,
        scan_rate_hz,
        range_min,
        range_max,
        ranges,
        intensities,
    ):
        """Build a complete LaserScan from assembled RTX lidar beams."""
        from sensor_msgs.msg import LaserScan

        num_beams = len(ranges)
        scan_time = 1.0 / scan_rate_hz
        msg = LaserScan()
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        msg.header.frame_id = frame_id
        msg.angle_min = template.angle_min
        msg.angle_max = template.angle_max
        msg.angle_increment = template.angle_increment
        msg.time_increment = scan_time / max(num_beams, 1)
        msg.scan_time = scan_time
        msg.range_min = range_min
        msg.range_max = range_max
        msg.ranges = list(ranges)
        msg.intensities = list(intensities)
        return msg

    # ── JointState ────────────────────────────────────────────────────────────

    @staticmethod
    def joint_states(obs, joint_indices, joint_names, stamp):
        """Build JointState message."""
        from sensor_msgs.msg import JointState

        joint_pos = obs["policy"]["joint_positions"][0]
        joint_vel = obs["policy"]["joint_velocities"][0]
        joint_effort = obs["policy"]["joint_efforts"][0]

        msg = JointState()
        copy_stamp(msg.header.stamp, stamp)
        msg.name = joint_names
        msg.position = [float(joint_pos[idx]) for idx in joint_indices]
        msg.velocity = [float(joint_vel[idx]) for idx in joint_indices]
        msg.effort = [float(joint_effort[idx]) for idx in joint_indices]

        return msg

    # ── PoseStamped ───────────────────────────────────────────────────────────

    @staticmethod
    def pose_stamped_from_root(obs, stamp, frame_id="world"):
        """Build PoseStamped from robot root pose."""
        from geometry_msgs.msg import Point
        from geometry_msgs.msg import Pose as RosPose
        from geometry_msgs.msg import PoseStamped, Quaternion

        root_pos = obs["policy"]["root_pos_w"][0]
        root_quat = obs["policy"]["root_quat_w"][0]

        msg = PoseStamped()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = frame_id

        msg.pose = RosPose()
        msg.pose.position = Point(
            x=float(root_pos[0]),
            y=float(root_pos[1]),
            z=float(root_pos[2]),
        )
        msg.pose.orientation = Quaternion(
            x=float(root_quat[0]),
            y=float(root_quat[1]),
            z=float(root_quat[2]),
            w=float(root_quat[3]),
        )
        return msg

    @staticmethod
    def obstacle_poses(env, obstacle_keys, stamp, frame_id="world"):
        """Build PoseArray from obstacle root poses across one or more scene entities."""
        from geometry_msgs.msg import Pose as RosPose
        from geometry_msgs.msg import PoseArray

        msg = PoseArray()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = frame_id

        for key in obstacle_keys:
            asset = env.scene[key]
            pos_w = to_numpy(asset.data.root_pos_w[0])
            quat_w = to_numpy(asset.data.root_quat_w[0])
            if pos_w.ndim == 1:
                pos_w = pos_w.reshape(1, -1)
                quat_w = quat_w.reshape(1, -1)
            for p, q in zip(pos_w, quat_w):
                pose = RosPose()
                pose.position.x = float(p[0])
                pose.position.y = float(p[1])
                pose.position.z = float(p[2])
                pose.orientation.x = float(q[0])
                pose.orientation.y = float(q[1])
                pose.orientation.z = float(q[2])
                pose.orientation.w = float(q[3])
                msg.poses.append(pose)
        return msg

    @staticmethod
    def pose_stamped_from_body(obs, body_idx, stamp, frame_id="world"):
        """Build PoseStamped from a specific body pose.

        ``policy["body_pose_w"]`` is stored flat as ``(N, num_bodies * 7)``.
        """
        from geometry_msgs.msg import Point
        from geometry_msgs.msg import Pose as RosPose
        from geometry_msgs.msg import PoseStamped, Quaternion

        from maniparena_sim.ros.sim_utils import get_body_pose

        body_pos, body_quat = get_body_pose(obs, body_idx)
        if body_pos is None or body_quat is None or len(body_pos) < 3 or len(body_quat) < 4:
            return None

        msg = PoseStamped()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = frame_id

        msg.pose = RosPose()
        msg.pose.position = Point(
            x=float(body_pos[0]),
            y=float(body_pos[1]),
            z=float(body_pos[2]),
        )
        msg.pose.orientation = Quaternion(
            x=float(body_quat[0]),
            y=float(body_quat[1]),
            z=float(body_quat[2]),
            w=float(body_quat[3]),
        )
        return msg

    # ── 3D LiDAR ─────────────────────────────────────────────────────────────

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def livox_custom_msg(lidar_3d, stamp, lidar_3d_cfg):
        """Build Livox CustomMsg from RTX 3D lidar point cloud."""
        try:
            from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg
            from livox_ros_driver2.msg import CustomPoint as LivoxCustomPoint
        except ImportError:
            return None

        data = lidar_3d.get_point_cloud_data()
        if data is None:
            return None
        points = data["points"]
        intensities = data["intensities"]
        if len(points) == 0:
            return None

        timebase_ns = MessageBuilder._stamp_to_ns(stamp)
        scan_period_ns = int(1e9 / 10.0)
        point_count = len(points)
        offset_step_ns = scan_period_ns // max(point_count, 1)

        msg = LivoxCustomMsg()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = lidar_3d_cfg["frame_id"]
        msg.timebase = timebase_ns
        msg.point_num = point_count
        msg.lidar_id = int(lidar_3d_cfg.get("lidar_id", 0))
        msg.rsvd = [0, 0, 0]

        livox_points = []
        for idx, (xyz, intensity) in enumerate(zip(points, intensities)):
            point = LivoxCustomPoint()
            point.offset_time = int(idx * offset_step_ns)
            point.x = float(xyz[0])
            point.y = float(xyz[1])
            point.z = float(xyz[2])
            point.reflectivity = int(np.clip(intensity, 0, 255))
            point.tag = 0
            point.line = 0
            livox_points.append(point)
        msg.points = livox_points
        return msg

    @staticmethod
    def lidar_pointcloud2(lidar_3d, stamp, frame_id):
        """Build PointCloud2 message from RTX 3D LiDAR scan buffer."""
        from sensor_msgs.msg import PointCloud2, PointField

        data = lidar_3d.get_point_cloud_data()
        if data is None:
            return None
        points = data["points"]
        intensities = data["intensities"]
        if len(points) == 0:
            return None

        xyzi = np.column_stack([points, intensities]).astype(np.float32)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg = PointCloud2()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = frame_id
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.height = 1
        msg.width = len(xyzi)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = xyzi.tobytes()
        return msg

    # ── Camera: PointCloud2 ───────────────────────────────────────────────────

    @staticmethod
    def depth_pointcloud(obs, camera_name, stamp, camera_config, camera_cache):
        """Build PointCloud2 from a depth image in observations.

        Args:
            camera_config: Full ``CAMERA_CONFIG`` dict (keyed by camera name).
            camera_cache: Dict returned by ``init_camera_cache``.
        """
        from sensor_msgs.msg import PointCloud2, PointField

        config = camera_config.get(camera_name)
        cache = camera_cache.get(camera_name)
        if not config or not cache:
            return None
        depth_key = config.get("depth_key")
        if not depth_key:
            return None
        camera_obs = obs.get("camera_obs", obs)
        depth_data = camera_obs.get(depth_key)
        if depth_data is None:
            return None

        depth = to_numpy_2d(depth_data[0]).squeeze().astype(np.float32)

        fx, fy = cache["fx"], cache["fy"]
        cx, cy = cache["cx"], cache["cy"]
        near_clip, far_clip = cache["near_clip"], cache["far_clip"]
        downsample = config.get("pointcloud_downsample", 1)

        depth = depth[::downsample, ::downsample]
        fx, fy = fx / downsample, fy / downsample
        cx, cy = cx / downsample, cy / downsample

        valid_depth_mask = np.isfinite(depth) & (depth > FLOAT_EPS)
        depth = np.where(valid_depth_mask, depth, 0.0)

        h, w = depth.shape
        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > near_clip) & (points[:, 2] < far_clip)
        points = points[valid].astype(np.float32)

        field_names = ["x", "y", "z"]
        dtype_size = points.dtype.itemsize
        msg = PointCloud2()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = config["frame_id"]
        msg.fields = [
            PointField(
                name=name,
                offset=i * dtype_size,
                datatype=PointField.FLOAT32,
                count=1,
            )
            for i, name in enumerate(field_names)
        ]
        msg.is_bigendian = False
        msg.point_step = len(field_names) * dtype_size
        msg.height = 1
        msg.width = len(points)
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.tobytes()
        return msg

    # ── Camera: Raw RGB ─────────────────────────────────────────────────────

    @staticmethod
    def raw_rgb(obs, camera_name, stamp, camera_config):
        """Build raw ``sensor_msgs/Image`` (rgb8) from camera observations."""
        from sensor_msgs.msg import Image

        config = camera_config.get(camera_name)
        if not config or not config.get("rgb_key"):
            return None
        camera_obs = obs.get("camera_obs", obs)
        rgb_data = camera_obs.get(config["rgb_key"])
        if rgb_data is None:
            return None

        rgb = to_numpy_2d(rgb_data[0])
        rgb = rgb.clip(0, RGB_MAX_VALUE).astype(np.uint8)[..., :RGB_CHANNELS]

        msg = Image()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = config["frame_id"]
        msg.height = int(rgb.shape[0])
        msg.width = int(rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = int(rgb.shape[1] * RGB_CHANNELS)
        msg.data = rgb.tobytes()
        return msg

    # ── Camera: Raw Depth ────────────────────────────────────────────────────

    @staticmethod
    def raw_depth(obs, camera_name, stamp, camera_config, camera_cache):
        """Build raw ``sensor_msgs/Image`` (16UC1, millimetres) from depth observations."""
        from sensor_msgs.msg import Image

        config = camera_config.get(camera_name)
        cache = camera_cache.get(camera_name)
        if not config or not cache:
            return None
        depth_key = config.get("depth_key")
        if not depth_key:
            return None
        camera_obs = obs.get("camera_obs", obs)
        depth_data = camera_obs.get(depth_key)
        if depth_data is None:
            return None

        depth = to_numpy_2d(depth_data[0]).squeeze().astype(np.float32)
        max_depth_m = cache["far_clip"]
        depth_clipped = np.clip(depth, 0, max_depth_m)
        depth_mm = meters_to_mm(depth_clipped)
        depth_mm = np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.uint16)

        msg = Image()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = config["frame_id"]
        msg.height = int(depth_mm.shape[0])
        msg.width = int(depth_mm.shape[1])
        msg.encoding = "16UC1"
        msg.is_bigendian = False
        msg.step = int(depth_mm.shape[1] * depth_mm.dtype.itemsize)
        msg.data = depth_mm.tobytes()
        return msg

    # ── Camera: Compressed RGB ────────────────────────────────────────────────

    @staticmethod
    def compressed_rgb(obs, camera_name, stamp, camera_config):
        """Build CompressedImage (RGB) from camera observations."""
        import cv2
        from sensor_msgs.msg import CompressedImage

        config = camera_config.get(camera_name)
        if not config or not config.get("rgb_key"):
            return None
        camera_obs = obs.get("camera_obs", obs)
        rgb_data = camera_obs.get(config["rgb_key"])
        if rgb_data is None:
            return None

        rgb = to_numpy_2d(rgb_data[0])
        rgb = rgb.clip(0, RGB_MAX_VALUE).astype(np.uint8)[..., :RGB_CHANNELS]

        fmt = config.get("compress_fmt", "jpeg")
        quality = config.get("compress_quality")
        ext = f".{fmt}"
        params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality is not None else []

        _, compressed = cv2.imencode(ext, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params)

        msg = CompressedImage()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = config["frame_id"]
        msg.format = fmt
        msg.data = compressed.tobytes()
        return msg

    # ── Camera: Compressed Depth ──────────────────────────────────────────────

    @staticmethod
    def compressed_depth(obs, camera_name, stamp, camera_config, camera_cache):
        """Build CompressedImage (16UC1 depth PNG) from depth observations."""
        import cv2
        from sensor_msgs.msg import CompressedImage

        config = camera_config.get(camera_name)
        cache = camera_cache.get(camera_name)
        if not config or not cache:
            return None
        depth_key = config.get("depth_key")
        if not depth_key:
            return None
        camera_obs = obs.get("camera_obs", obs)
        depth_data = camera_obs.get(depth_key)
        if depth_data is None:
            return None

        depth = to_numpy_2d(depth_data[0]).squeeze().astype(np.float32)

        max_depth_m = cache["far_clip"]
        depth_clipped = np.clip(depth, 0, max_depth_m)
        depth_mm = meters_to_mm(depth_clipped)
        depth_mm = np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.uint16)

        fmt = config.get("compress_fmt", "png")
        _, compressed = cv2.imencode(f".{fmt}", depth_mm)

        msg = CompressedImage()
        copy_stamp(msg.header.stamp, stamp)
        msg.header.frame_id = config["frame_id"]
        msg.format = fmt
        msg.data = compressed.tobytes()
        return msg
