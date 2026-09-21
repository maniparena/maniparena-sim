"""ROS SDK bridge for Quanta X2 and ArtiXon Arm-6A.

Both profiles command and report absolute end-frame poses in ``base_link``.
Joint and gripper positions are radians. Sensors publish only when instantiated
by the scene; the bridge never invents camera or laser data.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

from maniparena_sim.ros.math_utils import compute_relative_pose, quat_rotate_inverse, to_numpy
from maniparena_sim.ros.sdk_control import SdkControl
from maniparena_sim.ros.sdk_profiles import DEPTH_TOPICS, RGB_TOPICS, get_sdk_profile


class SdkRosBridge:
    def __init__(self, cfg, profile):
        self._cfg = cfg
        self.profile = get_sdk_profile(profile)
        self._communicator = None
        self._control = None
        self._sim_time_acc = 0.0
        self._stamp_holder = {"stamp": None}
        self._tf_broadcaster = None
        self._tf_static = None
        self._tf_elapsed = 0.0
        self._static_tf_sent = False
        self._lidar = None

    @property
    def enabled(self):
        return self._cfg.enabled

    def setup(self, env, robot, action_buffer=None):
        """Initialize validated controls, state publishers and present sensors."""
        import torch

        if env.num_envs != 1:
            raise ValueError("SDK ROS bridge supports exactly one environment")
        self._env, self._robot = env, robot
        if action_buffer is None:
            action_buffer = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        self._control = SdkControl(
            env,
            robot,
            action_buffer,
            self.profile,
            self._cfg.arm_control,
            self._cfg.cmd_vel_timeout_s,
        )
        if not self.enabled:
            return

        import isaacsim

        ros_path = Path(isaacsim.__file__).resolve().parent / "exts" / "isaacsim.ros2.core" / "jazzy" / "rclpy"
        if str(ros_path) not in sys.path:
            sys.path.insert(0, str(ros_path))
        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        from maniparena_sim.ros.sdk_communicator import SdkCommunicator
        from maniparena_sim.ros.sim_utils import init_camera_cache
        from maniparena_sim.ros.tf_publisher import OdomOrigin

        scene_keys = set(env.scene.keys())
        self._imu = env.scene["imu"] if "imu" in scene_keys else None
        self._odom_origin = OdomOrigin()
        base = self._control.mapping.base
        self._odom_origin.init_from_pose(
            to_numpy(robot.data.body_pos_w[0, base]),
            to_numpy(robot.data.body_quat_w[0, base]),
        )
        self._camera_config = self._available_camera_config()
        init_camera_cache(env, self._camera_config)
        self._setup_lidar()
        data_acquirers = self._build_data_acquirers()
        self._communicator = SdkCommunicator(
            self.profile,
            self._cfg.arm_control,
            self._control.callbacks,
            data_acquirers,
            self._cfg.use_sim_time,
        )
        self._tf_broadcaster = TransformBroadcaster(self._communicator)
        self._tf_static = StaticTransformBroadcaster(self._communicator)

    def reset(self, robot=None):
        """Call after env.reset(); old joint, Cartesian and velocity targets are cleared."""
        if self._communicator is not None:
            self._communicator.reset_subscriptions()
        if self._control is not None:
            self._control.reset(self._robot if robot is None else robot)
        self._static_tf_sent = False
        self._tf_elapsed = 0.0

    def seed_ee_hold(self, robot):
        self.reset(robot)

    def project_ee_pose_commands(self, robot=None):
        if self._control is not None:
            self._control.project_ee_pose_commands(robot)

    def latest_cmd_vel(self, sim_time_s):
        return (0.0, 0.0, 0.0) if self._control is None else self._control.latest_cmd_vel(sim_time_s)

    def update(self, dt):
        if self._communicator is None:
            return
        from maniparena_sim.ros.sim_utils import build_robot_state_snapshot, get_ros_time

        self._communicator.step_control(dt)
        self._sim_time_acc += float(dt)
        self._stamp_holder["stamp"] = get_ros_time(self._sim_time_acc if self._cfg.use_sim_time else None)
        self._communicator.publish_clock(self._sim_time_acc)
        due = self._communicator.collect_due_topics(dt)
        obs = build_robot_state_snapshot(self._env, self._robot, self._imu)
        camera_topics = RGB_TOPICS | DEPTH_TOPICS
        self._communicator.publish_topics(set(due) - camera_topics, obs, {})
        if camera_topics.intersection(due):
            camera_obs = self._camera_snapshot()
            self._communicator.publish_topics(set(due) & camera_topics, camera_obs, {})
        self._tf_elapsed += float(dt)
        if self._tf_elapsed >= 1.0 / max(float(self._cfg.control_rate_hz), 1.0):
            self._publish_tf()
            self._tf_elapsed = 0.0

    def shutdown(self):
        if self._lidar is not None:
            self._lidar.shutdown()
            self._lidar = None
        if self._communicator is not None:
            self._communicator.shutdown()
            self._communicator = None
        self._tf_broadcaster = None
        self._tf_static = None

    def _setup_lidar(self):
        if not self.profile.mobile:
            return
        import omni.usd

        from maniparena_sim.ros.quanta_x1_rtx_lidar import RtxLidarHelper, resolve_lidar_prim_path
        from maniparena_sim.utils.debug_print import maniparenaprint

        template = "{ENV_REGEX_NS}/Robot/a_d_laser/Laser"
        path = resolve_lidar_prim_path(template, scene=self._env.scene)
        prim = omni.usd.get_context().get_stage().GetPrimAtPath(path)
        if not prim.IsValid():
            maniparenaprint(f"INFO: [{self.profile.name} SDK] No RTX lidar at {path}; /scan is disabled")
            return
        self._lidar = RtxLidarHelper(template, frame_id="a_d_laser", use_sim_time=self._cfg.use_sim_time)
        if not self._lidar.initialize(self._env.scene):
            raise RuntimeError(f"Failed to initialize RTX lidar at {path}")

    def _camera_snapshot(self):
        """Read scene camera tensors only when an image publisher is due."""
        camera_obs = {}
        for cfg in self._camera_config.values():
            output = self._env.scene[cfg["scene_entity"]].data.output
            if cfg.get("rgb_key") and "rgb" in output:
                camera_obs[cfg["rgb_key"]] = output["rgb"]
            if cfg.get("depth_key") and "distance_to_image_plane" in output:
                camera_obs[cfg["depth_key"]] = output["distance_to_image_plane"]
        return {"camera_obs": camera_obs}

    def _available_camera_config(self):
        from maniparena_sim.ros.ros2_config import QuantaX1RosConfig

        # Scene keys and observation keys are shared by the SDK embodiments.
        scene_keys = set(self._env.scene.keys())
        configs = {
            name: dict(cfg)
            for name, cfg in QuantaX1RosConfig.CAMERA_CONFIG.items()
            if cfg["scene_entity"] in scene_keys
        }
        if self.profile.name == "quanta_x2":
            for cfg in configs.values():
                if cfg["scene_entity"] in ("left_wrist_camera", "right_wrist_camera"):
                    side = cfg["scene_entity"].split("_", 1)[0]
                    cfg["frame_id"] = f"{side}_gripper_camera_color_frame"
        return configs

    def _build_data_acquirers(self):
        from maniparena_sim.ros.message_builder import MessageBuilder
        from maniparena_sim.ros.ros2_config import QuantaX1RosConfig
        from maniparena_sim.ros.sim_utils import camera_cache

        mapping = self._control.mapping

        def stamped(builder, *args, **kwargs):
            def acquire(obs, extras):
                return builder(obs, *args, stamp=self._stamp_holder["stamp"], **kwargs)

            return acquire

        def joint_state(indices, names):
            return stamped(MessageBuilder.joint_states, indices, list(names))

        acquirers = {"/joint_states": joint_state(list(range(len(mapping.joint_names))), mapping.joint_names)}
        for side in range(2):
            acquirers[self.profile.arm_states[side]] = joint_state(
                mapping.arms[side], self.profile.arm_joint_names[side]
            )
            acquirers[self.profile.gripper_states[side]] = joint_state(
                [mapping.grippers[side]], [self.profile.gripper_state_names[side]]
            )
            acquirers[self.profile.arm_poses[side]] = partial(self._acquire_flange_pose, side=side)
        if self.profile.mobile:
            if self._lidar is not None:
                acquirers["/scan"] = lambda obs, extras: self._lidar.build_laserscan(self._stamp_holder["stamp"])
            acquirers["/head/joint_states"] = joint_state(mapping.head, [mapping.joint_names[i] for i in mapping.head])
            acquirers["/waist/state/joint"] = joint_state(
                mapping.waist, [mapping.joint_names[i] for i in mapping.waist]
            )
            acquirers["/tracked_pose"] = self._acquire_tracked_pose
            acquirers["/odom"] = self._acquire_odom
            if self._imu is not None:
                acquirers["/hal/chassis/imu"] = stamped(
                    MessageBuilder.imu,
                    imu_cfg=QuantaX1RosConfig.IMU_CONFIG["chassis_imu"],
                )
        camera_specs = (
            ("camera1", "/camera1/usb_cam1/image_raw/image_compressed", "rgb"),
            ("camera3", "/camera3/usb_cam3/image_raw/image_compressed", "rgb"),
            ("head_camera", "/camera_head_front/color/image_raw/compressed", "rgb"),
            (
                "head_camera",
                "/camera_head_front/depth/image_raw/compressedDepth",
                "depth",
            ),
            ("chassis_front_camera", "/camera_chassis_front/depth/points", "points"),
        )
        for camera, topic, kind in camera_specs:
            if camera not in self._camera_config or topic not in self.profile.publish_topics:
                continue
            builder = {
                "rgb": MessageBuilder.compressed_rgb,
                "depth": MessageBuilder.compressed_depth,
                "points": MessageBuilder.depth_pointcloud,
            }[kind]
            kwargs = {"camera_name": camera, "camera_config": self._camera_config}
            if kind != "rgb":
                kwargs["camera_cache"] = camera_cache
            acquirers[topic] = stamped(builder, **kwargs)
        return acquirers

    def _acquire_tracked_pose(self, obs, extras):
        from geometry_msgs.msg import PoseStamped

        from maniparena_sim.ros.stamp_utils import copy_stamp

        base = self._control.mapping.base
        pos = to_numpy(self._robot.data.body_pos_w[0, base])
        quat = to_numpy(self._robot.data.body_quat_w[0, base])
        msg = PoseStamped()
        copy_stamp(msg.header.stamp, self._stamp_holder["stamp"])
        msg.header.frame_id = "world"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(x) for x in pos]
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = [float(x) for x in quat]
        return msg

    def _acquire_odom(self, obs, extras):
        from nav_msgs.msg import Odometry

        from maniparena_sim.ros.stamp_utils import copy_stamp

        base = self._control.mapping.base
        data = self._robot.data
        base_pos = to_numpy(data.body_pos_w[0, base])
        base_quat = to_numpy(data.body_quat_w[0, base])
        origin = self._odom_origin
        pos, quat = compute_relative_pose(base_pos, base_quat, origin.position, origin.orientation)
        # Isaac Lab's link velocity is evaluated at the actor frame origin;
        # its COM velocity aliases are unsuitable for this ROS child frame.
        velocity = to_numpy(data.body_link_vel_w[0, base])
        linear = quat_rotate_inverse(base_quat, velocity[:3])
        angular = quat_rotate_inverse(base_quat, velocity[3:])
        msg = Odometry()
        copy_stamp(msg.header.stamp, self._stamp_holder["stamp"])
        msg.header.frame_id, msg.child_frame_id = "odom", "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = [float(x) for x in pos]
        (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ) = [float(x) for x in quat]
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = [float(x) for x in linear]
        (
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ) = [float(x) for x in angular]
        return msg

    def _acquire_flange_pose(self, obs, extras, side):
        from geometry_msgs.msg import PoseStamped

        from maniparena_sim.ros.message_builder import MessageBuilder
        from maniparena_sim.ros.stamp_utils import copy_stamp

        mapping = self._control.mapping
        pose = MessageBuilder.ee_pose_in_arm_base(obs, mapping.flanges[side], mapping.base)
        if pose is None:
            return None
        msg = PoseStamped()
        copy_stamp(msg.header.stamp, self._stamp_holder["stamp"])
        msg.header.frame_id = "base_link"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(x) for x in pose[0]]
        (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ) = [float(x) for x in pose[1]]
        return msg

    def _publish_tf(self):
        from geometry_msgs.msg import TransformStamped

        from maniparena_sim.ros.stamp_utils import copy_stamp

        def transform(parent, child, pos, quat):
            msg = TransformStamped()
            copy_stamp(msg.header.stamp, self._stamp_holder["stamp"])
            msg.header.frame_id, msg.child_frame_id = parent, child
            (
                msg.transform.translation.x,
                msg.transform.translation.y,
                msg.transform.translation.z,
            ) = [float(x) for x in pos]
            (
                msg.transform.rotation.x,
                msg.transform.rotation.y,
                msg.transform.rotation.z,
                msg.transform.rotation.w,
            ) = [float(x) for x in quat]
            return msg

        robot, mapping = self._robot, self._control.mapping
        base_pos = to_numpy(robot.data.body_pos_w[0, mapping.base])
        base_quat = to_numpy(robot.data.body_quat_w[0, mapping.base])
        transforms = []
        if self.profile.mobile:
            origin = self._odom_origin
            if not self._static_tf_sent:
                self._tf_static.sendTransform(transform("world", "odom", origin.position, origin.orientation))
                self._static_tf_sent = True
            pos, quat = compute_relative_pose(base_pos, base_quat, origin.position, origin.orientation)
            transforms.append(transform("odom", "base_link", pos, quat))
        elif not self._static_tf_sent:
            self._tf_static.sendTransform(transform("world", "base_link", base_pos, base_quat))
            self._static_tf_sent = True
        for i, name in enumerate(mapping.body_names):
            if i == mapping.base:
                continue
            pos, quat = compute_relative_pose(
                to_numpy(robot.data.body_pos_w[0, i]),
                to_numpy(robot.data.body_quat_w[0, i]),
                base_pos,
                base_quat,
            )
            transforms.append(transform("base_link", name, pos, quat))
        # Cameras may be authored directly under the robot, without a rigid
        # body of their own. Their ROS optical pose still needs a TF parent.
        for cfg in self._camera_config.values():
            frame = cfg["frame_id"]
            if frame in mapping.body_names:
                continue
            camera = self._env.scene[cfg["scene_entity"]].data
            pos, quat = compute_relative_pose(
                to_numpy(camera.pos_w[0]),
                to_numpy(camera.quat_w_ros[0]),
                base_pos,
                base_quat,
            )
            transforms.append(transform("base_link", frame, pos, quat))
        if transforms:
            self._tf_broadcaster.sendTransform(transforms)
