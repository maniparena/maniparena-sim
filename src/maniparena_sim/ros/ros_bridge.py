"""ROS2 bridge extension – YAML-driven ROS topic pub/sub for navigation.

Lifecycle matches the other extensions::

    ext = RosBridgeExtension(cfg)
    ext.setup(env, robot)
    while running:
        ext.update(dt)
    ext.shutdown()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from maniparena_sim.ros.sim_utils import build_robot_state_snapshot
from maniparena_sim.utils.debug_print import manaprint

CAMERA_TOPICS: set[str] = {
    "/camera_chassis_front/depth/points",
    "/camera1/usb_cam1/image_raw/image_compressed",
    "/camera3/usb_cam3/image_raw/image_compressed",
    "/camera_head_front/color/image_raw/compressed",
}


@dataclass
class RosBridgeCfg:
    enabled: bool = True
    nav_mode: str = "2d"
    chassis_input: str = "ros"
    use_sim_time: bool = False
    control_rate_hz: float = 10.0
    cmd_vel_timeout_s: float = 0.25


def load_ros_bridge_cfg(config_path: str | Path) -> RosBridgeCfg:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ros = raw.get("ros") or {}
    return RosBridgeCfg(
        enabled=bool(ros.get("enabled", True)),
        use_sim_time=bool(ros.get("use_sim_time", False)),
        control_rate_hz=float(ros.get("control_rate_hz", 10.0)),
        cmd_vel_timeout_s=float(ros.get("cmd_vel_timeout_s", 0.25)),
    )


class RosBridgeExtension:
    """Self-contained ROS2 bridge for the EX001 navigation environment."""

    def __init__(self, cfg: RosBridgeCfg):
        self._cfg = cfg
        self._communicator: Any = None
        self._tf_pub: Any = None
        self._stamp_holder: dict = {"stamp": None}
        self._body_names: list[str] = []
        self._sim_time_acc: float = 0.0
        self._get_ros_time: Any = None
        self._robot: Any = None
        self._imu_sensor: Any = None
        self._cmd_vel_buffer: Any = None
        self._lidar_2d: Any = None
        self._enabled_publishers: set[str] = set()
        self._fast_topics: set[str] = set()
        self._slow_topics: set[str] = set()
        self._camera_topics: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def chassis_input(self) -> str:
        return self._cfg.chassis_input

    # -- lifecycle ---------------------------------------------------------

    def setup(self, env: Any, robot: Any, action_buffer: Any | None = None) -> None:
        """Initialize ROS pub/sub, sensors, and control callbacks for one environment.

        When ``action_buffer`` is provided, ROS callbacks write directly into that
        shared buffer instead of allocating a private one inside the bridge.
        """
        if not self._cfg.enabled:
            return

        import isaacsim
        import torch

        ros2_python = Path(isaacsim.__file__).resolve().parent / "exts" / "isaacsim.ros2.core" / "jazzy" / "rclpy"
        if str(ros2_python) not in sys.path:
            sys.path.insert(0, str(ros2_python))

        from tf2_ros import TransformBroadcaster

        from maniparena_sim.ros.ex001_control_callbacks import fill_control_callbacks
        from maniparena_sim.ros.ex001_data_acquirers import fill_data_acquirer
        from maniparena_sim.ros.ex001_joint_mapping import EX001JointIndexMapping
        from maniparena_sim.ros.ex001_ros_communicator import EX001RosCommunicator
        from maniparena_sim.ros.prim_paths import EX001_PATHS
        from maniparena_sim.ros.ros2_config import EX001RosConfig
        from maniparena_sim.ros.sim_utils import get_root_pose, get_ros_time, init_camera_cache
        from maniparena_sim.ros.tf_publisher import OdomOrigin, TfPublisher

        self._get_ros_time = get_ros_time
        self._robot = robot
        self._imu_sensor = env.scene["imu"] if "imu" in env.scene.keys() else None
        self._cmd_vel_buffer = None
        self._sim_time_acc = 0.0
        nav_mode = self._cfg.nav_mode

        # -- Sensors (2D RTX lidar only) -----------------------------------
        if nav_mode != "2d":
            raise NotImplementedError(f"Only ros.nav_mode='2d' is supported, got {nav_mode!r}.")
        from maniparena_sim.ros.ex001_rtx_lidar import create_ex001_lidar

        lidar_cfg = EX001RosConfig.LIDAR_CONFIG["chassis_lidar"]
        self._lidar_2d = create_ex001_lidar(
            prim_path=EX001_PATHS.lidar,
            frame_id=lidar_cfg["frame_id"],
            topic_name="scan",
            scan_rate_hz=10.0,
            num_beams=314,
            use_sim_time=self._cfg.use_sim_time,
        )
        if not self._lidar_2d.initialize(env.scene):
            raise RuntimeError("Failed to initialize EX001 ROS2 RTX lidar publisher")

        # -- ROS chassis controller (differential drive) ------------------
        ros_chassis_ctrl = None
        # -- Chassis cmd_vel buffer ----------------------------------------
        # The chassis is driven through the env action vector (wheel-velocity
        # term) by the nav loop, which reads this buffer and sums it with the
        # keyboard twist. No direct-to-sim controller / gate here.
        from maniparena_sim.ros.ex001_control_callbacks import CmdVelCommandBuffer

        self._cmd_vel_buffer = CmdVelCommandBuffer()

        # -- Joint mapping / odom / camera cache ---------------------------
        joint_mapping = EX001JointIndexMapping(robot)
        from maniparena_sim.ros.ex001_joint_mapping import build_action_slot_map

        slot_map = build_action_slot_map(env.action_manager)
        odom_origin = OdomOrigin()
        init_camera_cache(env, EX001RosConfig.CAMERA_CONFIG)

        self._env = env
        obs0 = build_robot_state_snapshot(env, robot, self._imu_sensor)
        root_pos, root_quat = get_root_pose(obs0)
        odom_origin.init_from_pose(root_pos, root_quat)

        # -- Acquirers / callbacks -----------------------------------------
        data_acquirer = {t: None for t in EX001RosCommunicator.PUBLISHERS}
        control_callbacks = {t: None for t in EX001RosCommunicator.SUBSCRIBERS}

        fill_data_acquirer(
            data_acquirer,
            joint_mapping,
            self._stamp_holder,
            odom_origin,
            env,
        )
        shared_action_buffer = action_buffer
        if shared_action_buffer is None:
            shared_action_buffer = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

        fill_control_callbacks(
            control_callbacks,
            slot_map,
            shared_action_buffer,
            cmd_vel_buffer=self._cmd_vel_buffer,
        )

        # -- Communicator & TF ---------------------------------------------
        enabled_publishers = self._resolve_enabled_publishers(
            has_camera_obs="chassis_camera" in env.scene.keys(),
        )
        self._communicator = EX001RosCommunicator(
            control_callbacks=control_callbacks,
            data_acquirer=data_acquirer,
            use_sim_time=self._cfg.use_sim_time,
            enabled_publishers=enabled_publishers,
        )
        self._lidar_2d.start_ros_assembler()
        self._enabled_publishers = set(enabled_publishers)
        self._fast_topics = EX001RosCommunicator.FAST_TOPICS & self._enabled_publishers
        self._slow_topics = EX001RosCommunicator.LOW_RATE_TOPICS & self._enabled_publishers
        self._camera_topics = {
            "/camera_chassis_front/depth/points",
            "/camera1/usb_cam1/image_raw/image_compressed",
            "/camera3/usb_cam3/image_raw/image_compressed",
            "/camera_head_front/color/image_raw/compressed",
        } & self._enabled_publishers
        tf_broadcaster = TransformBroadcaster(self._communicator)
        self._tf_pub = TfPublisher(tf_broadcaster, odom_origin, publish_interval=1)
        self._tf_pub.init_from_robot(robot)
        self._body_names = list(robot.data.body_names)

        manaprint(
            f"INFO: [ROS] Initialized: nav_mode={nav_mode}  "
            f"chassis_input={self._cfg.chassis_input}  "
            f"use_sim_time={self._cfg.use_sim_time}  "
            f"control_rate_hz={self._cfg.control_rate_hz:.2f}"
        )

    def update(self, dt: float) -> None:
        """Call once per simulation step (after ``env.step``)."""
        if self._communicator is None:
            return

        self._communicator.step_control(dt)
        self._sim_time_acc += dt

        if self._cfg.use_sim_time:
            stamp = self._get_ros_time(self._sim_time_acc)
            self._communicator.publish_clock(self._sim_time_acc)
        else:
            stamp = self._get_ros_time()

        self._stamp_holder["stamp"] = stamp
        fast_obs = build_robot_state_snapshot(self._env, self._robot, self._imu_sensor)
        due_fast = self._communicator.collect_due_topics(dt, self._fast_topics)
        if due_fast:
            self._communicator.publish_topics(due_fast, fast_obs, {})
        self._tf_pub.publish_all(fast_obs, self._body_names, stamp)

        due_slow = self._communicator.collect_due_topics(dt, self._slow_topics)
        if due_slow:
            slow_obs = self._compute_slow_obs(due_slow, fast_obs)
            self._communicator.publish_topics(due_slow, slow_obs, {})

    def shutdown(self) -> None:
        if self._lidar_2d is not None:
            self._lidar_2d.shutdown()
            self._lidar_2d = None
        if self._communicator is not None:
            self._communicator.shutdown()
            self._communicator = None
        self._cmd_vel_buffer = None

    def _compute_obs(self) -> dict:
        """Get observations via the environment's observation manager."""
        return self._env.observation_manager.compute()

    def _compute_slow_obs(self, due_topics: list[str], fallback_obs: dict) -> dict:
        """Compute observations only when slow topics actually require them."""
        if any(topic in self._camera_topics for topic in due_topics):
            return self._compute_obs()
        return fallback_obs

    def _resolve_enabled_publishers(self, *, has_camera_obs: bool) -> set[str]:
        from maniparena_sim.ros.ex001_ros_communicator import EX001RosCommunicator

        publishers = set(EX001RosCommunicator.PUBLISHERS.keys())
        publishers.difference_update(EX001RosCommunicator.DEDICATED_PUBLISHERS)
        if not has_camera_obs:
            publishers.difference_update(CAMERA_TOPICS)
        return publishers
