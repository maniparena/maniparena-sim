"""SDK ROS2 bridge extension – YAML-driven ROS topic pub/sub.

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

import yaml

from maniparena_sim.ros.arm_mode import DEFAULT_MODE_SERVICE, ArmControlMode, install_mode_service
from maniparena_sim.utils.debug_print import maniparenaprint

CAMERA_TOPICS: set[str] = {
    "/camera_chassis_front/depth/points",
    "/camera1/usb_cam1/image_raw/image_compressed",
    "/camera3/usb_cam3/image_raw/image_compressed",
    "/camera_head_front/color/image_raw/compressed",
    "/camera_head_front/depth/image_raw/compressedDepth",
}


@dataclass
class RosBridgeCfg:
    enabled: bool = True
    nav_mode: str = "2d"
    chassis_input: str = "ros"
    use_sim_time: bool = False
    control_rate_hz: float = 10.0
    cmd_vel_timeout_s: float = 0.25
    arm_control: str = "ee"
    mode_service: str = DEFAULT_MODE_SERVICE


def load_ros_bridge_cfg(config_path: str | Path) -> RosBridgeCfg:
    from maniparena_sim.ros.quanta_x1_sdk_topics import normalize_arm_control

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ros = raw.get("ros") or {}
    return RosBridgeCfg(
        enabled=bool(ros.get("enabled", True)),
        use_sim_time=bool(ros.get("use_sim_time", False)),
        control_rate_hz=float(ros.get("control_rate_hz", 10.0)),
        cmd_vel_timeout_s=float(ros.get("cmd_vel_timeout_s", 0.25)),
        arm_control=normalize_arm_control(ros.get("arm_control", "ee")),
        mode_service=str(ros.get("mode_service", DEFAULT_MODE_SERVICE)),
    )


class RosBridgeExtension:
    """Self-contained QUANTA_X1 SDK ROS2 bridge."""

    def __init__(self, cfg: RosBridgeCfg):
        self._cfg = cfg
        self._communicator: Any = None
        self._tf_pub: Any = None
        self._tf_static: Any = None
        self._static_tf_sent: bool = False
        self._stamp_holder: dict = {"stamp": None}
        self._body_names: list[str] = []
        self._sim_time_acc: float = 0.0
        self._get_ros_time: Any = None
        self._robot: Any = None
        self._imu_sensor: Any = None
        self._cmd_vel_buffer: Any = None
        self._cmd_vel_last_seq: int = 0
        self._cmd_vel_last_sim_time: float = float("-inf")
        self._cmd_vel_held: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._cmd_vel_zero_sent: bool = False
        self._lidar_2d: Any = None
        self._ee_pose_buffer: Any = None
        self._ee_home: Any = None
        self._mode = ArmControlMode(cfg.arm_control)
        self._mode_service: Any = None
        self._ik: Any = None
        self._ik_results: list = [None, None]
        self._arm_slots: list[list[int]] = []
        self._arm_joint_ids: list[list[int]] = []
        self._flange_body_ids: list[int] = []
        self._slot_map: dict[str, int] = {}
        self._lift_body_idx: int | None = None
        self._action_buffer: Any = None
        self._enabled_publishers: set[str] = set()
        self._fast_topics: set[str] = set()
        self._slow_topics: set[str] = set()
        self._camera_topics: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def arm_control(self) -> str:
        """Active arm control mode (``ee`` or ``joint``)."""
        return self._mode.mode

    @property
    def chassis_input(self) -> str:
        return self._cfg.chassis_input

    @property
    def cmd_vel_timeout_s(self) -> float:
        return float(self._cfg.cmd_vel_timeout_s)

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

        from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

        from maniparena_sim.kinematics.arm_ik import ArmIkController, ArmSpec
        from maniparena_sim.ros.quanta_x1_control_callbacks import (
            CmdVelCommandBuffer,
            EePoseCommandBuffer,
            HomeEeRef,
        )
        from maniparena_sim.ros.quanta_x1_data_acquirers import fill_data_acquirer
        from maniparena_sim.ros.quanta_x1_joint_mapping import (
            QuantaX1JointIndexMapping,
            build_action_slot_map,
        )
        from maniparena_sim.ros.quanta_x1_ros_communicator import QuantaX1RosCommunicator
        from maniparena_sim.ros.quanta_x1_sdk_topics import QUANTA_X1_SDK_PUBLISH_TOPICS
        from maniparena_sim.ros.prim_paths import QUANTA_X1_PATHS
        from maniparena_sim.ros.ros2_config import QuantaX1RosConfig
        from maniparena_sim.ros.sim_utils import (
            build_robot_state_snapshot,
            get_root_pose,
            get_ros_time,
            init_camera_cache,
        )
        from maniparena_sim.ros.tf_publisher import OdomOrigin, TfPublisher

        if set(QuantaX1RosCommunicator.PUBLISHERS) != QUANTA_X1_SDK_PUBLISH_TOPICS:
            raise RuntimeError("QuantaX1RosCommunicator.PUBLISHERS drifted from QUANTA_X1_SDK_PUBLISH_TOPICS")

        self._get_ros_time = get_ros_time
        self._robot = robot
        self._imu_sensor = env.scene["imu"] if "imu" in env.scene.keys() else None
        self._cmd_vel_buffer = None
        self._sim_time_acc = 0.0
        self._static_tf_sent = False
        nav_mode = self._cfg.nav_mode

        # -- Sensors (2D RTX lidar only) -----------------------------------
        if nav_mode != "2d":
            raise NotImplementedError(f"Only ros.nav_mode='2d' is supported, got {nav_mode!r}.")
        from maniparena_sim.ros.quanta_x1_rtx_lidar import create_quanta_x1_lidar

        lidar_cfg = QuantaX1RosConfig.LIDAR_CONFIG["chassis_lidar"]
        self._lidar_2d = create_quanta_x1_lidar(
            prim_path=QUANTA_X1_PATHS.lidar,
            frame_id=lidar_cfg["frame_id"],
            topic_name="scan",
            scan_rate_hz=10.0,
            use_sim_time=self._cfg.use_sim_time,
        )
        if not self._lidar_2d.initialize(env.scene):
            raise RuntimeError("Failed to initialize QUANTA_X1 ROS2 RTX lidar publisher")

        self._cmd_vel_buffer = CmdVelCommandBuffer()
        self._cmd_vel_last_seq = 0
        self._cmd_vel_last_sim_time = float("-inf")
        self._cmd_vel_held = (0.0, 0.0, 0.0)
        self._cmd_vel_zero_sent = False

        # -- Joint mapping / odom / camera cache ---------------------------
        joint_mapping = QuantaX1JointIndexMapping(robot)
        slot_map = build_action_slot_map(env.action_manager)
        odom_origin = OdomOrigin()
        init_camera_cache(env, QuantaX1RosConfig.CAMERA_CONFIG)

        self._env = env
        obs0 = build_robot_state_snapshot(env, robot, self._imu_sensor)
        root_pos, root_quat = get_root_pose(obs0)
        odom_origin.init_from_pose(root_pos, root_quat)

        # -- Acquirers / callbacks -----------------------------------------
        data_acquirer = {t: None for t in QuantaX1RosCommunicator.PUBLISHERS}
        arm_control = self._mode.mode

        self._ee_home = HomeEeRef()
        fill_data_acquirer(
            data_acquirer,
            joint_mapping,
            self._stamp_holder,
            odom_origin,
            self._lidar_2d,
            home_ref=self._ee_home,
        )
        shared_action_buffer = action_buffer
        if shared_action_buffer is None:
            shared_action_buffer = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        self._action_buffer = shared_action_buffer
        self._slot_map = slot_map
        self._ee_pose_buffer = EePoseCommandBuffer()
        self._lift_body_idx = int(joint_mapping.arm_base_body[0])
        self._flange_body_ids = [int(joint_mapping.left_gripper_body[0]), int(joint_mapping.right_gripper_body[0])]
        arm_names = [
            [f"{side}_arm_joint{i}" for i in range(1, 7)] for side in ("left", "right")
        ]
        self._arm_slots = [[slot_map[n] for n in names] for names in arm_names]
        self._arm_joint_ids = [[joint_mapping.joint_name_to_index[n] for n in names] for names in arm_names]
        self._ik = ArmIkController.from_usd(
            str(robot.cfg.spawn.usd_path),
            [
                ArmSpec(tuple(names), body)
                for names, body in zip(arm_names, ("left_arm_gripper_base_link", "right_arm_gripper_base_link"))
            ],
        )
        self._ee_home.capture_from_robot(robot, self._lift_body_idx)
        control_callbacks = self._control_callbacks_for(arm_control)

        # -- Communicator & TF ---------------------------------------------
        enabled_publishers = self._resolve_enabled_publishers(
            has_camera_obs="chassis_camera" in env.scene.keys(),
            has_imu_sensor=self._imu_sensor is not None,
        )
        self._communicator = QuantaX1RosCommunicator(
            control_callbacks=control_callbacks,
            data_acquirer=data_acquirer,
            use_sim_time=self._cfg.use_sim_time,
            enabled_publishers=enabled_publishers,
            arm_control=arm_control,
        )
        self._enabled_publishers = set(enabled_publishers)
        self._fast_topics = QuantaX1RosCommunicator.FAST_TOPICS & self._enabled_publishers
        self._slow_topics = QuantaX1RosCommunicator.LOW_RATE_TOPICS & self._enabled_publishers
        self._camera_topics = CAMERA_TOPICS & self._enabled_publishers
        tf_broadcaster = TransformBroadcaster(self._communicator)
        self._tf_static = StaticTransformBroadcaster(self._communicator)
        self._tf_pub = TfPublisher(tf_broadcaster, odom_origin, publish_interval=1)
        self._tf_pub.init_from_robot(robot)
        self._body_names = list(robot.data.body_names)
        self._mode.bind(self._switch_mode)
        self._mode_service = install_mode_service(self._communicator, self._mode, self._cfg.mode_service)

        maniparenaprint(
            f"INFO: [ROS] Initialized: nav_mode={nav_mode}  "
            f"chassis_input={self._cfg.chassis_input}  "
            f"use_sim_time={self._cfg.use_sim_time}  "
            f"control_rate_hz={self._cfg.control_rate_hz:.2f}  "
            f"arm_control={arm_control}"
        )

    def _control_callbacks_for(self, mode: str) -> dict:
        from maniparena_sim.ros.quanta_x1_control_callbacks import fill_control_callbacks

        callbacks: dict = {}
        fill_control_callbacks(
            callbacks,
            self._slot_map,
            self._action_buffer,
            cmd_vel_buffer=self._cmd_vel_buffer,
            arm_control=mode,
            ee_pose_buffer=self._ee_pose_buffer,
        )
        return callbacks

    def _switch_mode(self, mode: str) -> None:
        """Hold the current joint targets and swap the arm command topics."""
        callbacks = self._control_callbacks_for(mode)
        self._ee_pose_buffer.clear()
        self._ik.reset()
        self._ik_results = [None, None]
        self._communicator.set_arm_control(mode, callbacks)

    def seed_ee_hold(self, robot) -> None:
        """After an env reset: recapture home, drop pending commands, hold the reset pose."""
        if self._ee_home is None or self._lift_body_idx is None:
            return
        self._ee_home.capture_from_robot(robot, self._lift_body_idx)
        self._ee_pose_buffer.clear()
        self._ik.reset()
        self._ik_results = [None, None]
        indices = {name: i for i, name in enumerate(robot.data.joint_names)}
        for name, slot in self._slot_map.items():
            if name not in ("left_wheel_joint", "right_wheel_joint"):
                self._action_buffer[0, slot] = float(robot.data.joint_pos[0, indices[name]])

    def project_ee_pose_commands(self, robot) -> None:
        """Solve current EE commands to arm joint targets (no-op in joint mode)."""
        from maniparena_sim.ros.quanta_x1_control_callbacks import project_ee_pose_commands

        if self._mode.mode != "ee" or self._ee_pose_buffer is None:
            return
        self._ik_results = project_ee_pose_commands(
            self._ee_pose_buffer,
            self._action_buffer,
            self._arm_slots,
            robot,
            self._arm_joint_ids,
            self._flange_body_ids,
            self._lift_body_idx,
            self._ee_home,
            self._ik,
        )

    def latest_cmd_vel(self, sim_time_s: float) -> tuple[float, float, float]:
        """Return held ``/chassis/cmd_vel`` with timeout zeroing."""
        buf = self._cmd_vel_buffer
        if buf is None:
            return (0.0, 0.0, 0.0)
        snapshot = buf.read_if_new(self._cmd_vel_last_seq)
        if snapshot is not None:
            self._cmd_vel_last_seq, self._cmd_vel_held = snapshot
            self._cmd_vel_last_sim_time = float(sim_time_s)
            self._cmd_vel_zero_sent = False
        timeout = self.cmd_vel_timeout_s
        if (
            timeout > 0.0
            and self._cmd_vel_last_sim_time != float("-inf")
            and (float(sim_time_s) - self._cmd_vel_last_sim_time) > timeout
        ):
            if self._cmd_vel_zero_sent:
                return (0.0, 0.0, 0.0)
            self._cmd_vel_held = (0.0, 0.0, 0.0)
            self._cmd_vel_zero_sent = True
            return self._cmd_vel_held
        if self._cmd_vel_last_sim_time == float("-inf"):
            return (0.0, 0.0, 0.0)
        return self._cmd_vel_held

    def update(self, dt: float) -> None:
        """Call once per simulation step (after ``env.step``)."""
        from maniparena_sim.ros.sim_utils import build_robot_state_snapshot

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
        if not self._static_tf_sent and self._tf_static is not None:
            self._tf_pub.publish_static_on_base(fast_obs, self._body_names, stamp, self._tf_static)
            self._static_tf_sent = True
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
        self._tf_static = None

    def _compute_obs(self) -> dict:
        """Get observations via the environment's observation manager."""
        return self._env.observation_manager.compute()

    def _compute_slow_obs(self, due_topics: list[str], fallback_obs: dict) -> dict:
        """Compute observations only when slow topics actually require them."""
        if any(topic in self._camera_topics for topic in due_topics):
            return self._compute_obs()
        return fallback_obs

    def _resolve_enabled_publishers(self, *, has_camera_obs: bool, has_imu_sensor: bool) -> set[str]:
        from maniparena_sim.ros.quanta_x1_ros_communicator import QuantaX1RosCommunicator

        publishers = set(QuantaX1RosCommunicator.PUBLISHERS.keys())
        publishers.difference_update(QuantaX1RosCommunicator.DEDICATED_PUBLISHERS)
        if not has_camera_obs:
            publishers.difference_update(CAMERA_TOPICS)
        if not has_imu_sensor:
            publishers.discard("/hal/chassis/imu")
        return publishers
