from __future__ import annotations

import math

import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd


class RtxLidarHelper:
    def __init__(
        self,
        prim_path: str,
        env_id: int = 0,
        *,
        frame_id: str = "a_d_laser",
        topic_name: str = "scan",
        scan_rate_hz: float = 10.0,
        num_beams: int = 314,
        use_sim_time: bool = False,
    ):
        self._prim_path = prim_path
        self._env_id = env_id
        self._frame_id = frame_id
        self._topic_name = topic_name
        self._scan_rate_hz = scan_rate_hz
        self._num_beams = num_beams
        self._use_sim_time = use_sim_time
        self._lidar = None
        self._render_product = None
        self._writer = None
        self._ros_node = None
        self._ros_executor = None
        self._ros_thread = None
        self._scan_publisher = None
        self._partial_subscription = None
        self._bucket = None
        self._bucket_is_warmup = True
        self._ranges = None
        self._intensities = None
        self._last_valid_ranges = [math.inf] * num_beams
        self._last_valid_intensities = [0.0] * num_beams
        self._last_valid_ages = [3] * num_beams
        self._scan_template = None
        self._near_m = 0.05
        self._far_m = 25.0

    def initialize(self, scene) -> bool:
        prim_path = self._prim_path.replace("{ENV_REGEX_NS}", f"{scene.env_ns}/env_{self._env_id}")

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.set_extension_enabled_immediate("isaacsim.sensors.experimental.rtx", True):
            return False

        from isaacsim.sensors.experimental.rtx import Lidar

        stage = omni.usd.get_context().get_stage()
        lidar_prim = stage.GetPrimAtPath(prim_path)
        if not lidar_prim.IsValid():
            return False

        self._near_m = float(self._read_attr(lidar_prim, "omni:sensor:Core:nearRangeM", 0.05))
        self._far_m = float(self._read_attr(lidar_prim, "omni:sensor:Core:farRangeM", 25.0))
        self._set_attr(lidar_prim, "omni:sensor:Core:scanRateBaseHz", self._scan_rate_hz)
        self._set_attr(lidar_prim, "omni:sensor:Core:reportRateBaseHz", self._scan_rate_hz * self._num_beams)

        self._lidar = Lidar(
            prim_path,
            accumulate_outputs=True,
            aux_output_level="FULL",
            tick_rate=self._scan_rate_hz,
            reset_xform_op_properties=False,
        )

        self._render_product = rep.create.render_product(
            camera=prim_path,
            resolution=(1, 1),
            render_vars=["GenericModelOutput"],
        )
        self._writer = rep.writers.get("RtxLidarROS2PublishLaserScan")
        self._writer.initialize(
            frameId=self._frame_id,
            nodeNamespace="",
            queueSize=10,
            topicName="_maniparena/scan_raw",
            context=0,
            qosProfile="",
            horizontalFov=360.0,
            horizontalResolution=360.0 / self._num_beams,
            depthRange=[self._near_m, self._far_m],
            rotationRate=self._scan_rate_hz,
            azimuthRange=[-180.0, 180.0],
        )
        self._writer.attach([self._render_product.path])

        timeline = omni.timeline.get_timeline_interface()
        if float(timeline.get_end_time()) < 1_000_000.0:
            timeline.set_end_time(1_000_000.0)
        if not timeline.is_playing():
            timeline.play()

        return True

    def start_ros_assembler(self) -> None:
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan

        self._ros_node = rclpy.create_node("ex001_lidar_scan_assembler")
        self._scan_publisher = self._ros_node.create_publisher(LaserScan, self._topic_name, 10)
        self._partial_subscription = self._ros_node.create_subscription(
            LaserScan,
            "/maniparena/scan_raw",
            self._on_partial_scan,
            qos_profile_sensor_data,
        )
        self._ros_executor = SingleThreadedExecutor()
        self._ros_executor.add_node(self._ros_node)
        self._ros_thread = threading.Thread(target=self._ros_executor.spin, daemon=True)
        self._ros_thread.start()

    def _on_partial_scan(self, msg) -> None:
        if len(msg.ranges) != self._num_beams:
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        period_ns = int(round(1_000_000_000 / self._scan_rate_hz))
        bucket = stamp_ns // period_ns
        if self._bucket is None:
            self._reset_bucket(bucket)
        elif bucket > self._bucket:
            elapsed_buckets = bucket - self._bucket
            if not self._bucket_is_warmup:
                self._publish_completed_scan(period_ns)
                self._advance_cache_ages(elapsed_buckets - 1)
            self._bucket_is_warmup = False
            self._reset_bucket(bucket)
        elif bucket < self._bucket:
            return

        self._merge_partial_scan(msg)

    def _merge_partial_scan(self, msg) -> None:
        self._scan_template = msg
        for index, range_m in enumerate(msg.ranges):
            if math.isfinite(range_m) and msg.range_min <= range_m <= msg.range_max:
                self._ranges[index] = float(range_m)
                if index < len(msg.intensities):
                    self._intensities[index] = float(msg.intensities[index])

    def _reset_bucket(self, bucket: int) -> None:
        self._bucket = bucket
        self._ranges = [math.inf] * self._num_beams
        self._intensities = [0.0] * self._num_beams
        self._scan_template = None

    def _publish_completed_scan(self, period_ns: int) -> None:
        if self._scan_publisher is None or self._scan_template is None:
            return

        from maniparena_sim.ros.message_builder import MessageBuilder

        template = self._scan_template
        if self._use_sim_time:
            stamp_ns = self._bucket * period_ns
        else:
            stamp_ns = self._ros_node.get_clock().now().nanoseconds
        observed_ranges = self._ranges
        observed_intensities = self._intensities
        ranges = list(observed_ranges)
        intensities = list(observed_intensities)
        for index, range_m in enumerate(observed_ranges):
            if math.isfinite(range_m):
                self._last_valid_ranges[index] = range_m
                self._last_valid_intensities[index] = observed_intensities[index]
                self._last_valid_ages[index] = 0
                continue
            self._last_valid_ages[index] += 1
            if self._last_valid_ages[index] <= 2:
                ranges[index] = self._last_valid_ranges[index]
                intensities[index] = self._last_valid_intensities[index]
        msg = MessageBuilder.laserscan(
            template,
            stamp_ns=stamp_ns,
            frame_id=self._frame_id,
            scan_rate_hz=self._scan_rate_hz,
            range_min=self._near_m,
            range_max=self._far_m,
            ranges=ranges,
            intensities=intensities,
        )
        self._scan_publisher.publish(msg)

    def _advance_cache_ages(self, skipped_buckets: int) -> None:
        if skipped_buckets <= 0:
            return
        for index in range(self._num_beams):
            self._last_valid_ages[index] += skipped_buckets

    def shutdown(self) -> None:
        if self._ros_executor is not None:
            self._ros_executor.shutdown(timeout_sec=1.0)
        if self._ros_thread is not None:
            self._ros_thread.join(timeout=1.0)
        if self._ros_node is not None and self._partial_subscription is not None:
            self._ros_node.destroy_subscription(self._partial_subscription)
        if self._ros_node is not None and self._scan_publisher is not None:
            self._ros_node.destroy_publisher(self._scan_publisher)
        if self._ros_executor is not None and self._ros_node is not None:
            self._ros_executor.remove_node(self._ros_node)
        if self._ros_node is not None:
            self._ros_node.destroy_node()
        self._partial_subscription = None
        self._scan_publisher = None
        self._ros_executor = None
        self._ros_thread = None
        self._ros_node = None
        if self._writer is not None:
            self._writer.detach()
            self._writer = None
        if self._render_product is not None:
            self._render_product.destroy()
            self._render_product = None
        self._lidar = None

    @staticmethod
    def _read_attr(prim, name: str, default):
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid() and attr.Get() is not None:
            return attr.Get()
        return default

    @staticmethod
    def _set_attr(prim, name: str, value) -> None:
        attr = prim.GetAttribute(name)
        if attr and attr.IsValid():
            attr.Set(value)


def create_ex001_lidar(
    prim_path: str,
    env_id: int = 0,
    *,
    frame_id: str = "a_d_laser",
    topic_name: str = "scan",
    scan_rate_hz: float = 10.0,
    num_beams: int = 314,
    use_sim_time: bool = False,
) -> RtxLidarHelper:
    """Create EX001 lidar helper. Path must be passed from config."""
    return RtxLidarHelper(
        prim_path,
        env_id,
        frame_id=frame_id,
        topic_name=topic_name,
        scan_rate_hz=scan_rate_hz,
        num_beams=num_beams,
        use_sim_time=use_sim_time,
    )
