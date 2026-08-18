"""ROS2 publisher for the EX001 navigation robot (SDK topic surface)."""

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu as RosImu
from sensor_msgs.msg import JointState, LaserScan, PointCloud2
from std_msgs.msg import Float64MultiArray
from tf2_msgs.msg import TFMessage

from maniparena_sim.ros.ex001_sdk_topics import EX001_SDK_PUBLISH_TOPICS, EX001_SDK_SUBSCRIBE_TOPICS
from maniparena_sim.ros.ros2_config import ROS_QOS_CONFIG
from maniparena_sim.ros.ros_communicator import RosCommunicator


class EX001RosCommunicator(RosCommunicator):
    """ROS bridge implementation for EX001 SDK topic names."""

    FAST_TOPICS = frozenset(
        {
            "/hal/chassis/imu",
            "/odom",
            "/tracked_pose",
            "/joint_states",
            "/head/joint_states",
            "/left_arm/joint_states",
            "/right_arm/joint_states",
            "/left_gripper/joint_states",
            "/right_gripper/joint_states",
            "/lift/joint_states",
            "/left_arm/end_pose",
            "/right_arm/end_pose",
        }
    )

    LOW_RATE_TOPICS = frozenset(
        {
            "/scan",
            "/camera_chassis_front/depth/points",
            "/camera_head_front/color/image_raw/compressed",
            "/camera_head_front/depth/image_raw/compressedDepth",
            "/camera1/usb_cam1/image_raw/image_compressed",
            "/camera3/usb_cam3/image_raw/image_compressed",
            "/tf_static",
        }
    )

    PUBLISHERS = {
        "/joint_states": JointState,
        "/left_arm/joint_states": JointState,
        "/right_arm/joint_states": JointState,
        "/left_gripper/joint_states": JointState,
        "/right_gripper/joint_states": JointState,
        "/head/joint_states": JointState,
        "/lift/joint_states": JointState,
        "/left_arm/end_pose": PoseStamped,
        "/right_arm/end_pose": PoseStamped,
        "/odom": Odometry,
        "/tracked_pose": PoseStamped,
        "/hal/chassis/imu": RosImu,
        "/scan": LaserScan,
        "/camera_chassis_front/depth/points": PointCloud2,
        "/camera_head_front/color/image_raw/compressed": CompressedImage,
        "/camera_head_front/depth/image_raw/compressedDepth": CompressedImage,
        "/camera1/usb_cam1/image_raw/image_compressed": CompressedImage,
        "/camera3/usb_cam3/image_raw/image_compressed": CompressedImage,
        "/tf_static": TFMessage,
    }

    # /scan via RTX OmniGraph; /tf_static via StaticTransformBroadcaster.
    DEDICATED_PUBLISHERS = frozenset({"/scan", "/tf_static"})

    SUBSCRIBERS = {
        "/left_arm_joint_controller/commands": Float64MultiArray,
        "/right_arm_joint_controller/commands": Float64MultiArray,
        "/left_gripper_controller/commands": Float64MultiArray,
        "/right_gripper_controller/commands": Float64MultiArray,
        "/head_position_controller/commands": Float64MultiArray,
        "/lift_position_controller/commands": Float64MultiArray,
        "/chassis/cmd_vel": Twist,
    }

    _ex001_sampling_rate = {
        "default": 25,
        "/joint_states": 50,
        "/left_arm/joint_states": 50,
        "/right_arm/joint_states": 50,
        "/left_gripper/joint_states": 50,
        "/right_gripper/joint_states": 50,
        "/head/joint_states": 50,
        "/lift/joint_states": 50,
        "/left_arm/end_pose": 30,
        "/right_arm/end_pose": 30,
        "/camera_head_front/color/image_raw/compressed": 15,
        "/camera_head_front/depth/image_raw/compressedDepth": 5,
        "/camera1/usb_cam1/image_raw/image_compressed": 15,
        "/camera3/usb_cam3/image_raw/image_compressed": 15,
        "/camera_chassis_front/depth/points": 15,
        "/hal/chassis/imu": 120,
        "/scan": 10,
        "/odom": 30,
        "/tracked_pose": 30,
        "/tf_static": 1,
    }

    def _initRobotPublisher(self):
        if set(self.PUBLISHERS) != EX001_SDK_PUBLISH_TOPICS:
            raise ValueError("EX001RosCommunicator.PUBLISHERS must match EX001 SDK publish topics")
        for topic, msg_type in self.PUBLISHERS.items():
            if topic in self.DEDICATED_PUBLISHERS:
                continue
            self._ros_publishers[topic] = self.create_publisher(
                msg_type, topic, ROS_QOS_CONFIG["default_publisher_depth"]
            )
        if getattr(self, "_use_sim_time", False):
            from rosgraph_msgs.msg import Clock

            self._clock_publisher = self.create_publisher(Clock, "/clock", ROS_QOS_CONFIG["default_publisher_depth"])
            from rclpy.exceptions import ParameterAlreadyDeclaredException
            from rclpy.parameter import Parameter

            try:
                self.declare_parameter("use_sim_time", True)
            except ParameterAlreadyDeclaredException:
                pass
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        else:
            self._clock_publisher = None

    def _initRobotSubscriber(self):
        if set(self.SUBSCRIBERS) != EX001_SDK_SUBSCRIBE_TOPICS:
            raise ValueError("EX001RosCommunicator.SUBSCRIBERS must match EX001 SDK subscribe topics")
        for topic, msg_type in self.SUBSCRIBERS.items():
            cb = self._control_callbacks.get(topic)
            if cb is None or not callable(cb):
                raise ValueError(f"[ROS] Missing or invalid callback for topic: {topic}")
            qos = ROS_QOS_CONFIG["default_subscriber_depth"]
            if topic == "/chassis/cmd_vel":
                qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                )
            self._ros_subscribers[topic] = self.create_subscription(msg_type, topic, cb, qos)

    def __init__(
        self,
        control_callbacks,
        data_acquirer,
        sampling_rate=None,
        use_sim_time: bool = False,
        enabled_publishers=None,
    ) -> None:
        self._use_sim_time = use_sim_time
        if sampling_rate is None:
            sampling_rate = EX001RosCommunicator._ex001_sampling_rate
        super().__init__(
            control_callbacks=control_callbacks,
            data_acquirer=data_acquirer,
            sampling_rate=sampling_rate,
            node_name="ex001_ros_communicator",
        )
        if enabled_publishers is not None:
            enabled_publishers = set(enabled_publishers)
            for topic in list(self._ros_publishers.keys()):
                if topic in enabled_publishers:
                    continue
                pub = self._ros_publishers.pop(topic)
                self.destroy_publisher(pub)
                self._sim_timers.pop(topic, None)
                self._last_pub_time.pop(topic, None)

    def publish_clock(self, sim_time: float) -> None:
        """Publish simulation time to ``/clock``. No-op when *use_sim_time* is False."""
        if self._clock_publisher is not None:
            from rosgraph_msgs.msg import Clock

            from maniparena_sim.ros.sim_utils import get_ros_time

            msg = Clock()
            msg.clock = get_ros_time(sim_time)
            self._clock_publisher.publish(msg)
