"""ROS runtime transport for the public SDK profiles (loaded after AppLauncher)."""

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Imu, JointState, LaserScan, PointCloud2
from std_msgs.msg import Float64MultiArray

from maniparena_sim.ros.ros_communicator import RosCommunicator
from maniparena_sim.ros.sdk_profiles import DEPTH_TOPICS, RGB_TOPICS, get_sdk_profile


class SdkCommunicator(RosCommunicator):
    def __init__(self, profile, arm_control, control_callbacks, data_acquirer, use_sim_time=False):
        self.profile = get_sdk_profile(profile)
        self.arm_control = arm_control
        self.use_sim_time = use_sim_time
        if set(control_callbacks) != self.profile.subscribe_topics(arm_control):
            raise ValueError("SDK callback topics do not match the selected profile and control mode")
        if not set(data_acquirer) <= self.profile.publish_topics:
            raise ValueError("Unknown SDK publisher topic")
        rates = {
            "default": 50,
            "/odom": 30,
            "/tracked_pose": 30,
            "/hal/chassis/imu": 120,
        }
        rates.update({topic: 30 for topic in self.profile.arm_poses})
        rates.update({topic: 15 for topic in RGB_TOPICS | DEPTH_TOPICS})
        rates["/camera_head_front/depth/image_raw/compressedDepth"] = 5
        rates["/scan"] = 10
        super().__init__(control_callbacks, data_acquirer, rates, f"{self.profile.name}_sdk")

    def _initRobotPublisher(self):
        for topic in self._data_acquirer:
            msg_type = JointState
            if topic in self.profile.arm_poses or topic == "/tracked_pose":
                msg_type = PoseStamped
            elif topic == "/odom":
                msg_type = Odometry
            elif topic == "/hal/chassis/imu":
                msg_type = Imu
            elif topic == "/scan":
                msg_type = LaserScan
            elif topic == "/camera_chassis_front/depth/points":
                msg_type = PointCloud2
            elif topic in RGB_TOPICS | DEPTH_TOPICS:
                msg_type = CompressedImage
            self._ros_publishers[topic] = self.create_publisher(msg_type, topic, 10)
        self._clock_publisher = None
        if self.use_sim_time:
            from rclpy.exceptions import ParameterAlreadyDeclaredException
            from rclpy.parameter import Parameter
            from rosgraph_msgs.msg import Clock

            self._clock_publisher = self.create_publisher(Clock, "/clock", 10)
            try:
                self.declare_parameter("use_sim_time", True)
            except ParameterAlreadyDeclaredException:
                pass
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, True)])

    def _initRobotSubscriber(self):
        for topic, callback in self._control_callbacks.items():
            msg_type = PoseStamped if topic in self.profile.arm_pose_commands else Float64MultiArray
            if topic == "/chassis/cmd_vel":
                msg_type = Twist
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self._ros_subscribers[topic] = self.create_subscription(msg_type, topic, callback, qos)

    def publish_clock(self, sim_time):
        if self._clock_publisher is not None:
            from rosgraph_msgs.msg import Clock

            from maniparena_sim.ros.sim_utils import get_ros_time

            msg = Clock()
            msg.clock = get_ros_time(sim_time)
            self._clock_publisher.publish(msg)

    def reset_subscriptions(self):
        """Drop old DDS readers so queued commands cannot cross a reset boundary."""
        for subscription in self._ros_subscribers.values():
            self.destroy_subscription(subscription)
        self._ros_subscribers.clear()
        self._initRobotSubscriber()
