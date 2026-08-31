# If system level rclpy is sourced in bashrc or terminal, it is imported
# otherwise backup rclpy libraries shipped with Isaac Sim is used
from abc import ABC, abstractmethod
from typing import Any, Collection, Iterable

import rclpy
from rclpy.node import Node

from maniparena_sim.utils.debug_print import maniparenaprint
from maniparena_sim.ros.timer import FixedTimeStampTimer


class RosCommunicator(Node, ABC):

    immediate_publish: bool = True
    _CONTROL_SPIN_BUDGET = 8

    _default_sampling_rate = {
        "default": 50,  # in Hz
    }

    @abstractmethod
    def _initRobotPublisher(self):
        """Initialize robot publishers. Must be implemented by subclass."""
        pass

    @abstractmethod
    def _initRobotSubscriber(self):
        """Initialize robot subscribers. Must be implemented by subclass."""
        pass

    def __init__(self, control_callbacks, data_acquirer, sampling_rate=None, node_name="ros_communicator") -> None:
        # Initialize Node parent class
        if not rclpy.ok():
            rclpy.init()
        super().__init__(node_name)
        # Publisher mapping: key (string) -> publisher object
        self._ros_publishers = {}
        self._ros_subscribers = {}
        # Callback mapping: key (topic name) -> callback function
        if control_callbacks is None:
            raise ValueError("callbacks cannot be None")
        if data_acquirer is None:
            raise ValueError("data_acquirer cannot be None")
        self._control_callbacks = control_callbacks
        self._data_acquirer = data_acquirer
        if sampling_rate is None:
            self._sampling_rate = dict(self._default_sampling_rate)
        else:
            self._sampling_rate = dict(sampling_rate)
        self._sim_timers = {}

        self._initRobotPublisher()
        self._initRobotSubscriber()
        self._last_pub_time = {topic: 0.0 for topic in self._ros_publishers.keys()}
        default_rate = self._sampling_rate.get("default")
        self._sim_timers = {
            topic: FixedTimeStampTimer(1.0 / self._sampling_rate.get(topic, default_rate))
            for topic in self._ros_publishers.keys()
        }

    def step_control(self, sim_dt):
        """Drain queued ROS callbacks in the main loop.

        The *sim_dt* argument is currently unused and retained only for API
        compatibility with older call sites and potential subclasses.
        """
        del sim_dt
        # Drain a small bounded batch so bursts of cmd_vel / reset callbacks
        # are serviced within one sim step without blocking indefinitely.
        for _ in range(self._CONTROL_SPIN_BUDGET):
            rclpy.spin_once(self, timeout_sec=0.0)

    def get_publisher_keys(self):
        """Get all publisher keys."""
        return list(self._ros_publishers.keys())

    def get_subscriber_keys(self):
        """Get all subscriber keys."""
        return list(self._ros_subscribers.keys())

    def alloc_data_acquirers(self):
        """Allocate data acquire callbacks."""
        return {key: None for key in self.get_publisher_keys()}

    def alloc_control_callbacks(self):
        """Allocate control callbacks."""
        return {key: None for key in self.get_subscriber_keys()}

    def print_data_acquirer_names(self):
        """Print all publisher keys."""
        keys = self.get_publisher_keys()
        maniparenaprint(f"INFO: Publishers ({len(keys)}): {keys}")

    def print_control_names(self):
        """Print all subscriber keys."""
        keys = self.get_subscriber_keys()
        maniparenaprint(f"INFO: Subscribers ({len(keys)}): {keys}")

    def check_data_aquire_callback(self, data_acquirer):
        """Check if all publishers have corresponding data acquire callbacks."""
        publisher_keys = self.get_publisher_keys()
        missing_keys = [key for key in publisher_keys if key not in data_acquirer or not callable(data_acquirer[key])]
        if missing_keys:
            raise ValueError(f"Missing or invalid data acquire callbacks for publishers: {missing_keys}")
        return True

    def check_control_callback(self, control_callbacks):
        """Check if all subscribers have corresponding control callbacks."""
        subscriber_keys = self.get_subscriber_keys()
        missing_keys = [
            key for key in subscriber_keys if key not in control_callbacks or not callable(control_callbacks[key])
        ]
        if missing_keys:
            raise ValueError(f"Missing or invalid control callbacks for subscribers: {missing_keys}")
        return True

    def step_data_acquire(self, obs, extras, sim_time=1 / 120.0):
        """Step function to acquire data."""
        for topic, pub in self._ros_publishers.items():
            timer = self._sim_timers.get(topic)
            if timer is None:
                continue
            steps = timer.step(sim_time)
            if steps == 0:
                continue

            acquirer = self._data_acquirer.get(topic, None)
            if acquirer is None or not callable(acquirer):
                continue
            msg = acquirer(obs, extras)
            if msg is None:
                continue
            if self.immediate_publish:
                pub.publish(msg)
            else:
                pass

    def publish_topics(self, topics: Iterable[str], obs: Any, extras: Any) -> None:
        """Publish an explicit set of topics immediately.

        Caller owns the scheduling and timestamp choice. This is used by the
        ROS bridge fast/slow paths once timer gating has been resolved.
        """
        for topic in topics:
            pub = self._ros_publishers.get(topic)
            if pub is None:
                continue
            acquirer = self._data_acquirer.get(topic)
            if acquirer is None or not callable(acquirer):
                continue
            msg = acquirer(obs, extras)
            if msg is None:
                continue
            pub.publish(msg)

    def collect_due_topics(self, sim_time: float, topic_filter: Collection[str] | None = None) -> list[str]:
        """Return topics whose sim-time timers are due for publishing."""
        due_topics = []
        for topic in self._ros_publishers.keys():
            if topic_filter is not None and topic not in topic_filter:
                continue
            timer = self._sim_timers.get(topic)
            if timer is None:
                continue
            if timer.step(sim_time) > 0:
                due_topics.append(topic)
        return due_topics

    def step_publish(self, sim_dt):
        """Step function to publish data."""
        pass

    def shutdown(self):
        """Shutdown ROS2 publishers."""
        self.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
            maniparenaprint("INFO: ROS2 Bridge publishers shutdown")
