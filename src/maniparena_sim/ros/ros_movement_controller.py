"""ROS Movement Controller - Kinematic position control via Twist messages."""

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from maniparena_sim.ros.constants import FLOAT_EPS
from maniparena_sim.ros.math_utils import quat_wxyz_to_yaw, rotate_2d


class RosMovementController:
    """ROS movement controller that processes Twist messages and applies kinematic pose changes.

    Directly writes to robot root pose, suitable for navigation stack position control.
    """

    def __init__(self, robot: Articulation):
        """Initialize the controller.

        Args:
            robot: Isaac Lab Articulation robot instance
        """
        self._robot = robot
        self._device = robot.device

    def apply_twist(self, linear_x: float, linear_y: float, angular_z: float):
        """Apply Twist message to robot pose.

        Args:
            linear_x: Forward displacement (m/frame)
            linear_y: Lateral displacement (m/frame)
            angular_z: Rotation angle (rad/frame)
        """
        position_delta = np.array([linear_x, linear_y, 0.0])
        rotation_delta = angular_z

        if abs(position_delta).sum() > FLOAT_EPS or abs(rotation_delta) > FLOAT_EPS:
            self._apply_kinematic_delta(position_delta, rotation_delta)

    def _apply_kinematic_delta(self, position_delta: np.ndarray, rotation_delta: float):
        """Apply position and rotation delta to robot pose."""
        current_root_state = self._robot.data.root_state_w.clone()
        current_pos = current_root_state[:, 0:3]
        current_quat = current_root_state[:, 3:7]

        if abs(position_delta).sum() > FLOAT_EPS:
            yaw = quat_wxyz_to_yaw(current_quat[0].cpu().numpy())
            rotated = rotate_2d(position_delta, yaw)
            position_delta_tensor = torch.tensor(rotated, dtype=torch.float32, device=self._device).unsqueeze(0)
        else:
            position_delta_tensor = torch.zeros((1, 3), dtype=torch.float32, device=self._device)

        new_pos = current_pos + position_delta_tensor

        # Apply rotation delta
        if abs(rotation_delta) > FLOAT_EPS:
            euler_delta = torch.tensor([0.0, 0.0, rotation_delta], dtype=torch.float32, device=self._device).unsqueeze(
                0
            )
            quat_delta = quat_from_euler_xyz(euler_delta[:, 0], euler_delta[:, 1], euler_delta[:, 2])
            new_quat = quat_mul(quat_delta, current_quat)
        else:
            new_quat = current_quat

        new_root_pose = torch.cat([new_pos, new_quat], dim=1)
        self._robot.write_root_pose_to_sim(new_root_pose)


class RosDifferentialDriveChassisController:
    """ROS chassis controller that converts Twist into differential wheel velocities."""

    def __init__(
        self,
        robot: Articulation,
        wheel_joint_names: tuple,
        wheel_radius: float,
        wheel_track_width: float,
    ):
        self._robot = robot
        self._device = robot.device
        self._wheel_radius = float(wheel_radius)
        self._wheel_track_width = float(wheel_track_width)

        joint_name_to_id = {name: idx for idx, name in enumerate(robot.data.joint_names)}
        missing = [n for n in wheel_joint_names if n not in joint_name_to_id]
        if missing:
            raise ValueError(f"Differential-drive wheel joints not found: {missing}")
        self._wheel_joint_ids = [joint_name_to_id[n] for n in wheel_joint_names]

    def apply_twist(self, linear_x: float, linear_y: float, angular_z: float):
        """Apply Twist as wheel velocities. *linear_y* ignored (non-holonomic)."""
        left_vel = (linear_x - 0.5 * angular_z * self._wheel_track_width) / self._wheel_radius
        right_vel = (linear_x + 0.5 * angular_z * self._wheel_track_width) / self._wheel_radius
        targets = torch.tensor([[left_vel, right_vel]], dtype=torch.float32, device=self._device)
        self._robot.set_joint_velocity_target(targets, joint_ids=self._wheel_joint_ids)
        self._robot.write_data_to_sim()
