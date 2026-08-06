"""Differential-drive keyboard controller config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DifferentialDriveKeyboardControllerCfg:
    """Configuration for differential-drive keyboard control."""

    mode_name: str = "differential"
    linear_velocity: float = 0.5
    angular_velocity: float = 1.0
    wheel_joint_names: tuple[str, str] = (
        "left_wheel_joint",
        "right_wheel_joint",
    )
    wheel_radius: float = 0.084
    wheel_track_width: float = 0.458
    forward_key: str = "W"
    backward_key: str = "S"
    left_key: str = "A"
    right_key: str = "D"
    ccw_key: str = "Q"
    cw_key: str = "E"
    reset_key: str = "R"
    randomize_key: str = "T"
