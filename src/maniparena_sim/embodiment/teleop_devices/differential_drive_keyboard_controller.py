# src/maniparena_sim/embodiment/teleop_devices/differential_drive_keyboard_controller.py
"""Differential-drive keyboard controller config (ported from manaenv)."""

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
    wheel_radius: float = 0.078
    wheel_track_width: float = 0.48
    forward_key: str = "W"
    backward_key: str = "S"
    left_key: str = "A"
    right_key: str = "D"
    ccw_key: str = "Q"
    cw_key: str = "E"
    reset_key: str = "R"
    randomize_key: str = "T"


# ex001 wheel geometry (manaenv EX001DiffDriveKeyboardControllerCfg).
EX001_DIFF_DRIVE_KEYBOARD_CFG = DifferentialDriveKeyboardControllerCfg(
    mode_name="ex001_differential",
    linear_velocity=0.5,
    angular_velocity=2.0,
    wheel_joint_names=("left_wheel_joint", "right_wheel_joint"),
    wheel_radius=0.078,
    wheel_track_width=0.48,
)
