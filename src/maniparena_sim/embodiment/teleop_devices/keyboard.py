"""Shared bimanual SE(3) keyboard teleop device.

One keyboard device for every robot in the runtime (desktop bimanual arm and
ex001 whole-body). Base/chassis control is opt-in: pass a
``DifferentialDriveKeyboardControllerCfg`` (via the embodiment's
``diff_drive_keyboard_controller_cfg``) to enable a third "base" target.

Output:
  - 14-dim arm-only tensor (no ``differential_cfg``):
    [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
     right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]
  - 16-dim arm+base tensor (``differential_cfg`` set):
    [... arm terms ..., left_wheel_velocity, right_wheel_velocity]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device
from scipy.spatial.transform import Rotation

from maniparena_sim.embodiment.teleop_devices.differential_drive_keyboard_controller import (
    DifferentialDriveKeyboardControllerCfg,
)


@dataclass
class BimanualSe3KeyboardCfg:
    pos_sensitivity: float = 0.02
    rot_sensitivity: float = 0.2
    sim_device: str | None = None
    differential_cfg: DifferentialDriveKeyboardControllerCfg | None = None


class BimanualSe3Keyboard:
    """Keyboard controller with shared keymap and active target switching.

    Default active target is the left arm.
    Motion: W/S/A/D/Q/E (arm pos), Z/X/T/G/C/V (arm rot)
    Base mode: W/S (linear), A/D/Q/E (yaw)
    K: toggle active gripper
    B: switch active target
    R: reset device state
    """

    def __init__(self, cfg: BimanualSe3KeyboardCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self._sim_device = cfg.sim_device
        self._differential_cfg = cfg.differential_cfg
        self.enable_base_target = cfg.differential_cfg is not None
        self.base_linear_sensitivity = (
            cfg.differential_cfg.linear_velocity if cfg.differential_cfg is not None else cfg.pos_sensitivity
        )
        self.base_angular_sensitivity = (
            cfg.differential_cfg.angular_velocity if cfg.differential_cfg is not None else cfg.rot_sensitivity
        )

        import carb
        import omni

        self._carb = carb
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=self: obj._on_keyboard_event(event, *args),
        )

        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        self._left_close_gripper = True
        self._right_close_gripper = True
        self._left_delta_pos = np.zeros(3)
        self._left_delta_rot = np.zeros(3)
        self._right_delta_pos = np.zeros(3)
        self._right_delta_rot = np.zeros(3)
        self._base_delta = np.zeros(2)
        self._active_target = "left"
        self._pressed_motion_keys: dict[str, str] = {}

        self._create_key_bindings()

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass
        self._keyboard_sub = None

    def __str__(self) -> str:
        msg = f"Bimanual Keyboard Controller (SE(3)+gripper): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tDefault active target: left\n"
        msg += "\tMove active arm: W/S (x), A/D (y), Q/E (z)\n"
        msg += "\tRotate active arm: Z/X (rx), T/G (ry), C/V (rz)\n"
        if self.enable_base_target:
            msg += "\tMove active base: W/S (linear), A/D or Q/E (yaw)\n"
        msg += "\tToggle active gripper: K\n"
        msg += "\tSwitch active target: B\n"
        msg += "\tReset device state: R\n"
        return msg

    def reset(self) -> None:
        self._left_close_gripper = True
        self._right_close_gripper = True
        self._left_delta_pos = np.zeros(3)
        self._left_delta_rot = np.zeros(3)
        self._right_delta_pos = np.zeros(3)
        self._right_delta_rot = np.zeros(3)
        self._base_delta = np.zeros(2)
        self._active_target = "left"
        self._pressed_motion_keys.clear()

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        left_rot_vec = Rotation.from_euler("XYZ", self._left_delta_rot).as_rotvec()
        right_rot_vec = Rotation.from_euler("XYZ", self._right_delta_rot).as_rotvec()
        left_cmd = np.concatenate(
            [
                self._left_delta_pos,
                left_rot_vec,
                [-1.0 if self._left_close_gripper else 1.0],
            ]
        )
        right_cmd = np.concatenate(
            [
                self._right_delta_pos,
                right_rot_vec,
                [-1.0 if self._right_close_gripper else 1.0],
            ]
        )
        cmd_parts = [left_cmd, right_cmd]
        if self.enable_base_target:
            cmd_parts.append(self._compute_base_wheel_command())
        cmd = np.concatenate(cmd_parts)
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _create_key_bindings(self) -> None:
        ps = self.pos_sensitivity
        rs = self.rot_sensitivity
        self._ACTIVE_ARM_POS = {
            "W": np.asarray([1, 0, 0], dtype=float) * ps,
            "S": np.asarray([-1, 0, 0], dtype=float) * ps,
            "A": np.asarray([0, 1, 0], dtype=float) * ps,
            "D": np.asarray([0, -1, 0], dtype=float) * ps,
            "Q": np.asarray([0, 0, 1], dtype=float) * ps,
            "E": np.asarray([0, 0, -1], dtype=float) * ps,
        }
        self._ACTIVE_ARM_ROT = {
            "Z": np.asarray([1, 0, 0], dtype=float) * rs,
            "X": np.asarray([-1, 0, 0], dtype=float) * rs,
            "T": np.asarray([0, 1, 0], dtype=float) * rs,
            "G": np.asarray([0, -1, 0], dtype=float) * rs,
            "C": np.asarray([0, 0, 1], dtype=float) * rs,
            "V": np.asarray([0, 0, -1], dtype=float) * rs,
        }
        bs = self.base_linear_sensitivity
        brs = self.base_angular_sensitivity
        self._BASE_LINEAR = {
            "W": np.asarray([1.0, 0.0], dtype=float) * bs,
            "S": np.asarray([-1.0, 0.0], dtype=float) * bs,
        }
        self._BASE_ANGULAR = {
            "A": np.asarray([0.0, 1.0], dtype=float) * brs,
            "D": np.asarray([0.0, -1.0], dtype=float) * brs,
            "Q": np.asarray([0.0, 1.0], dtype=float) * brs,
            "E": np.asarray([0.0, -1.0], dtype=float) * brs,
        }

    def _toggle_active_target(self) -> None:
        if self.enable_base_target:
            order = ("left", "right", "base")
        else:
            order = ("left", "right")
        current_idx = order.index(self._active_target)
        self._active_target = order[(current_idx + 1) % len(order)]
        print(f"[INFO] Active target switched to: {self._active_target}")

    def _toggle_active_gripper(self) -> None:
        if self._active_target == "base":
            return
        if self._active_target == "left":
            self._left_close_gripper = not self._left_close_gripper
        else:
            self._right_close_gripper = not self._right_close_gripper

    def _compute_base_wheel_command(self) -> np.ndarray:
        """Convert desired base linear/angular velocity into wheel velocities."""
        if self._differential_cfg is None:
            return np.zeros(2, dtype=float)
        linear_velocity, angular_velocity = self._base_delta
        left_wheel_velocity = (
            linear_velocity - 0.5 * angular_velocity * self._differential_cfg.wheel_track_width
        ) / self._differential_cfg.wheel_radius
        right_wheel_velocity = (
            linear_velocity + 0.5 * angular_velocity * self._differential_cfg.wheel_track_width
        ) / self._differential_cfg.wheel_radius
        return np.asarray([left_wheel_velocity, right_wheel_velocity], dtype=float)
    def _apply_motion_key(self, target: str, key: str, is_press: bool) -> None:
        sign = 1.0 if is_press else -1.0
        if target == "base":
            if key in self._BASE_LINEAR:
                self._base_delta += sign * self._BASE_LINEAR[key]
            elif key in self._BASE_ANGULAR:
                self._base_delta += sign * self._BASE_ANGULAR[key]
            return

        if key in self._ACTIVE_ARM_POS:
            delta = sign * self._ACTIVE_ARM_POS[key]
            if target == "left":
                self._left_delta_pos += delta
            else:
                self._right_delta_pos += delta
        elif key in self._ACTIVE_ARM_ROT:
            delta = sign * self._ACTIVE_ARM_ROT[key]
            if target == "left":
                self._left_delta_rot += delta
            else:
                self._right_delta_rot += delta

    def _is_motion_key(self, key: str, target: str) -> bool:
        if target == "base":
            return self.enable_base_target and (key in self._BASE_LINEAR or key in self._BASE_ANGULAR)
        return key in self._ACTIVE_ARM_POS or key in self._ACTIVE_ARM_ROT

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        carb = self._carb
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "R":
                self.reset()
            elif key == "B":
                self._toggle_active_target()
            elif key == "K":
                self._toggle_active_gripper()
            elif self._is_motion_key(key, self._active_target):
                if key not in self._pressed_motion_keys:
                    self._pressed_motion_keys[key] = self._active_target
                    self._apply_motion_key(self._active_target, key, is_press=True)
            callback = self._additional_callbacks.get(key)
            if callback:
                callback()

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            target = self._pressed_motion_keys.pop(key, None)
            if target is not None:
                self._apply_motion_key(target, key, is_press=False)
        return True


def _get_expected_action_dim(env) -> int | None:
    if hasattr(env, "single_action_space") and hasattr(env.single_action_space, "shape"):
        return env.single_action_space.shape[0]
    if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        return env.action_space.shape[-1]
    return None


def _map_action_dim(action: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if action.shape[-1] == expected_dim:
        return action
    if expected_dim < action.shape[-1]:
        return action[:expected_dim]
    mapped = torch.zeros((expected_dim,), device=action.device, dtype=action.dtype)
    mapped[: action.shape[-1]] = action
    return mapped


# ── Arena TeleopDeviceBase wrapper ────────────────────────────────────────────


@register_device
class BimanualKeyboardTeleopDevice(TeleopDeviceBase):
    """Shared keyboard teleop device registered with the Arena DeviceRegistry."""

    # Distinct from Arena's stock KeyboardCfg (also name="keyboard").
    name = "bimanual_keyboard"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 0.05,
        rot_sensitivity: float = 0.5,
        enable_base_target: bool = False,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.enable_base_target = enable_base_target
        self._keyboard: BimanualSe3Keyboard | None = None

    def get_device_cfg(
        self,
        pipeline_builder=None,
        embodiment: object | None = None,
    ) -> BimanualSe3KeyboardCfg:
        """Arena DeviceRegistry API (Lab3/Arena main)."""
        del pipeline_builder  # unused for keyboard; OpenXR retargeters need it
        differential_cfg = None
        if self.enable_base_target and embodiment is not None:
            differential_cfg = getattr(
                embodiment,
                "diff_drive_keyboard_controller_cfg",
                None,
            )
        return BimanualSe3KeyboardCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
            sim_device=self.sim_device,
            differential_cfg=differential_cfg,
        )

    # Back-compat alias for older call sites / docs.
    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return self.get_device_cfg(embodiment=embodiment)

    def create_controller(
        self,
        device: str | None = None,
        embodiment: object | None = None,
    ) -> BimanualSe3Keyboard:
        """Create the actual keyboard controller."""
        cfg = self.get_device_cfg(embodiment=embodiment)
        cfg.sim_device = device or self.sim_device
        self._keyboard = BimanualSe3Keyboard(cfg)
        return self._keyboard

