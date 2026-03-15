"""Bimanual keyboard teleop device for EX001-6R."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase
from scipy.spatial.transform import Rotation


@dataclass
class BimanualSe3KeyboardCfg:
    pos_sensitivity: float = 0.02
    rot_sensitivity: float = 0.2
    sim_device: str | None = None


class BimanualSe3Keyboard:
    """Shared-keymap bimanual SE(3) keyboard controller."""

    def __init__(self, cfg: BimanualSe3KeyboardCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self._sim_device = cfg.sim_device

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
        self.reset()
        self._create_key_bindings()

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass

    def reset(self) -> None:
        self._left_close_gripper = True
        self._right_close_gripper = True
        self._left_delta_pos = np.zeros(3)
        self._left_delta_rot = np.zeros(3)
        self._right_delta_pos = np.zeros(3)
        self._right_delta_rot = np.zeros(3)
        self._active_arm = "left"
        self._pressed_motion_keys: dict[str, str] = {}

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        left_rot_vec = Rotation.from_euler("XYZ", self._left_delta_rot).as_rotvec()
        right_rot_vec = Rotation.from_euler("XYZ", self._right_delta_rot).as_rotvec()
        left_cmd = np.concatenate([self._left_delta_pos, left_rot_vec, [-1.0 if self._left_close_gripper else 1.0]])
        right_cmd = np.concatenate([self._right_delta_pos, right_rot_vec, [-1.0 if self._right_close_gripper else 1.0]])
        return torch.tensor(np.concatenate([left_cmd, right_cmd]), dtype=torch.float32, device=self._sim_device)

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

    def _toggle_active_arm(self) -> None:
        self._active_arm = "right" if self._active_arm == "left" else "left"

    def _toggle_active_gripper(self) -> None:
        if self._active_arm == "left":
            self._left_close_gripper = not self._left_close_gripper
        else:
            self._right_close_gripper = not self._right_close_gripper

    def _apply_motion_key(self, arm: str, key: str, is_press: bool) -> None:
        sign = 1.0 if is_press else -1.0
        source = self._ACTIVE_ARM_POS if key in self._ACTIVE_ARM_POS else self._ACTIVE_ARM_ROT
        delta = sign * source[key]
        if arm == "left":
            if key in self._ACTIVE_ARM_POS:
                self._left_delta_pos += delta
            else:
                self._left_delta_rot += delta
        else:
            if key in self._ACTIVE_ARM_POS:
                self._right_delta_pos += delta
            else:
                self._right_delta_rot += delta

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        carb = self._carb
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "R":
                self.reset()
            elif key == "B":
                self._toggle_active_arm()
            elif key == "K":
                self._toggle_active_gripper()
            elif key in self._ACTIVE_ARM_POS or key in self._ACTIVE_ARM_ROT:
                if key not in self._pressed_motion_keys:
                    self._pressed_motion_keys[key] = self._active_arm
                    self._apply_motion_key(self._active_arm, key, is_press=True)
            callback = self._additional_callbacks.get(key)
            if callback:
                callback()
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            arm = self._pressed_motion_keys.pop(key, None)
            if arm is not None:
                self._apply_motion_key(arm, key, is_press=False)
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


@register_device
class BimanualKeyboardTeleopDevice(TeleopDeviceBase):
    """Arena-compatible wrapper around the keyboard controller."""

    name = "bimanual_keyboard"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.05, rot_sensitivity: float = 0.5):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_teleop_device_cfg(self, embodiment: object | None = None):
        return None

    def create_controller(self, device: str | None = None) -> BimanualSe3Keyboard:
        return BimanualSe3Keyboard(
            BimanualSe3KeyboardCfg(
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
                sim_device=device or self.sim_device,
            )
        )
