"""Vuer/WebXR controller input for non-immersive Pico teleoperation.

This is separate from ``teleop_devices.vr``: that path uses Isaac/OpenXR
through the NVIDIA CloudXR style immersive runtime, while this device only
opens a browser Vuer page and reads WebXR motion-controller pose/buttons.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from scipy.spatial.transform import Rotation

from maniparena_sim.utils.math_utils import quat_wxyz_normalize, quat_wxyz_to_xyzw


def manaprint(*args, **kwargs):
    print(*args, **kwargs)


def _matrix_from_vuer(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.size != 16:
        return None
    return matrix.reshape(4, 4, order="F")


@dataclass
class VuerControllerDeviceCfg(DeviceCfg):
    """Configuration for :class:`VuerControllerDevice`."""

    host: str = "0.0.0.0"
    port: int | None = None
    display_fps: float = 60.0
    apply_xr_anchor: bool = True
    anchor_pose_provider: Callable[[], tuple[np.ndarray, np.ndarray]] | None = None
    gripper_input: str = "trigger"
    include_buttons: bool = True
    include_joysticks: bool = True
    class_type: type[DeviceBase] | None = None

    def __post_init__(self) -> None:
        self.class_type = VuerControllerDevice


class VuerControllerDevice(DeviceBase):
    """Expose Vuer ``MotionControllers`` data in IsaacLab's OpenXR controller format."""

    _DEFAULT_PORT = 8012
    _instances: dict[tuple[str, int], "VuerControllerDevice"] = {}
    _instances_lock = threading.Lock()

    def __new__(cls, cfg: VuerControllerDeviceCfg, retargeters=None):
        key = (str(cfg.host), cls._resolve_port(cfg))
        with cls._instances_lock:
            existing = cls._instances.get(key)
            if existing is not None:
                return existing
            instance = super().__new__(cls)
            cls._instances[key] = instance
            return instance

    @classmethod
    def _resolve_port(cls, cfg: VuerControllerDeviceCfg) -> int:
        return int(cfg.port) if cfg.port is not None else cls._DEFAULT_PORT

    def __init__(self, cfg: VuerControllerDeviceCfg, retargeters=None):
        if getattr(self, "_initialized", False):
            self.cfg = cfg
            if not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run_vuer, daemon=True)
                self._thread.start()
            return
        super().__init__(retargeters=retargeters)
        self.cfg = cfg
        self._callbacks: dict[Any, Callable] = {}
        self._lock = threading.Lock()
        self._left_matrix: np.ndarray | None = None
        self._right_matrix: np.ndarray | None = None
        self._left_inputs = np.zeros(7, dtype=np.float32)
        self._right_inputs = np.zeros(7, dtype=np.float32)
        self._thread = threading.Thread(target=self._run_vuer, daemon=True)
        self._thread.start()
        self._initialized = True

    def reset(self):
        return None

    def add_callback(self, key: Any, func: Callable):
        self._callbacks[key] = func

    def close(self) -> None:
        # Vuer does not expose a stable public stop API across versions. Keep
        # one shared server per host/port alive so scene switches can reuse it
        # instead of trying to bind the same port again.
        return None

    def advance(self) -> torch.Tensor:
        """Return the raw VR planner action without the OpenXR retargeter stack."""
        left = self._controller_array("left")
        right = self._controller_array("right")
        left_pose, left_inputs = self._split_controller_array(left)
        right_pose, right_inputs = self._split_controller_array(right)
        left_gripper = self._gripper_command(left_inputs, side="left")
        right_gripper = self._gripper_command(right_inputs, side="right")
        buttons = np.asarray(
            [
                left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value],
                left_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value],
                right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_0.value],
                right_inputs[DeviceBase.MotionControllerInputIndex.BUTTON_1.value],
            ],
            dtype=np.float32,
        )
        joysticks = np.asarray(
            [
                left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value],
                left_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value],
                right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_X.value],
                right_inputs[DeviceBase.MotionControllerInputIndex.THUMBSTICK_Y.value],
            ],
            dtype=np.float32,
        )
        raw_parts = [
            left_pose,
            np.asarray([left_gripper], dtype=np.float32),
            right_pose,
            np.asarray([right_gripper], dtype=np.float32),
        ]
        if self.cfg.include_buttons:
            raw_parts.append(buttons)
        if self.cfg.include_joysticks:
            raw_parts.append(joysticks)
        raw = np.concatenate(raw_parts)
        action = torch.as_tensor(raw, dtype=torch.float32, device=self.cfg.sim_device)
        return action

    def _get_raw_data(self) -> dict:
        data = {}
        left = self._controller_array("left")
        right = self._controller_array("right")
        if left is not None:
            data[DeviceBase.TrackingTarget.CONTROLLER_LEFT] = left
        if right is not None:
            data[DeviceBase.TrackingTarget.CONTROLLER_RIGHT] = right
        return data

    def _controller_array(self, side: str) -> np.ndarray | None:
        with self._lock:
            matrix = self._left_matrix.copy() if side == "left" and self._left_matrix is not None else None
            if side == "right" and self._right_matrix is not None:
                matrix = self._right_matrix.copy()
            inputs = self._left_inputs.copy() if side == "left" else self._right_inputs.copy()

        if matrix is None:
            return None

        position = matrix[:3, 3].astype(np.float32)
        rotation = Rotation.from_matrix(matrix[:3, :3])
        if self.cfg.apply_xr_anchor and self.cfg.anchor_pose_provider is not None:
            anchor_pos, anchor_quat_wxyz = self.cfg.anchor_pose_provider()
            anchor_rot = Rotation.from_quat(quat_wxyz_to_xyzw(quat_wxyz_normalize(anchor_quat_wxyz)))
            position = (anchor_pos + anchor_rot.apply(position)).astype(np.float32)
            rotation = anchor_rot * rotation
        quat_xyzw = rotation.as_quat()
        pose_row = np.array(
            [position[0], position[1], position[2], quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        return np.stack([pose_row, inputs], axis=0)

    @staticmethod
    def _split_controller_array(controller: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        if controller is None:
            return (
                np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.zeros(7, dtype=np.float32),
            )
        return (
            np.asarray(controller[DeviceBase.MotionControllerDataRowIndex.POSE.value], dtype=np.float32),
            np.asarray(controller[DeviceBase.MotionControllerDataRowIndex.INPUTS.value], dtype=np.float32),
        )

    def _gripper_command(self, inputs: np.ndarray, *, side: str) -> float:
        """Return the analog trigger/squeeze value in [0, 1] (0 = open, 1 = closed)."""
        input_idx = (
            DeviceBase.MotionControllerInputIndex.SQUEEZE.value
            if self.cfg.gripper_input == "squeeze"
            else DeviceBase.MotionControllerInputIndex.TRIGGER.value
        )
        value = float(inputs[input_idx])
        return float(np.clip(value, 0.0, 1.0))

    def _run_vuer(self) -> None:
        try:
            from vuer import Vuer
            from vuer.schemas import MotionControllers
        except Exception as exc:
            manaprint(f"ERROR: Vuer controller input requires the 'vuer' package: {exc}")
            return

        kwargs = {
            "host": self.cfg.host,
            "queries": {"grid": False},
            "queue_len": 3,
        }
        if self.cfg.port is not None:
            kwargs["port"] = int(self.cfg.port)

        try:
            app = Vuer(**kwargs)
        except TypeError:
            kwargs.pop("port", None)
            app = Vuer(**kwargs)

        app.add_handler("CONTROLLER_MOVE")(self._on_controller_move)

        async def main(session):
            session.upsert(
                MotionControllers(stream=True, key="motionControllers", left=True, right=True),
                to="bgChildren",
            )
            while True:
                import asyncio

                await asyncio.sleep(1.0 / max(float(self.cfg.display_fps), 1.0))

        app.spawn(start=False)(main)
        url_port = self._resolve_port(self.cfg)
        manaprint(f"Vuer controller input ready: open https://<this-computer-ip>:{url_port} on Pico.")
        try:
            app.run()
        except Exception as exc:
            manaprint(f"ERROR: Vuer controller input stopped: {exc}")

    async def _on_controller_move(self, event, session, fps=60) -> None:
        del session, fps
        try:
            value = event.value
            left_matrix = _matrix_from_vuer(value.get("left"))
            right_matrix = _matrix_from_vuer(value.get("right"))
            left_inputs = self._inputs_from_state(value.get("leftState", {}))
            right_inputs = self._inputs_from_state(value.get("rightState", {}))
            with self._lock:
                if left_matrix is not None:
                    self._left_matrix = left_matrix
                if right_matrix is not None:
                    self._right_matrix = right_matrix
                self._left_inputs = left_inputs
                self._right_inputs = right_inputs
        except Exception:
            return

    @staticmethod
    def _inputs_from_state(controller_state: dict[str, Any]) -> np.ndarray:
        """Return thumbstick/trigger/grip/A/B in IsaacLab controller layout."""
        thumbstick = controller_state.get("thumbstickValue", [0.0, 0.0])
        thumbstick_x = float(thumbstick[0]) if len(thumbstick) > 0 else 0.0
        thumbstick_y = float(thumbstick[1]) if len(thumbstick) > 1 else 0.0
        return np.array(
            [
                thumbstick_x,
                thumbstick_y,
                float(controller_state.get("triggerValue", 0.0)),
                float(controller_state.get("squeezeValue", 0.0)),
                float(bool(controller_state.get("aButton", False))),
                float(bool(controller_state.get("bButton", False))),
                0.0,
            ],
            dtype=np.float32,
        )
