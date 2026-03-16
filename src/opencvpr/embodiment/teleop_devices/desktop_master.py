"""WebSocket master-arm teleop device for Desktop joint control."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class DesktopMasterCfg:
    remote_ip: str = "10.0.0.100"
    remote_port: int = 5555
    sim_device: str | None = None
    reconnect_interval: float = 2.0
    debug: bool = False
    joint_signs: tuple[float, ...] = (1.0, 1.0, -1.0, -1.0, -1.0, -1.0)
    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class DesktopMasterTeleop:
    """Background WebSocket receiver that emits 14D joint actions."""

    def __init__(self, cfg: DesktopMasterCfg):
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._additional_callbacks: dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._connected = False
        self._joint_signs = np.array(cfg.joint_signs, dtype=np.float64)
        self._joint_offsets = np.array(cfg.joint_offsets, dtype=np.float64)
        self._debug_counter = 0
        self._ws = None
        self._should_run = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()
        self._keyboard_sub = None
        self._setup_keyboard_listener()

    def __del__(self) -> None:
        self._should_run = False

    def _recv_loop(self) -> None:
        import websocket

        url = f"ws://{self.cfg.remote_ip}:{self.cfg.remote_port}"
        while self._should_run:
            try:
                ws = websocket.create_connection(url, timeout=5)
                self._ws = ws
                self._connected = True
                while self._should_run:
                    with self._lock:
                        self._latest = json.loads(ws.recv())
            except Exception as exc:
                self._ws = None
                if self._connected:
                    logger.warning("master teleop disconnected: %s", exc)
                self._connected = False
                time.sleep(self.cfg.reconnect_interval)

    def _setup_keyboard_listener(self) -> None:
        try:
            import carb.input
            import omni.appwindow

            appwindow = omni.appwindow.get_default_app_window()
            input_iface = carb.input.acquire_input_interface()
            keyboard = appwindow.get_keyboard()

            def _on_keyboard_event(event, *args, **kwargs) -> bool:
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    callback = self._additional_callbacks.get(event.input.name)
                    if callback is not None:
                        callback()
                return True

            self._keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, _on_keyboard_event)
        except Exception as exc:
            logger.warning("keyboard listener unavailable: %s", exc)

    def reset(self) -> None:
        return None

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def send_signal(self, cmd: str, **kwargs) -> None:
        if self._ws is None or not self._connected:
            return
        try:
            self._ws.send(json.dumps({"cmd": cmd, **kwargs}))
        except Exception as exc:
            logger.warning("send_signal(%s) failed: %s", cmd, exc)

    def advance(self) -> torch.Tensor:
        with self._lock:
            state = self._latest
        if state is None:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        left = state.get("left") or {}
        right = state.get("right") or {}
        left_joints, left_grip = self._extract_joint_state(left)
        right_joints, right_grip = self._extract_joint_state(right)
        left_mapped = self._joint_signs * left_joints + self._joint_offsets
        right_mapped = self._joint_signs * right_joints + self._joint_offsets
        if self.cfg.debug:
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:
                logger.info("master teleop grip L=%.3f R=%.3f", left_grip, right_grip)
        cmd = np.concatenate([left_mapped, [left_grip], right_mapped, [right_grip]])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _extract_joint_state(side: dict) -> tuple[np.ndarray, float]:
        if "joint_pos" not in side:
            return np.zeros(6, dtype=np.float64), 0.0
        jp = np.array(side["joint_pos"], dtype=np.float64)
        return jp[:6], jp[6] if len(jp) > 6 else 0.0
