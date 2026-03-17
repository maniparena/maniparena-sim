"""Keyboard teleop planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from maniparena_sim.embodiment.teleop_devices.keyboard import BimanualSe3Keyboard, BimanualSe3KeyboardCfg, _get_expected_action_dim, _map_action_dim
from maniparena_sim.planners.teleop_base import TeleopPlanner, TeleopSettings


@dataclass
class KeyboardTeleopSettings(TeleopSettings):
    pass


class KeyboardTeleopPlanner(TeleopPlanner):
    Settings = KeyboardTeleopSettings
    device_name = "keyboard"

    def __init__(self):
        super().__init__()
        self._keyboard_controller = None
        self._expected_action_dim: int | None = None

    def _init_device(self, env: Any) -> None:
        cfg = BimanualSe3KeyboardCfg(
            pos_sensitivity=self.settings.position_scale,
            rot_sensitivity=self.settings.rotation_scale,
            sim_device=str(env.device),
        )
        self._keyboard_controller = BimanualSe3Keyboard(cfg)
        self._expected_action_dim = _get_expected_action_dim(env)
        self._keyboard_controller.add_callback("R", lambda: self.signal_done(success=False))
        self._keyboard_controller.add_callback("H", lambda: self.signal_done(success=True))

    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raw = self._keyboard_controller.advance()
        action = _map_action_dim(raw, self._expected_action_dim) if self._expected_action_dim is not None else raw
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        if self._keyboard_controller is not None:
            self._keyboard_controller.reset()
