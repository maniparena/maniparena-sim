"""Teleop planner base with task-success detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from maniparena_sim.planners.planner_base import PlannerBase, SettingsCFG


@dataclass
class TeleopSettings(SettingsCFG):
    max_steps: int | None = None
    position_scale: float = 0.05
    rotation_scale: float = 0.5


class TeleopPlanner(PlannerBase):
    Settings = TeleopSettings
    device_name = "teleop"

    def __init__(self):
        super().__init__()
        self.last_action: np.ndarray | None = None
        self.done_signal = False
        self.success_signal = False
        self._torch_device: torch.device | None = None
        self._success_term = None

    def setup(self, env: Any, task: Any) -> bool:
        self._success_term = task.get_termination_cfg().success
        self._init_device(env)
        return True

    def prepare_episode(self, gym_env: Any, obs: Dict[str, Any]) -> None:
        self._torch_device = gym_env.device

    def _step_single(self, gym_env: Any, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.step += 1
        self._check_task_success(gym_env)
        return self._get_device_action(obs)

    def signal_done(self, success: bool = False) -> None:
        self.done_signal = True
        self.success_signal = success

    def reset(self, success: Optional[bool] = None) -> None:
        super().reset(success=success)
        self.last_action = None
        self.done_signal = False
        self.success_signal = False
        self._reset_device()

    def _init_device(self, env: Any) -> None:
        return None

    def _get_device_action(self, obs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.last_action is None:
            self.last_action = np.zeros(8)
        device = self._resolve_torch_device(obs)
        action = torch.as_tensor(self.last_action, dtype=torch.float32, device=device).clone()
        return action, {"device": self.device_name}

    def _reset_device(self) -> None:
        return None

    def _check_task_success(self, gym_env: Any) -> None:
        if self.done_signal or gym_env is None or self._success_term is None:
            return
        result = self._success_term.func(gym_env, **self._success_term.params)
        if result.any():
            self.success_signal = True
            self.done_signal = True

    def get_actions(self, env: Any, obs: Dict[str, Any]) -> torch.Tensor:
        result, _ = self._step_single(env, obs)
        if result is None:
            act_dim = env.action_manager.action.shape[-1]
            return torch.zeros(1, act_dim, device=env.device)
        action = result.to(device=env.device, dtype=torch.float32)
        return action.unsqueeze(0) if action.ndim == 1 else action

    def on_reset(self, env, obs, env_ids, env_id_to_success=None):
        self.reset()
        self.prepare_episode(env, obs)

    def _resolve_torch_device(self, obs: Dict[str, Any]) -> torch.device:
        if self._torch_device is not None:
            return self._torch_device
        for value in (obs or {}).values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")
