"""Base planner interfaces used by collection loops."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import torch


@dataclass
class SettingsCFG:
    max_steps: int = 400
    step_hz: int = 20
    num_success_steps: int = 10


@dataclass
class PlannerCFG:
    class_type: type["PlannerBase"] | None = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlannerBase(ABC):
    Settings: type = SettingsCFG
    device_name: str | None = None

    def __init__(self):
        self.cfg: PlannerCFG | None = None
        self.settings: Any = None
        self.env_idx = 0
        self.step = 0

    def load(self, cfg: PlannerCFG) -> None:
        self.cfg = cfg
        try:
            self.settings = self.Settings(**cfg.raw_data) if cfg.raw_data else self.Settings()
        except TypeError:
            fields = getattr(self.Settings, "__dataclass_fields__", {})
            self.settings = self.Settings(**{key: value for key, value in cfg.raw_data.items() if key in fields})

    @abstractmethod
    def setup(self, env: Any, task: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_actions(self, env: Any, obs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def on_reset(self, env: Any, obs: Dict[str, Any], env_ids: torch.Tensor, env_id_to_success: Dict[int, bool] | None = None) -> None:
        return None

    def initial_plan(self, env: Any, settle_steps: int = 30) -> None:
        return None

    def setup_instances(self, num_envs: int, create_fn, setup_fn) -> None:
        return None

    def handle_planner_done(self, env: Any, grace: int, skip_ids: Set[int] | None = None):
        return []

    def cleanup(self) -> None:
        return None

    def reset(self, success: Optional[bool] = None) -> None:
        self.step = 0


def create_planner(cfg: PlannerCFG) -> PlannerBase:
    if cfg.class_type is None:
        raise ValueError("cfg.class_type required")
    planner = cfg.class_type()
    planner.load(cfg)
    return planner
