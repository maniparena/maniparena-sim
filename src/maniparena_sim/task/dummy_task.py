"""Dummy task for open-scene teleop and navigation bring-up.

No objects and no success condition — episodes end manually. Used by both
teleop collection (open-scene practice) and the ROS2 navigation script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from maniparena_sim.task.base import TaskCFG
from maniparena_sim.task.utils import find_background


def _never_success(env):
    """Dummy episodes end manually; success never auto-fires."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


@configclass
class DummyTaskTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = TerminationTermCfg(func=_never_success)


class DummyTask(TaskBase):
    def __init__(self, background_scene, episode_length_s: float | None = None):
        super().__init__(episode_length_s=episode_length_s)
        self.background_scene = background_scene
        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = None
        self.viewer_cfg = None

    def get_scene_cfg(self):
        return self.scene_config

    def get_events_cfg(self):
        return self.events_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_prompt(self) -> str:
        return "Teleoperate or navigate in the open scene."

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return self.viewer_cfg if self.viewer_cfg is not None else ViewerCfg()

    @classmethod
    def from_scene(cls, scene, **kwargs):
        bg = find_background(scene)
        if bg is None:
            raise ValueError("DummyTask needs a background asset")
        return cls(background_scene=bg, **kwargs)


@dataclass
class DummyTaskCFG(TaskCFG):
    class_type: type = DummyTask
