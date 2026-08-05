"""Touch three buttons in sequence or any order."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from maniparena_sim.task.base import TaskCFG
from maniparena_sim.task.utils import find_assets_by_tag


@configclass
class ButtonsContactTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING


class ButtonsContactTask(TaskBase):
    def __init__(self, button_1: ObjectBase, button_2: ObjectBase, button_3: ObjectBase, episode_length_s: float | None = None, pose_range: dict | None = None):
        super().__init__(episode_length_s=episode_length_s)
        self.buttons = [button_1, button_2, button_3]
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
        return "Use the robot arms to touch the green, pink, and blue buttons."

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        if self.viewer_cfg is not None:
            return self.viewer_cfg
        return ViewerCfg()

    @classmethod
    def from_scene(cls, scene, **kwargs):
        buttons = find_assets_by_tag(scene, "button")
        if len(buttons) < 3:
            raise ValueError("ButtonsContactTask needs at least 3 buttons")
        return cls(button_1=buttons[0], button_2=buttons[1], button_3=buttons[2], **kwargs)


@dataclass
class ButtonsContactTaskCFG(TaskCFG):
    class_type: type = ButtonsContactTask
