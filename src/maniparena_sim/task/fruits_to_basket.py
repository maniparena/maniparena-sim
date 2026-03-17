"""Put fruits into the basket."""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from maniparena_sim.task.base import TaskCFG
from maniparena_sim.task.utils import find_assets_by_tag, find_background


@configclass
class FruitsToBasketTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING
    any_object_dropped: TerminationTermCfg = MISSING


class FruitsToBasketTask(TaskBase):
    def __init__(self, target_objects: list[ObjectBase], all_objects: list[ObjectBase], basket: ObjectBase, background_scene: Asset, episode_length_s: float | None = None, pose_range: dict | None = None):
        super().__init__(episode_length_s=episode_length_s)
        self.target_objects = target_objects
        self.all_objects = all_objects
        self.basket = basket
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
        return "Pick up the fruits and put them into the pink basket."

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        metrics: list[MetricBase] = [SuccessRateMetric()]
        if self.target_objects:
            metrics.append(ObjectMovedRateMetric(self.target_objects[0]))
        return metrics

    def get_viewer_cfg(self) -> ViewerCfg:
        return self.viewer_cfg if self.viewer_cfg is not None else ViewerCfg()

    @classmethod
    def from_scene(cls, scene, **kwargs):
        targets = find_assets_by_tag(scene, "collect_target")
        containers = find_assets_by_tag(scene, "container")
        bg = find_background(scene)
        basket = containers[0] if containers else None
        if not targets or basket is None or bg is None:
            raise ValueError("FruitsToBasketTask needs collect_target, container and background")
        return cls(target_objects=targets, all_objects=targets, basket=basket, background_scene=bg, **kwargs)


@dataclass
class FruitsToBasketTaskCFG(TaskCFG):
    class_type: type = FruitsToBasketTask
