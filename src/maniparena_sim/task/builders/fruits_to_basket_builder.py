"""Builder for fruits-to-basket runtime configs."""

from __future__ import annotations

import numpy as np
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from maniparena_sim.task.fruits_to_basket import FruitsToBasketTask, FruitsToBasketTerminationsCfg
from maniparena_sim.task.utils import build_default_events_cfg, find_assets_by_tag, find_background
from maniparena_sim.terms.terminations import all_objects_in_basket, any_object_dropped


class FruitsToBasketBuilder:
    DEFAULT_POSE_RANGE = {"x": (0.0, 0.1), "y": (-0.7, 0.15), "z": (-0.15, -0.15), "yaw": (-0.5, 0.5)}

    def finalize(self, task_cfg, scene):
        targets = find_assets_by_tag(scene, "collect_target")
        containers = find_assets_by_tag(scene, "container")
        bg = find_background(scene)
        basket = containers[0] if containers else None
        if not targets or basket is None or bg is None:
            raise ValueError("FruitsToBasketTask needs collect_target, container and background")
        task = FruitsToBasketTask(target_objects=targets, all_objects=targets, basket=basket, background_scene=bg)
        task.events_cfg = build_default_events_cfg(targets, domain_randomization_cfg=getattr(task_cfg, "domain_randomization_cfg", None), pose_range=self.DEFAULT_POSE_RANGE)
        task.termination_cfg = FruitsToBasketTerminationsCfg(
            success=TerminationTermCfg(
                func=all_objects_in_basket,
                params={
                    "object_cfgs": [SceneEntityCfg(obj.name) for obj in targets],
                    "basket_cfg": SceneEntityCfg(basket.name),
                    "x_threshold": 0.14,
                    "y_threshold": 0.14,
                    "z_threshold": 0.12,
                    "velocity_threshold": 0.5,
                },
            ),
            any_object_dropped=TerminationTermCfg(
                func=any_object_dropped,
                params={"object_cfgs": [SceneEntityCfg(obj.name) for obj in targets], "minimum_height": bg.object_min_z},
            ),
        )
        task.viewer_cfg = get_viewer_cfg_look_at_object(lookat_object=targets[0], offset=np.array([-1.0, 0.0, 0.5])) if targets else ViewerCfg()
        return task
