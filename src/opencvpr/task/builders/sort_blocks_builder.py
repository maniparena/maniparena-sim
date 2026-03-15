"""Builder for sort-blocks runtime configs."""

from __future__ import annotations

import numpy as np
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from opencvpr.task.sort_blocks import SortBlocksTask, SortBlocksTerminationsCfg
from opencvpr.task.utils import build_default_events_cfg, find_assets_by_tag, find_background
from opencvpr.terms.terminations import all_objects_on_destinations, any_object_dropped


class SortBlocksBuilder:
    DEFAULT_POSE_RANGE = {"x": (0.0, 0.1), "y": (-0.7, 0.15), "z": (-0.15, -0.15), "yaw": (-0.5, 0.5)}

    def finalize(self, task_cfg, scene):
        blocks = find_assets_by_tag(scene, "sort_target")
        dests = find_assets_by_tag(scene, "sort_dest")
        bg = find_background(scene)
        if not blocks or not dests or bg is None:
            raise ValueError("SortBlocksTask needs sort_target, sort_dest and background")
        task = SortBlocksTask(blocks=blocks, destinations=dests, background_scene=bg)
        task.events_cfg = build_default_events_cfg(blocks, domain_randomization_cfg=getattr(task_cfg, "domain_randomization_cfg", None), pose_range=self.DEFAULT_POSE_RANGE)
        cfg = SortBlocksTerminationsCfg()
        cfg.success = TerminationTermCfg(
            func=all_objects_on_destinations,
            params={
                "object_cfgs": [SceneEntityCfg(block.name) for block in blocks],
                "destination_cfgs": [SceneEntityCfg(dest.name) for dest in dests],
                "x_threshold": 0.08,
                "y_threshold": 0.12,
                "z_threshold": 0.04,
                "velocity_threshold": 0.5,
            },
        )
        cfg.any_block_dropped = TerminationTermCfg(
            func=any_object_dropped,
            params={"object_cfgs": [SceneEntityCfg(block.name) for block in blocks], "minimum_height": bg.object_min_z},
        )
        task.termination_cfg = cfg
        task.viewer_cfg = get_viewer_cfg_look_at_object(lookat_object=blocks[0], offset=np.array([-1.0, 0.0, 0.5])) if blocks else ViewerCfg()
        return task
