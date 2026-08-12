"""Builder for the put_bottle_on_woodshelf task."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg

from maniparena_sim.task.put_bottle_on_woodshelf import (
    PutBottleOnWoodshelfTask,
    PutBottleOnWoodshelfTerminationsCfg,
    build_put_bottle_events_cfg,
)
from maniparena_sim.task.utils import find_background


class PutBottleOnWoodshelfBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        if bg is None:
            raise ValueError("PutBottleOnWoodshelfTask needs a background asset")
        task = PutBottleOnWoodshelfTask(background_scene=bg)
        task.termination_cfg = PutBottleOnWoodshelfTerminationsCfg()
        task.events_cfg = build_put_bottle_events_cfg()
        # Front-left third-person: elevated and pulled back (table, robot, shelf).
        task.viewer_cfg = ViewerCfg(
            eye=(-1.0, 1.8, 1.5),
            lookat=(0.55, -0.85, 0.55),
            origin_type='world',
        )
        return task
