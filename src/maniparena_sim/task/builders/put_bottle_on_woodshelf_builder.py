"""Builder for the put_bottle_on_woodshelf task."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from maniparena_sim.task.put_bottle_on_woodshelf import (
    PutBottleOnWoodshelfTask,
    PutBottleOnWoodshelfTerminationsCfg,
    build_put_bottle_events_cfg,
)
from maniparena_sim.task.mobile_layouts import MOBILE_RETRACT, SHELF_SUCCESS
from maniparena_sim.task.utils import find_background
from maniparena_sim.terms.terminations import place_then_retract


class PutBottleOnWoodshelfBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        if bg is None:
            raise ValueError("PutBottleOnWoodshelfTask needs a background asset")
        task = PutBottleOnWoodshelfTask(background_scene=bg)
        terminations = PutBottleOnWoodshelfTerminationsCfg()
        # Success: bottle upright in the shelf-frame box, then arms home with the bottle still there.
        terminations.success = TerminationTermCfg(
            func=place_then_retract,
            params={
                "placements": [
                    {
                        "kind": "frame_box",
                        "object_cfg": SceneEntityCfg("put_bottle_on_woodshelf_bottle_s"),
                        "reference_cfg": SceneEntityCfg("put_bottle_on_woodshelf_shelf"),
                        **SHELF_SUCCESS,
                    }
                ],
                **MOBILE_RETRACT,
            },
        )
        task.termination_cfg = terminations
        task.events_cfg = build_put_bottle_events_cfg()
        # Front-left third-person: elevated and pulled back (table, robot, shelf).
        task.viewer_cfg = ViewerCfg(
            eye=(-1.0, 1.8, 1.5),
            lookat=(0.55, -0.85, 0.55),
            origin_type='world',
        )
        return task
