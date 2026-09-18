"""Builder for the empty dummy runtime task (teleop / SDK ROS2 bring-up)."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg

from maniparena_sim.task.dummy_task import DummyTask, DummyTaskTerminationsCfg
from maniparena_sim.task.utils import find_background


class DummyTaskBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        if bg is None:
            raise ValueError("DummyTask needs a background asset")
        task = DummyTask(background_scene=bg)
        task.termination_cfg = DummyTaskTerminationsCfg()
        # Static opening view aimed at the floor-spawned robot; does not follow.
        task.viewer_cfg = ViewerCfg(
            eye=(-2.92, 1.74, 1.6),
            lookat=(-0.52, -0.26, 0.8),
            origin_type='world',
        )
        return task
