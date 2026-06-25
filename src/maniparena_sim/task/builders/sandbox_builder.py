"""Builder for the empty sandbox runtime task."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg

from maniparena_sim.task.sandbox import SandboxTask, SandboxTerminationsCfg
from maniparena_sim.task.utils import find_background


class SandboxBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        if bg is None:
            raise ValueError("SandboxTask needs a background asset")
        task = SandboxTask(background_scene=bg)
        task.termination_cfg = SandboxTerminationsCfg()
        task.viewer_cfg = ViewerCfg()
        return task
