"""Builder for the quanta_x1 move_bottle_to_bin task."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from maniparena_sim.task.mobile_layouts import BIN_SUCCESS, MOBILE_RETRACT
from maniparena_sim.task.mobile_pick_place import (
    MobilePickPlaceTerminationsCfg,
    MoveBottleToBinTask,
    build_bin_events_cfg,
)
from maniparena_sim.task.utils import find_assets_by_tag, find_background
from maniparena_sim.terms.terminations import place_then_retract


class MoveBottleToBinBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        bottles = find_assets_by_tag(scene, "bottle")
        bins = find_assets_by_tag(scene, "bin")
        if bg is None or not bottles or not bins:
            raise ValueError("MoveBottleToBinTask needs a background, a bottle and a bin")
        bottle, bin_asset = bottles[0], bins[0]
        seed = (getattr(scene, "scene_options", None) or {}).get("seed")

        task = MoveBottleToBinTask(background_scene=bg)
        task.events_cfg = build_bin_events_cfg(seed=seed)
        terminations = MobilePickPlaceTerminationsCfg()
        # Success: placed in the live footprint (held, latched), then arms home with the object still in.
        terminations.success = TerminationTermCfg(
            func=place_then_retract,
            params={
                "placements": [
                    {
                        "kind": "footprint",
                        "object_cfg": SceneEntityCfg(bottle.name),
                        "reference_cfg": SceneEntityCfg(bin_asset.name),
                        "reference_asset": bin_asset,
                        **BIN_SUCCESS,
                    }
                ],
                **MOBILE_RETRACT,
            },
        )
        task.termination_cfg = terminations
        task.viewer_cfg = ViewerCfg(eye=(-1.0, 1.9, 1.7), lookat=(1.2, -0.2, 0.5), origin_type="world")
        return task
