"""Builder for the quanta_x1 move_fruit_to_basket task."""

from __future__ import annotations

from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

from maniparena_sim.task.mobile_layouts import FRUIT_SUCCESS, MOBILE_RETRACT
from maniparena_sim.task.mobile_pick_place import (
    MobilePickPlaceTerminationsCfg,
    MoveFruitToBasketTask,
    build_fruit_events_cfg,
)
from maniparena_sim.task.utils import find_assets_by_tag, find_background
from maniparena_sim.terms.terminations import place_then_retract


class MoveFruitToBasketBuilder:
    def finalize(self, task_cfg, scene):
        bg = find_background(scene)
        fruits = find_assets_by_tag(scene, "fruit")
        baskets = find_assets_by_tag(scene, "basket")
        if bg is None or not fruits or not baskets:
            raise ValueError("MoveFruitToBasketTask needs a background, a fruit and a basket")
        fruit, basket = fruits[0], baskets[0]
        seed = (getattr(scene, "scene_options", None) or {}).get("seed")

        task = MoveFruitToBasketTask(background_scene=bg)
        task.events_cfg = build_fruit_events_cfg(fruit, basket, seed=seed)
        terminations = MobilePickPlaceTerminationsCfg()
        # Success: placed in the live footprint (held, latched), then arms home with the object still in.
        terminations.success = TerminationTermCfg(
            func=place_then_retract,
            params={
                "placements": [
                    {
                        "kind": "footprint",
                        "object_cfg": SceneEntityCfg(fruit.name),
                        "reference_cfg": SceneEntityCfg(basket.name),
                        "reference_asset": basket,
                        **FRUIT_SUCCESS,
                    }
                ],
                **MOBILE_RETRACT,
            },
        )
        task.termination_cfg = terminations
        task.viewer_cfg = ViewerCfg(eye=(-1.2, 0.0, 2.2), lookat=(1.5, 0.0, 0.5), origin_type="world")
        return task
