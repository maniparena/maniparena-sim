"""quanta_x1 mobile pick-and-place tasks.

* ``move_bottle_to_bin``      table -> bin (bottle pool)
* ``move_fruit_to_basket``    fruit table -> basket table (fruit / basket pools)

Both start the floor-standing robot at a random base pose (uniform in a rectangle, yaw
facing the pickable) and share one task class. Scene entity names are fixed by ``scene_builder`` (``robot``, ``bottle``,
``bin``, ``fruit``, ``basket``) whichever pool member was spawned.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

from maniparena_sim.task.base import TaskCFG
from maniparena_sim.task.mobile_events import env_rng, write_root_pose
from maniparena_sim.task.mobile_geometry import (
    pose_local_xy_to_world,
    sample_rect_robot_start,
    support_local_to_world,
    yaw_to_quat_xyzw,
)
from maniparena_sim.task.mobile_layouts import (
    BASKET_TABLE_POSE,
    BIN_RESET,
    FRUIT_RESET,
    FRUIT_TABLE_POSE,
    TABLETOP_LOCAL_Z,
)
from maniparena_sim.task.utils import find_background

ROBOT_KEY = "robot"
IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]


def sample_bin_task_reset(
    env: ManagerBasedEnv, env_ids: torch.Tensor, *, seed: int | None = None
) -> None:
    """Per-episode layout of ``move_bottle_to_bin``: bottle on the table, bin on the floor, robot facing the bottle."""
    if env_ids is None:
        return
    cfg = BIN_RESET
    rng = env_rng(env, "_bin_task_reset_rng", seed)
    for env_id in env_ids.to(device=env.device, dtype=torch.long).reshape(-1).tolist():
        local_x = rng.uniform(*cfg["bottle_local_x"])
        local_y = rng.uniform(*cfg["bottle_local_y"])
        bottle_xyz = support_local_to_world(cfg["support_frame"], local_x, local_y)
        robot_x, robot_y, robot_yaw = sample_rect_robot_start(
            rng, cfg["robot_start_x"], cfg["robot_start_y"], face_xy=bottle_xyz[:2]
        )
        bin_xyz = [rng.uniform(*cfg["bin_x"]), rng.uniform(*cfg["bin_y"]), cfg["bin_z"]]
        write_root_pose(
            env.scene[ROBOT_KEY], env, env_id, [robot_x, robot_y, 0.0], yaw_to_quat_xyzw(robot_yaw),
            reset_joints_to_default=True,
        )
        write_root_pose(env.scene["bottle"], env, env_id, list(bottle_xyz), IDENTITY_QUAT)
        write_root_pose(env.scene["bin"], env, env_id, bin_xyz, list(cfg["bin_quat_xyzw"]))


def sample_fruit_task_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    fruit_asset: Any,
    basket_asset: Any,
    seed: int | None = None,
) -> None:
    """Per-episode layout of ``move_fruit_to_basket``: fruit and basket on their tables, robot facing the fruit."""
    if env_ids is None:
        return
    cfg = FRUIT_RESET
    rng = env_rng(env, "_fruit_task_reset_rng", seed)
    fruit_root_z = TABLETOP_LOCAL_Z - float(fruit_asset.get_bounding_box().min_point[0, 2]) + cfg["root_clearance_m"]
    basket_root_z = TABLETOP_LOCAL_Z - float(basket_asset.get_bounding_box().min_point[0, 2]) + cfg["root_clearance_m"]
    yaw = cfg["table_yaw_rad"]
    for env_id in env_ids.to(device=env.device, dtype=torch.long).reshape(-1).tolist():
        fruit_xy = pose_local_xy_to_world(
            FRUIT_TABLE_POSE[0][:2], yaw, rng.uniform(*cfg["fruit_local_x"]), rng.uniform(*cfg["fruit_local_y"])
        )
        basket_xy = pose_local_xy_to_world(
            BASKET_TABLE_POSE[0][:2], yaw, rng.uniform(*cfg["basket_local_x"]), rng.uniform(*cfg["basket_local_y"])
        )
        robot_x, robot_y, robot_yaw = sample_rect_robot_start(
            rng, cfg["robot_start_x"], cfg["robot_start_y"], face_xy=fruit_xy
        )
        write_root_pose(
            env.scene[ROBOT_KEY], env, env_id, [robot_x, robot_y, 0.0], yaw_to_quat_xyzw(robot_yaw),
            reset_joints_to_default=True,
        )
        write_root_pose(env.scene["fruit"], env, env_id, [*fruit_xy, fruit_root_z], IDENTITY_QUAT)
        write_root_pose(env.scene["basket"], env, env_id, [*basket_xy, basket_root_z], IDENTITY_QUAT)


def _events_cfg(class_name: str, func, params: dict) -> Any:
    return make_configclass(
        class_name,
        [
            (
                "reset_scene_to_default",
                EventTermCfg,
                EventTermCfg(
                    func=mdp_isaac_lab.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
                ),
            ),
            ("sample_layout", EventTermCfg, EventTermCfg(func=func, mode="reset", params=params)),
        ],
    )()


def build_bin_events_cfg(seed: int | None = None) -> Any:
    return _events_cfg("MoveBottleToBinEventsCfg", sample_bin_task_reset, {"seed": seed})


def build_fruit_events_cfg(fruit_asset: Any, basket_asset: Any, seed: int | None = None) -> Any:
    return _events_cfg(
        "MoveFruitToBasketEventsCfg",
        sample_fruit_task_reset,
        {"fruit_asset": fruit_asset, "basket_asset": basket_asset, "seed": seed},
    )


@configclass
class MobilePickPlaceTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = MISSING


class MobilePickPlaceTask(TaskBase):
    PROMPT = ""

    def __init__(self, background_scene, episode_length_s: float | None = None):
        super().__init__(episode_length_s=episode_length_s)
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
        return self.PROMPT

    def get_mimic_env_cfg(self, embodiment_name: str) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return self.viewer_cfg if self.viewer_cfg is not None else ViewerCfg()

    @classmethod
    def from_scene(cls, scene, **kwargs):
        bg = find_background(scene)
        if bg is None:
            raise ValueError(f"{cls.__name__} needs a background asset")
        return cls(background_scene=bg, **kwargs)


class MoveBottleToBinTask(MobilePickPlaceTask):
    PROMPT = (
        "Navigate to the source table, grasp the correct bottle on the table, navigate to the "
        "destination bin, place the bottle in the bin, and retract the arm safely."
    )


class MoveFruitToBasketTask(MobilePickPlaceTask):
    PROMPT = (
        "Navigate to the source fruit table, grasp the correct fruit on the table, navigate to the "
        "destination basket table, place the fruit in the basket, and retract the arm safely."
    )


@dataclass
class MoveBottleToBinTaskCFG(TaskCFG):
    class_type: type = MoveBottleToBinTask


@dataclass
class MoveFruitToBasketTaskCFG(TaskCFG):
    class_type: type = MoveFruitToBasketTask
