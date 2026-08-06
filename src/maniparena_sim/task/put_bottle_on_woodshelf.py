"""put_bottle_on_woodshelf task (teleop/eval; random robot/bottle reset)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
import torch
import yaml
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.configclass import make_configclass

from maniparena_sim.task.base import TaskCFG
from maniparena_sim.task.utils import find_background

_CASE_BANK_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "tasks"
    / "put_bottle_on_woodshelf"
    / "case_bank.yaml"
)
_ROBOT_SCENE_KEY = "robot"
_BOTTLE_SCENE_KEY = "put_bottle_on_woodshelf_bottle_s"


def _never_success(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _yaw_deg_to_quat_xyzw(yaw_deg: float) -> list[float]:
    half = math.radians(float(yaw_deg)) * 0.5
    # Lab root pose: xyzw
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _sample_disk(rng: random.Random, radius_m: float) -> tuple[float, float]:
    radius = max(0.0, float(radius_m))
    if radius <= 0.0:
        return 0.0, 0.0
    r = radius * math.sqrt(rng.random())
    theta = 2.0 * math.pi * rng.random()
    return r * math.cos(theta), r * math.sin(theta)


def _load_case_bank(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"case bank must be a mapping: {path}")
    return data


def _sample_robot_bottle_poses(
    rng: random.Random, bank: dict[str, Any]
) -> tuple[list[float], list[float], list[float], list[float], str]:
    axes = bank.get("case_axes") or {}
    robot_ids = list(axes.get("robot_start_ids") or [])
    object_ids = list(axes.get("object_position_ids") or [])
    if not robot_ids or not object_ids:
        raise ValueError("case_bank case_axes needs robot_start_ids and object_position_ids")

    robot_id = str(rng.choice(robot_ids))
    object_id = str(rng.choice(object_ids))
    robot_anchor = (bank.get("robot_start_anchors") or {})[robot_id]
    object_anchor = (bank.get("object_anchors") or {})[object_id]
    support = ((bank.get("support_frames") or {}).get("pick_support") or {})

    robot_pos = [float(v) for v in robot_anchor["position"][:3]]
    while len(robot_pos) < 3:
        robot_pos.append(0.0)
    yaw = float(robot_anchor.get("yaw_deg", 0.0))
    jitter = abs(float(robot_anchor.get("yaw_jitter_deg", 0.0)))
    if jitter > 0.0:
        yaw += rng.uniform(-jitter, jitter)
    dx, dy = _sample_disk(rng, float(robot_anchor.get("radius_m", 0.0)))
    robot_pos[0] += dx
    robot_pos[1] += dy
    robot_quat = _yaw_deg_to_quat_xyzw(yaw)

    center = [float(v) for v in support["center"][:3]]
    x_axis = [float(v) for v in support["x_axis"][:3]]
    y_axis = [float(v) for v in support["y_axis"][:3]]
    root_z = float(support.get("object_root_z", center[2]))
    local_xy = [float(v) for v in object_anchor["local_xy"][:2]]
    odx, ody = _sample_disk(rng, float(object_anchor.get("radius_m", 0.0)))
    local_xy[0] += odx
    local_xy[1] += ody
    bottle_pos = [
        center[0] + local_xy[0] * x_axis[0] + local_xy[1] * y_axis[0],
        center[1] + local_xy[0] * x_axis[1] + local_xy[1] * y_axis[1],
        root_z,
    ]
    bottle_quat = _yaw_deg_to_quat_xyzw(float(object_anchor.get("yaw_deg", 0.0)))
    label = f"r{robot_id}_o{object_id}"
    return robot_pos, robot_quat, bottle_pos, bottle_quat, label


def _write_root_pose(
    asset: Any,
    env: ManagerBasedEnv,
    env_id: int,
    position_xyz: list[float],
    quat_xyzw: list[float],
    *,
    reset_joints_to_default: bool,
) -> None:
    env_ids = torch.as_tensor([int(env_id)], dtype=torch.long, device=env.device)
    pose = torch.zeros(1, 7, dtype=torch.float32, device=env.device)
    pose[0, :3] = torch.as_tensor(position_xyz, dtype=torch.float32, device=env.device)
    pose[0, 3:7] = torch.as_tensor(quat_xyzw, dtype=torch.float32, device=env.device)
    pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    if hasattr(asset, "write_joint_state_to_sim") and hasattr(asset.data, "joint_pos"):
        if reset_joints_to_default and hasattr(asset.data, "default_joint_pos"):
            joint_pos = asset.data.default_joint_pos[env_ids].clone()
        else:
            joint_pos = asset.data.joint_pos[env_ids].clone()
        if joint_pos.numel() > 0:
            asset.write_joint_state_to_sim(
                joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
            )
        _ = asset.data.body_link_pose_w[env_ids]
    asset.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=env.device), env_ids=env_ids
    )


def sample_put_bottle_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    case_bank_path: str | None = None,
    seed: int | None = 3366,
) -> None:
    """Randomize robot + bottle root poses on episode reset."""
    if env_ids is None:
        return
    path = Path(case_bank_path).expanduser() if case_bank_path else _CASE_BANK_PATH
    bank = _load_case_bank(path)
    rng = getattr(env, "_put_bottle_reset_rng", None)
    if rng is None:
        rng = random.Random(None if seed is None else int(seed))
        setattr(env, "_put_bottle_reset_rng", rng)

    for env_id_t in env_ids.to(device=env.device, dtype=torch.long).reshape(-1):
        env_id = int(env_id_t.item())
        robot_pos, robot_quat, bottle_pos, bottle_quat, label = _sample_robot_bottle_poses(
            rng, bank
        )
        _write_root_pose(
            env.scene[_ROBOT_SCENE_KEY],
            env,
            env_id,
            robot_pos,
            robot_quat,
            reset_joints_to_default=True,
        )
        _write_root_pose(
            env.scene[_BOTTLE_SCENE_KEY],
            env,
            env_id,
            bottle_pos,
            bottle_quat,
            reset_joints_to_default=False,
        )
        print(f"[episode] reset case={label}")


def build_put_bottle_events_cfg(case_bank_path: str | Path | None = None) -> Any:
    path = str(case_bank_path or _CASE_BANK_PATH)
    return make_configclass(
        "PutBottleEventsCfg",
        [
            (
                "reset_scene_to_default",
                EventTermCfg,
                EventTermCfg(
                    func=mdp_isaac_lab.reset_scene_to_default,
                    mode="reset",
                    params={"reset_joint_targets": True},
                ),
            ),
            (
                "sample_put_bottle_reset",
                EventTermCfg,
                EventTermCfg(
                    func=sample_put_bottle_reset,
                    mode="reset",
                    params={"case_bank_path": path, "seed": 3366},
                ),
            ),
        ],
    )()


@configclass
class PutBottleOnWoodshelfTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out, time_out=True)
    success: TerminationTermCfg = TerminationTermCfg(func=_never_success)


class PutBottleOnWoodshelfTask(TaskBase):
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
        return "Pick the bottle from the table and place it on the wooden shelf."

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
            raise ValueError("PutBottleOnWoodshelfTask needs a background asset")
        return cls(background_scene=bg, **kwargs)


@dataclass
class PutBottleOnWoodshelfTaskCFG(TaskCFG):
    class_type: type = PutBottleOnWoodshelfTask
