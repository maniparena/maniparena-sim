"""Coordinate-frame transforms shared by teleop planners.

Stateless world<->base pose conversions plus small tensor-coercion helpers.
Kept dependency-light so any planner can import without pulling in a heavy base.
"""

import importlib
from typing import Any, Tuple

import numpy as np
import torch


def squeeze_if_single(t: torch.Tensor) -> torch.Tensor:
    """``(1, …)`` → ``squeeze(0)``; otherwise return as-is."""
    return t.squeeze(0) if t.shape[0] == 1 else t


def as_quat_tensor(q, device=None) -> torch.Tensor:
    """Coerce quaternion (np / tensor) to float32 tensor (…, 4) XYZW. 1-D -> (1,4)."""
    if isinstance(q, torch.Tensor):
        t = q
    else:
        t = torch.as_tensor(np.asarray(q, dtype=np.float32))
    if device is not None:
        t = t.to(device=device)
    t = t.to(dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


def as_vec3_tensor(v, device=None) -> torch.Tensor:
    """Coerce 3-vector (np / tensor) to float32 tensor (…, 3). 1-D -> (1,3)."""
    if isinstance(v, torch.Tensor):
        t = v
    else:
        t = torch.as_tensor(np.asarray(v, dtype=np.float32))
    if device is not None:
        t = t.to(device=device)
    t = t.to(dtype=torch.float32)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


__all__ = [
    "arm_base_pose_w",
    "base_to_world_frame",
    "world_to_base_frame",
    "as_quat_tensor",
    "as_vec3_tensor",
    "squeeze_if_single",
]


_as_quat = as_quat_tensor
_as_vec3 = as_vec3_tensor
_squeeze1 = squeeze_if_single


def _isaaclab_math_utils() -> Any:
    return importlib.import_module("isaaclab.utils.math")


def world_to_base_frame(
    pos_world: torch.Tensor,
    quat_world: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from world frame to robot base frame."""
    math_utils = _isaaclab_math_utils()
    bq = _as_quat(base_quat)
    base_quat_inv = _squeeze1(math_utils.quat_inv(bq))
    delta = pos_world - base_pos
    v_t = _as_vec3(delta, device=base_quat_inv.device)
    pos_base = _squeeze1(math_utils.quat_apply(base_quat_inv.unsqueeze(0), v_t))
    qw_t = _as_quat(quat_world, device=base_quat_inv.device)
    quat_base = _squeeze1(math_utils.quat_mul(base_quat_inv.unsqueeze(0), qw_t))
    return pos_base, quat_base


def arm_base_pose_w(robot: Any, body_name: str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """World pose ``(pos(3), quat_xyzw(4))`` of the arm-base frame; the articulation root if ``body_name`` is unset."""
    data = robot.data
    if not body_name:
        return data.root_pos_w[0].clone(), data.root_quat_w[0].clone()
    body_ids, _ = robot.find_bodies(body_name)
    pos, quat = data.body_pos_w, data.body_quat_w
    pos, quat = getattr(pos, "torch", pos), getattr(quat, "torch", quat)
    return pos[0, int(body_ids[0])].clone(), quat[0, int(body_ids[0])].clone()


def base_to_world_frame(
    pos_base: torch.Tensor,
    quat_base: torch.Tensor,
    base_pos: torch.Tensor,
    base_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transform pose from robot base frame to world frame.

    Inverse of :func:`world_to_base_frame`.
    """
    math_utils = _isaaclab_math_utils()
    bq = _as_quat(base_quat)
    bp = _as_vec3(base_pos, device=bq.device)
    pb = _as_vec3(pos_base, device=bq.device)
    qb = _as_quat(quat_base, device=bq.device)
    pos_world = bp + _squeeze1(math_utils.quat_apply(bq.unsqueeze(0), pb))
    quat_world = _squeeze1(math_utils.quat_mul(bq.unsqueeze(0), qb.unsqueeze(0)))
    return pos_world, quat_world
