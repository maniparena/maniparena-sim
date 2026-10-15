"""Isaac-free geometry helpers shared by the quanta_x1 mobile-manipulation tasks.

Everything here is plain ``math`` / ``torch`` so it can be unit-tested without
launching Isaac Sim. Quaternions are XYZW (Isaac Lab root-pose convention).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch


def yaw_to_quat_xyzw(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def quat_xyzw_to_yaw(quat_xyzw: Sequence[float]) -> float:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_xy(yaw_rad: float, x: float, y: float) -> tuple[float, float]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return c * x - s * y, s * x + c * y


def pose_local_xy_to_world(
    origin_xy: Sequence[float], yaw_rad: float, local_x: float, local_y: float
) -> tuple[float, float]:
    """Map a point expressed in a yawed frame (e.g. a table top) to world XY."""
    dx, dy = rotate_xy(yaw_rad, local_x, local_y)
    return float(origin_xy[0]) + dx, float(origin_xy[1]) + dy


def support_local_to_world(
    support: dict, local_x: float, local_y: float, z: float | None = None
) -> tuple[float, float, float]:
    """``support`` is a case-bank style frame: center / x_axis / y_axis (+ object_root_z)."""
    cx, cy, cz = (float(v) for v in support["center"][:3])
    ax, ay = (float(v) for v in support["x_axis"][:2])
    bx, by = (float(v) for v in support["y_axis"][:2])
    wz = float(support.get("object_root_z", cz)) if z is None else float(z)
    return cx + local_x * ax + local_y * bx, cy + local_x * ay + local_y * by, wz


def sample_rect_robot_start(
    rng: random.Random,
    x_range: Sequence[float],
    y_range: Sequence[float],
    face_xy: Sequence[float] | None,
) -> tuple[float, float, float]:
    """Uniform base position in a rectangle; yaw faces ``face_xy`` (random if None).

    The base z stays 0 (floor scene) and yaw = atan2(target - base).
    """
    rx = rng.uniform(float(x_range[0]), float(x_range[1]))
    ry = rng.uniform(float(y_range[0]), float(y_range[1]))
    if face_xy is None:
        yaw = rng.uniform(-math.pi, math.pi)
    else:
        yaw = math.atan2(float(face_xy[1]) - ry, float(face_xy[0]) - rx)
    return rx, ry, yaw


def quat_rotate_xyzw(quat_xyzw: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` (..., 3) by ``quat_xyzw`` (..., 4)."""
    q_xyz = quat_xyzw[..., :3]
    q_w = quat_xyzw[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def quat_conjugate_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    return torch.cat([-quat_xyzw[..., :3], quat_xyzw[..., 3:4]], dim=-1)


def world_aabb_from_local_bbox(
    local_min: torch.Tensor,
    local_max: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat_xyzw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame AABB of a local (object-frame) box under a root pose.

    All inputs are batched over envs: ``(N, 3)`` / ``(N, 4)``. Returns ``(min, max)``.
    """
    lo, hi = local_min, local_max
    corners = torch.stack(
        [
            torch.stack([lo[:, 0], lo[:, 1], lo[:, 2]], dim=-1),
            torch.stack([lo[:, 0], lo[:, 1], hi[:, 2]], dim=-1),
            torch.stack([lo[:, 0], hi[:, 1], lo[:, 2]], dim=-1),
            torch.stack([lo[:, 0], hi[:, 1], hi[:, 2]], dim=-1),
            torch.stack([hi[:, 0], lo[:, 1], lo[:, 2]], dim=-1),
            torch.stack([hi[:, 0], lo[:, 1], hi[:, 2]], dim=-1),
            torch.stack([hi[:, 0], hi[:, 1], lo[:, 2]], dim=-1),
            torch.stack([hi[:, 0], hi[:, 1], hi[:, 2]], dim=-1),
        ],
        dim=1,
    )  # (N, 8, 3)
    quat = root_quat_xyzw.unsqueeze(1).expand(-1, 8, -1)
    world = quat_rotate_xyzw(quat, corners) + root_pos.unsqueeze(1)
    return world.amin(dim=1), world.amax(dim=1)


def in_footprint_zone(
    obj_pos: torch.Tensor,
    ref_min: torch.Tensor,
    ref_max: torch.Tensor,
    *,
    xy_inset_ratio: float,
    z_range: Sequence[float],
    eps: float = 1.0e-4,
) -> torch.Tensor:
    """Object centre inside the reference AABB footprint.

    XY box = reference world AABB shrunk by ``xy_inset_ratio`` of its extent on
    each side. Height is measured above the reference base (``ref_min`` z).
    """
    ext = ref_max[:, :2] - ref_min[:, :2]
    lo = ref_min[:, :2] + xy_inset_ratio * ext - eps
    hi = ref_max[:, :2] - xy_inset_ratio * ext + eps
    xy_ok = torch.all((obj_pos[:, :2] >= lo) & (obj_pos[:, :2] <= hi), dim=-1)
    z_above = obj_pos[:, 2] - ref_min[:, 2]
    z_ok = (z_above >= float(z_range[0])) & (z_above <= float(z_range[1]))
    return xy_ok & z_ok


def in_local_frame_box(
    obj_pos: torch.Tensor,
    ref_pos: torch.Tensor,
    ref_quat_xyzw: torch.Tensor,
    *,
    x_range: Sequence[float],
    y_range: Sequence[float],
    z_range: Sequence[float],
) -> torch.Tensor:
    """Object centre inside a box defined in the reference's own frame."""
    rel = quat_rotate_xyzw(quat_conjugate_xyzw(ref_quat_xyzw), obj_pos - ref_pos)
    return (
        (rel[:, 0] >= float(x_range[0]))
        & (rel[:, 0] <= float(x_range[1]))
        & (rel[:, 1] >= float(y_range[0]))
        & (rel[:, 1] <= float(y_range[1]))
        & (rel[:, 2] >= float(z_range[0]))
        & (rel[:, 2] <= float(z_range[1]))
    )


def upright_dot(obj_quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Dot product of the object's +Z axis with world +Z."""
    z_axis = torch.zeros_like(obj_quat_xyzw[..., :3])
    z_axis[..., 2] = 1.0
    return quat_rotate_xyzw(obj_quat_xyzw, z_axis)[..., 2]
