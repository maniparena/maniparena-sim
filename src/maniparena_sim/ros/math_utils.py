"""Numpy XYZW-quaternion / pose helpers for the SDK ROS2 bridge.

Self-contained: implementations inlined so the ``ros`` subpackage has no
external dependency. Lab 3 / Arena quaternions are XYZW.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_numpy(val: Any) -> np.ndarray:
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    return np.asarray(val)


def to_numpy(val: Any) -> np.ndarray:
    """Convert tensor / array-like to a flattened numpy array."""
    return _as_numpy(val).flatten()


def to_numpy_2d(val: Any) -> np.ndarray | None:
    """Convert tensor / array-like to numpy without changing shape (None-safe)."""
    if val is None:
        return None
    return _as_numpy(val)


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate / inverse (unit quaternion). XYZW convention."""
    q = np.asarray(q, dtype=np.float32)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two XYZW quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by the inverse of an XYZW quaternion."""
    q_inv = quat_inverse(q)
    q_v = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    result = quat_multiply(quat_multiply(q_inv, q_v), q)
    return result[:3]


def quat_xyzw_to_yaw(q: np.ndarray) -> float:
    """Extract yaw (rotation around Z) from a single XYZW quaternion."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# Back-compat alias for older call sites.
quat_wxyz_to_yaw = quat_xyzw_to_yaw


def rotate_2d(vec: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 3-vector around Z by *angle* radians (Z passes through)."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return np.array(
        [vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c, vec[2]],
        dtype=np.float32,
    )


def compute_relative_pose(
    pos: np.ndarray,
    quat: np.ndarray,
    origin_pos: np.ndarray,
    origin_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pose relative to *origin* (numpy, XYZW quaternions)."""
    pos_diff = np.asarray(pos, dtype=np.float32) - np.asarray(origin_pos, dtype=np.float32)
    rel_pos = quat_rotate_inverse(origin_quat, pos_diff)
    rel_quat = quat_multiply(quat_inverse(origin_quat), np.asarray(quat, dtype=np.float32))
    return rel_pos, rel_quat


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by an XYZW quaternion."""
    q = np.asarray(q, dtype=np.float32)
    q_v = np.array([v[0], v[1], v[2], 0.0], dtype=np.float32)
    result = quat_multiply(quat_multiply(q, q_v), quat_inverse(q))
    return result[:3]


def compose_pose(
    origin_pos: np.ndarray,
    origin_quat: np.ndarray,
    rel_pos: np.ndarray,
    rel_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a relative pose on top of *origin* (numpy, XYZW)."""
    world_pos = np.asarray(origin_pos, dtype=np.float32) + quat_rotate(origin_quat, rel_pos)
    world_quat = quat_multiply(np.asarray(origin_quat, dtype=np.float32), np.asarray(rel_quat, dtype=np.float32))
    return world_pos, world_quat


def lift_joint_pose_to_root(
    cmd_pos,
    cmd_quat,
    lift_joint: float,
    rest_pos,
    rest_quat,
    axis=(0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Map an EE pose in ``lift_link`` into root using measured ``lift_joint``.

    ``rest_pos`` / ``rest_quat`` are ``lift_link`` in the root frame at
    ``lift_joint = 0``. The live lift signal is the prismatic joint value.
    """
    axis_n = np.asarray(axis, dtype=np.float32).reshape(3)
    lift_pos, lift_quat = compose_pose(
        rest_pos,
        rest_quat,
        axis_n * float(lift_joint),
        (0.0, 0.0, 0.0, 1.0),
    )
    return compose_pose(lift_pos, lift_quat, cmd_pos, cmd_quat)


def meters_to_mm(meters: float | np.ndarray) -> float | np.ndarray:
    """Convert metres to millimetres for either scalars or depth arrays."""
    return meters * 1000.0
