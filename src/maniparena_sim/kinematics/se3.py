"""Small float64 SE(3) helpers. Quaternions are ROS/Isaac Lab ``xyzw``."""

from __future__ import annotations

import numpy as np


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-24:
        raise ValueError("zero quaternion")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ]
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit ``xyzw`` quaternion with ``w >= 0``."""
    m = np.asarray(R, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0.0 else q


def pose_to_matrix(pos, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat_xyzw)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def matrix_to_pose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, 3].copy(), matrix_to_quat(T[:3, :3])


def invert(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    Rt = T[:3, :3].T
    out[:3, :3] = Rt
    out[:3, 3] = -Rt @ T[:3, 3]
    return out


def axis_rotation(axis: int, angle: float) -> np.ndarray:
    """4x4 rotation about local ``x`` (0), ``y`` (1) or ``z`` (2)."""
    c, s = np.cos(angle), np.sin(angle)
    T = np.eye(4)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    T[i, i], T[i, j], T[j, i], T[j, j] = c, -s, s, c
    return T


def rotation_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of ``R`` (stable near 0 and pi)."""
    cos_t = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return 0.5 * skew
    if np.pi - theta < 1e-4:
        # Near pi the antisymmetric part vanishes; recover the axis from R + I.
        B = (R + np.eye(3)) * 0.5
        k = int(np.argmax(np.diag(B)))
        axis = B[:, k] / np.sqrt(max(B[k, k], 1e-16))
        axis /= np.linalg.norm(axis)
        if axis @ skew < 0.0:
            axis = -axis
        return axis * theta
    return skew * (theta / (2.0 * np.sin(theta)))


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.linalg.norm(rotation_log(Ra.T @ Rb)))
