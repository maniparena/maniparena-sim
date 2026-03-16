"""Small math helpers shared across dataset export and robotics glue."""

from __future__ import annotations

import numpy as np


def quat_xyzw_to_euler_xyz(
    quat_xyzw: np.ndarray,
    unwrap: bool = True,
) -> np.ndarray:
    """Convert XYZW quaternions to XYZ Euler angles."""
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    squeeze = quat.ndim == 1
    if squeeze:
        quat = quat[None, :]
    if quat.shape[-1] != 4:
        raise ValueError(
            f'Expected quaternion shape (*, 4), '
            f'got {quat.shape}'
        )

    x, y, z, w = (
        quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],
    )
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(
        sinr_cosp.astype(np.float64),
        cosr_cosp.astype(np.float64),
    ).astype(np.float32)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.where(
        np.abs(sinp) >= 1.0,
        np.sign(sinp) * (np.pi / 2.0),
        np.arcsin(np.clip(sinp, -1.0, 1.0)),
    ).astype(np.float32)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(
        siny_cosp.astype(np.float64),
        cosy_cosp.astype(np.float64),
    ).astype(np.float32)

    euler = np.stack([roll, pitch, yaw], axis=1).astype(np.float32)
    if unwrap:
        euler = np.unwrap(euler, axis=0).astype(np.float32)
    return euler[0] if squeeze else euler


def euler_xyz_to_quat_wxyz(
    euler_xyz: np.ndarray,
) -> np.ndarray:
    """Convert XYZ Euler angles (radians) to WXYZ quaternion."""
    euler = np.asarray(euler_xyz, dtype=np.float32)
    squeeze = euler.ndim == 1
    if squeeze:
        euler = euler[None, :]
    if euler.shape[-1] != 3:
        raise ValueError(
            f'Expected euler shape (*, 3), got {euler.shape}'
        )

    half = euler.astype(np.float64) * 0.5
    cx = np.cos(half[:, 0])
    cy = np.cos(half[:, 1])
    cz = np.cos(half[:, 2])
    sx = np.sin(half[:, 0])
    sy = np.sin(half[:, 1])
    sz = np.sin(half[:, 2])

    quat = np.stack([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ], axis=1).astype(np.float32)
    return quat[0] if squeeze else quat


def to_numpy_keep_shape(value):
    """Best-effort tensor-like to numpy conversion."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)
