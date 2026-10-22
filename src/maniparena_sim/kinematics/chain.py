"""Serial revolute chain: forward kinematics and base-frame Jacobian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maniparena_sim.kinematics.se3 import axis_rotation


@dataclass(frozen=True)
class KinematicChain:
    """``T_end = K[0] Rot(a1, q1) K[1] ... Rot(an, qn) K[n]`` (all relative to the chain base).

    ``K[i]`` fold every fixed joint and joint-frame offset between two revolute
    joints, so only the ``n`` actuated joints appear explicitly.
    """

    joint_names: tuple[str, ...]
    constants: tuple[np.ndarray, ...]
    axes: tuple[int, ...]
    lower: np.ndarray
    upper: np.ndarray
    base_body: str = ""
    end_body: str = ""

    def __post_init__(self):
        n = len(self.joint_names)
        if len(self.constants) != n + 1 or len(self.axes) != n:
            raise ValueError("chain needs n axes and n + 1 constant transforms")
        if self.lower.shape != (n,) or self.upper.shape != (n,):
            raise ValueError("joint limits must have one value per joint")

    @property
    def dof(self) -> int:
        return len(self.axes)

    def forward(self, q) -> np.ndarray:
        T = self.constants[0].copy()
        for i, axis in enumerate(self.axes):
            T = T @ axis_rotation(axis, float(q[i])) @ self.constants[i + 1]
        return T

    def forward_with_jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(T_end, J)`` where ``J`` (6 x n) is [linear; angular] in the chain base frame."""
        T = self.constants[0].copy()
        origins, zs = [], []
        for i, axis in enumerate(self.axes):
            T = T @ axis_rotation(axis, float(q[i]))
            origins.append(T[:3, 3].copy())
            zs.append(T[:3, axis].copy())
            T = T @ self.constants[i + 1]
        J = np.zeros((6, self.dof))
        p_end = T[:3, 3]
        for i in range(self.dof):
            J[:3, i] = np.cross(zs[i], p_end - origins[i])
            J[3:, i] = zs[i]
        return T, J

    def clamp(self, q) -> np.ndarray:
        return np.minimum(np.maximum(q, self.lower), self.upper)
