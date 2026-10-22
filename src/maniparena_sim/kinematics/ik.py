"""Damped least-squares IK with Armijo backtracking for a serial chain.

Follows manaenv's X2Robot controller (DLS step + Armijo line search), but
iterates to convergence inside one call so the sim can always be driven with
absolute joint-position targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maniparena_sim.kinematics.chain import KinematicChain
from maniparena_sim.kinematics.se3 import rotation_log


@dataclass(frozen=True)
class IkConfig:
    damping: float = 1e-4
    max_iterations: int = 80
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-4
    armijo_alpha_init: float = 1.0
    armijo_c: float = 1e-4
    armijo_beta: float = 0.5
    armijo_alpha_min: float = 1e-3
    # Orientation error is weighted against metres so one radian does not
    # dominate centimetres when the target is out of reach.
    rotation_weight: float = 0.5


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray
    position_error: float
    rotation_error: float
    iterations: int
    converged: bool


def pose_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """6-vector [position; rotation] taking ``current`` to ``target`` in the base frame."""
    return np.concatenate(
        (target[:3, 3] - current[:3, 3], rotation_log(target[:3, :3] @ current[:3, :3].T))
    )


class ChainIkSolver:
    def __init__(self, chain: KinematicChain, config: IkConfig | None = None):
        self.chain = chain
        self.config = config or IkConfig()

    def _cost(self, err: np.ndarray) -> float:
        w = self.config.rotation_weight
        return 0.5 * (err[:3] @ err[:3] + w * w * (err[3:] @ err[3:]))

    def solve(self, target: np.ndarray, seed) -> IkResult:
        """Return joints whose FK reaches ``target`` (4x4, chain-base frame), starting at ``seed``."""
        cfg, chain = self.config, self.chain
        w = np.array([1.0] * 3 + [cfg.rotation_weight] * 3)
        q = chain.clamp(np.asarray(seed, dtype=np.float64).copy())
        T = chain.forward(q)
        err = pose_error(T, target)
        eye = np.eye(chain.dof)
        iterations = 0
        for iterations in range(1, cfg.max_iterations + 1):
            if np.linalg.norm(err[:3]) < cfg.position_tolerance and np.linalg.norm(err[3:]) < cfg.rotation_tolerance:
                iterations -= 1
                break
            _, J = chain.forward_with_jacobian(q)
            Jw = J * w[:, None]
            ew = err * w
            dq = np.linalg.solve(Jw.T @ Jw + cfg.damping * eye, Jw.T @ ew)
            f0 = self._cost(err)
            slope = -float(ew @ (Jw @ dq))
            alpha = cfg.armijo_alpha_init
            accepted = False
            while alpha >= cfg.armijo_alpha_min:
                q_new = chain.clamp(q + alpha * dq)
                err_new = pose_error(chain.forward(q_new), target)
                if self._cost(err_new) <= f0 + cfg.armijo_c * alpha * slope:
                    accepted = True
                    break
                alpha *= cfg.armijo_beta
            if not accepted or np.allclose(q_new, q, atol=1e-12):
                break
            q, err = q_new, err_new
        pos_err, rot_err = float(np.linalg.norm(err[:3])), float(np.linalg.norm(err[3:]))
        return IkResult(
            q=q,
            position_error=pos_err,
            rotation_error=rot_err,
            iterations=iterations,
            converged=pos_err < cfg.position_tolerance * 10 and rot_err < cfg.rotation_tolerance * 10,
        )
