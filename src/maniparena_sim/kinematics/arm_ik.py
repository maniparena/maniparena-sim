"""Cartesian command -> joint-position targets for one or more arms.

Every SDK robot drives the simulator with joint positions only. In EE mode this
module turns a flange pose command into those joint targets.

The command and the measured flange pose are expressed in a *command frame*
(``base_link``, ``lift_link`` ...) that is rigidly tied to the IK chain base up
to the state of any joints in between (waist, lift). Because the measured flange
pose is ``M = T_cb F(q)``, the chain base in the command frame is recovered from
the sim itself as ``T_cb = M F(q)^-1``, so no static USD-to-frame calibration
is needed and a moving waist or lift is absorbed every control tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maniparena_sim.kinematics.chain import KinematicChain
from maniparena_sim.kinematics.ik import ChainIkSolver, IkConfig, IkResult
from maniparena_sim.kinematics.se3 import invert, pose_to_matrix


@dataclass(frozen=True)
class ArmSpec:
    joint_names: tuple[str, ...]
    end_body: str


class ArmIkController:
    def __init__(self, chains: list[KinematicChain], config: IkConfig | None = None):
        self.chains = list(chains)
        self.solvers = [ChainIkSolver(chain, config) for chain in self.chains]
        self._last: list[np.ndarray | None] = [None] * len(self.chains)

    @classmethod
    def from_usd(cls, usd_path: str, arms: list[ArmSpec], config: IkConfig | None = None) -> ArmIkController:
        from maniparena_sim.kinematics.usd_chain import extract_chain

        return cls([extract_chain(usd_path, arm.joint_names, arm.end_body) for arm in arms], config)

    def reset(self) -> None:
        """Forget warm starts (call after a sim reset or a mode switch)."""
        self._last = [None] * len(self.chains)

    def solve(
        self,
        arm: int,
        command_pos,
        command_quat_xyzw,
        measured_pos,
        measured_quat_xyzw,
        joint_pos,
    ) -> IkResult:
        """Joint targets that bring the flange to the command.

        ``joint_pos`` is the measured arm joints (chain order). All poses share
        the command frame; quaternions are ``xyzw``.
        """
        chain, solver = self.chains[arm], self.solvers[arm]
        q_meas = np.asarray(joint_pos, dtype=np.float64)
        measured = pose_to_matrix(measured_pos, measured_quat_xyzw)
        command = pose_to_matrix(command_pos, command_quat_xyzw)
        base_in_frame = measured @ invert(chain.forward(q_meas))
        target = invert(base_in_frame) @ command

        seed = self._last[arm]
        result = solver.solve(target, q_meas if seed is None else seed)
        if not result.converged and seed is not None:
            retry = solver.solve(target, q_meas)
            if retry.position_error + retry.rotation_error < result.position_error + result.rotation_error:
                result = retry
        self._last[arm] = result.q
        return result
