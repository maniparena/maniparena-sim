"""Differential IK evaluated once per environment command.

Ported from ManaEnv ``manaenv/actions/command_rate_diff_ik.py`` so the
four physics substeps in a 30 Hz control tick reuse one joint target.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.configclass import configclass


class CommandRateDifferentialInverseKinematicsAction(
    DifferentialInverseKinematicsAction,
):
    """Compute DiffIK once per command and reuse the joint target."""

    cfg: 'CommandRateDifferentialInverseKinematicsActionCfg'

    def __init__(
        self,
        cfg: 'CommandRateDifferentialInverseKinematicsActionCfg',
        env,
    ) -> None:
        super().__init__(cfg, env)
        self._cached_joint_pos_des: torch.Tensor | None = None
        self._ik_update_pending = True

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._ik_update_pending = True

    def apply_actions(self) -> None:
        if self._ik_update_pending or self._cached_joint_pos_des is None:
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
            if ee_quat_curr.norm() != 0:
                jacobian = self._compute_frame_jacobian()
                joint_pos_des = self._ik_controller.compute(
                    ee_pos_curr, ee_quat_curr, jacobian, joint_pos,
                )
            else:
                joint_pos_des = joint_pos.clone()
            self._cached_joint_pos_des = joint_pos_des.clone()
            self._ik_update_pending = False
        self._asset.set_joint_position_target_index(
            target=self._cached_joint_pos_des,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._ik_update_pending = True


@configclass
class CommandRateDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg,
):
    """Configuration for command-rate differential inverse kinematics."""

    class_type: type[ActionTerm] = (
        CommandRateDifferentialInverseKinematicsAction
    )
