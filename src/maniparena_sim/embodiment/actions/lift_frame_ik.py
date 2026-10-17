"""Differential IK expressed in the arm-base (``lift_link``) frame.

Isaac Lab's ``DifferentialInverseKinematicsAction`` reads the end-effector pose and
Jacobian in the articulation *root* frame. On QUANTA_X1 the arms are mounted on the
lift, so a root-frame target moves with the lift joint and lift motion leaks into arm
control. Using ``lift_link`` as the base frame keeps arms and lift independent.
"""

from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.configclass import configclass


class LiftFrameDiffIKAction(DifferentialInverseKinematicsAction):
    """Diff IK whose command / state frame is ``cfg.base_body_name`` instead of the root."""

    cfg: "LiftFrameDiffIKActionCfg"

    def __init__(self, cfg: "LiftFrameDiffIKActionCfg", env) -> None:
        super().__init__(cfg, env)
        body_ids, body_names = self._asset.find_bodies(cfg.base_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for base_body_name '{cfg.base_body_name}'. "
                f"Found {len(body_ids)}: {body_names}."
            )
        self._base_body_idx = body_ids[0]

    def _base_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._asset.data
        return (
            data.body_pos_w.torch[:, self._base_body_idx],
            data.body_quat_w.torch[:, self._base_body_idx],
        )

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        _, base_quat_w = self._base_pose_w()
        base_rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(base_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_inv, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_inv, jacobian[:, 3:, :])
        return jacobian

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._asset.data
        ee_pos_w = data.body_pos_w.torch[:, self._body_idx]
        ee_quat_w = data.body_quat_w.torch[:, self._body_idx]
        base_pos_w, base_quat_w = self._base_pose_w()
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w,
        )
        if self.cfg.body_offset is not None:
            ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot,
            )
        return ee_pos_b, ee_quat_b


@configclass
class LiftFrameDiffIKActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type = LiftFrameDiffIKAction

    base_body_name: str = "lift_link"
