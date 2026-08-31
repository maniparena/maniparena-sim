"""Zero-dimensional, bounded feed-forward effort action."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils.configclass import configclass


class SafeEffortOverlayAction(ActionTerm):
    """Apply a ramped effort overlay without adding an SDK action channel."""

    cfg: "SafeEffortOverlayActionCfg"

    def __init__(self, cfg: "SafeEffortOverlayActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._elapsed_s = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        joint_ids, _ = self._asset.find_joints(list(cfg.joint_names))
        self._joint_ids = [int(index) for index in joint_ids]

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = actions

    def apply_actions(self) -> None:
        if not self.cfg.enabled or not self._joint_ids:
            return
        self._elapsed_s += max(0.0, float(self._env.cfg.sim.dt))
        scale = (
            torch.clamp(self._elapsed_s / self.cfg.ramp_s, 0.0, 1.0)
            if self.cfg.ramp_s > 0.0
            else torch.ones_like(self._elapsed_s)
        )
        limit = abs(float(self.cfg.max_effort_n))
        effort = (scale * float(self.cfg.effort_n)).clamp(-limit, limit)
        target = self._asset.data.joint_effort_target.clone()
        target[:, self._joint_ids] = effort.to(dtype=target.dtype).reshape(-1, 1)
        self._asset.set_joint_effort_target(target)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        rows = slice(None) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self._elapsed_s[rows] = 0.0
        if self._joint_ids:
            target = self._asset.data.joint_effort_target.clone()
            target[rows, self._joint_ids] = 0.0
            self._asset.set_joint_effort_target(target)


@configclass
class SafeEffortOverlayActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = SafeEffortOverlayAction
    enabled: bool = False
    joint_names: tuple[str, ...] = ()
    effort_n: float = 0.0
    max_effort_n: float = 0.0
    ramp_s: float = 0.0
