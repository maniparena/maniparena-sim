"""Isaac Lab official StablePD, with a Tan-2011 fallback on older Lab.

ManaEnv now authors :class:`isaaclab.actuators.StablePDActuatorCfg` and, on
newer Lab, routes it through ``SimulationCfg.use_newton_actuators``. The
Isaac Lab pin in this repo does not export that cfg yet, so we keep the
same config surface and evaluate the look-ahead PD law in Python.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.types import ArticulationActions

try:
    from isaaclab.actuators import StablePDActuatorCfg as _OfficialStablePDActuatorCfg
except ImportError:
    _OfficialStablePDActuatorCfg = None


class _FallbackStablePDActuator(IdealPDActuator):
    """Explicit Stable PD: ``tau = -kp (q + dt qd - q*) - kd (qd - qd*)``."""

    cfg: '_FallbackStablePDActuatorCfg'

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        dt = getattr(self.cfg, 'sim_dt', None)
        try:
            dt = float(dt) if dt is not None else 0.0
        except (TypeError, ValueError):
            dt = 0.0
        if dt <= 0.0:
            dt = 1.0 / 120.0
        q_des = control_action.joint_positions
        qd_des = control_action.joint_velocities
        tau_ff = control_action.joint_efforts
        if qd_des is None:
            qd_des = torch.zeros_like(joint_vel)
        if tau_ff is None:
            tau_ff = torch.zeros_like(joint_vel)
        pred_pos = joint_pos + dt * joint_vel
        error_pos = q_des - pred_pos
        error_vel = qd_des - joint_vel
        self.computed_effort = (
            self.stiffness * error_pos
            + self.damping * error_vel
            + tau_ff
        )
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class _FallbackStablePDActuatorCfg(IdealPDActuatorCfg):
    """Tan 2011 Stable PD when the installed Isaac Lab has no official cfg."""

    class_type: type = _FallbackStablePDActuator
    sim_dt: float | None = None
    gravity_compensation: str = 'none'

if _OfficialStablePDActuatorCfg is not None:
    StablePDActuatorCfg = _OfficialStablePDActuatorCfg
    _USING_OFFICIAL_STABLE_PD = True
else:
    StablePDActuatorCfg = _FallbackStablePDActuatorCfg
    _USING_OFFICIAL_STABLE_PD = False


def using_official_stable_pd() -> bool:
    return _USING_OFFICIAL_STABLE_PD


def prepare_stable_pd_on_env_cfg(env_cfg: Any) -> None:
    """Seed StablePD ``sim_dt`` and opt into Newton actuators when official."""
    sim_cfg = getattr(env_cfg, 'sim', None)
    dt = getattr(sim_cfg, 'dt', None)
    try:
        dt = float(dt) if dt is not None else None
    except (TypeError, ValueError):
        dt = None
    robot_cfg = getattr(getattr(env_cfg, 'scene', None), 'robot', None)
    actuators = getattr(robot_cfg, 'actuators', None)
    disable_gravity = bool(
        getattr(
            getattr(getattr(robot_cfg, 'spawn', None), 'rigid_props', None),
            'disable_gravity',
            False,
        )
    )
    if isinstance(actuators, dict):
        for cfg in actuators.values():
            if hasattr(cfg, 'sim_dt') and dt is not None and dt > 0.0:
                current = getattr(cfg, 'sim_dt', None)
                try:
                    needs = current is None or float(current) <= 0.0
                except (TypeError, ValueError):
                    needs = True
                if needs:
                    cfg.sim_dt = dt
            if disable_gravity and hasattr(cfg, 'gravity_compensation'):
                cfg.gravity_compensation = 'none'
            if disable_gravity and hasattr(cfg, 'use_gravity_compensation'):
                cfg.use_gravity_compensation = False
    if (
        _USING_OFFICIAL_STABLE_PD
        and sim_cfg is not None
        and hasattr(sim_cfg, 'use_newton_actuators')
    ):
        sim_cfg.use_newton_actuators = True
